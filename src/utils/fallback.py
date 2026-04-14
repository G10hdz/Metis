"""Fallback chain: Ollama → Opencode Zen API → Qwen → Gemini API → Telegram interrupt.

Tiers:
  1. Ollama local (langchain-ollama, keep_alive=0s, 30s timeout)
  2. Opencode Zen HTTP API (OpenAI-compatible, free models, 45s timeout)
  3. Qwen headless (qwen -p, subprocess, 45s timeout)
  4. Gemini HTTP API (free tier, quota-guard via SQLite, 30s timeout)
  5. Telegram interrupt — ask user permission to try Copilot TUI

All failures logged to ergane-logs.md (ES format).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.config import ollama as ollama_cfg
from src.telemetry.store import get_telemetry

logger = logging.getLogger(__name__)

# --- Tier names ---
TIER_OLLAMA = "ollama"
TIER_OPENCODE = "opencode"
TIER_QWEN = "qwen"
TIER_GEMINI = "gemini"
TIER_TELEGRAM = "telegram"

# --- Per-model timeout mapping (seconds) ---
MODEL_TIMEOUTS = {
    "phi3:mini": 45,
    "phi3:latest": 45,
    "qwen2.5-coder:7b": 90,
    "qwen2.5-coder:7b-q4_K_M": 90,
    "qwen2.5:14b": 120,
    "deepseek-r1:14b": 120,
    "nomic-embed-text": 30,
    "nomic-embed-text:latest": 30,
}
DEFAULT_MODEL_TIMEOUT = 60  # fallback for unknown models

# --- Error types ---
VRAM_ERROR_PATTERNS = (
    "cuda out of memory",
    "cuda",
    "out of memory",
    "rocm",
    "oom",
    "memory allocation",
    "not enough memory",
    "memory pool",
    "failed to allocate",
    "memory exhausted",
)

# --- Gemini quota guard ---
GEMINI_MAX_PER_HOUR = 20  # configurable buffer before quota hits


class FallbackError(Exception):
    """Raised when all fallback tiers are exhausted."""
    pass


class VRAMError(Exception):
    """Ollama ran out of VRAM. Immediate fallback trigger."""
    pass


class TierTimeout(Exception):
    """A tier exceeded its timeout."""
    pass


# ---- logging helpers ----

def _log_fallback(
    from_tier: str,
    to_tier: str,
    query: str,
    reason: str,
    elapsed: float,
) -> None:
    """Append fallback event to ergane-logs.md (ES)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"[{ts}] **FALLBACK**: `{from_tier}` → `{to_tier}`\n"
        f"  - **query**: \"{query[:120]}\"\n"
        f"  - **razón**: {reason}\n"
        f"  - **tiempo**: {elapsed:.1f}s\n\n"
    )
    try:
        with open(settings.ERGANE_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        logger.warning("Could not write ergane log: %s", exc)


def _is_vram_error(exc: Exception) -> bool:
    """Check if an exception message indicates a VRAM/OOM error."""
    msg = str(exc).lower()
    return any(p in msg for p in VRAM_ERROR_PATTERNS)


# ---- Tier implementations ----

def _get_model_timeout(model: str) -> int:
    """Get timeout for a specific model from the MODEL_TIMEOUTS mapping."""
    return MODEL_TIMEOUTS.get(model, DEFAULT_MODEL_TIMEOUT)


def _tier_ollama(query: str, model: str, timeout_s: int) -> str:
    """
    Tier 1: Ollama local via langchain-ollama with signal timeout.
    Raises TierTimeout on timeout, VRAMError on OOM.
    """
    # Use signal.alarm for sync timeout
    def _handler(signum, frame):
        raise TierTimeout(f"Ollama timed out after {timeout_s}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)

    try:
        llm = ollama_cfg.get_chat_model(model)
        result = llm.invoke([
            ollama_cfg.ChatOllama.__module__.split(".")[0]
            and __import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(
                content=query
            )
        ])
        signal.alarm(0)  # cancel
        text = result.content if isinstance(result.content, str) else str(result.content)
        return text
    except TierTimeout:
        raise
    except Exception as exc:
        signal.alarm(0)
        if _is_vram_error(exc):
            raise VRAMError(f"Ollama VRAM error: {exc}") from exc
        raise


def _tier_opencode(query: str, timeout_s: int) -> str:
    """
    Tier 2: Opencode Zen HTTP API (OpenAI-compatible).
    Falls back to CLI if no API key configured.
    Returns response text.
    """
    import httpx

    api_key = settings.OPENCODE_ZEN_API_KEY
    if not api_key:
        logger.warning("No OPENCODE_ZEN_API_KEY, falling back to CLI")
        return _tier_opencode_cli(query, timeout_s)

    for model in (settings.OPENCODE_MODEL, settings.OPENCODE_ALT_MODEL):
        try:
            r = httpx.post(
                f"{settings.OPENCODE_ZEN_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 4096,
                },
                timeout=timeout_s,
            )
            if r.status_code == 200:
                body = r.json()
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content
            elif r.status_code == 401:
                err = r.text[:120]
                if "CreditsError" in err or "No payment method" in err:
                    logger.warning("Opencode Zen: no credits — trying alt model")
                    continue
                elif "ModelError" in err:
                    logger.warning("Opencode Zen: model %s not supported", model)
                    continue
                else:
                    logger.warning("Opencode Zen auth error: %s", err)
                    return _tier_opencode_cli(query, timeout_s)
            else:
                logger.warning("Opencode Zen HTTP %d for model %s: %s", r.status_code, model, r.text[:200])
        except httpx.TimeoutException:
            logger.warning("Opencode Zen timed out for model %s", model)
            continue
        except Exception as exc:
            logger.warning("Opencode Zen error (model=%s): %s", model, exc)
            continue

    # All HTTP models failed → try CLI as last resort
    return _tier_opencode_cli(query, timeout_s)


def _tier_opencode_cli(query: str, timeout_s: int) -> str:
    """
    Fallback: opencode run CLI subprocess.
    """
    for model in (settings.OPENCODE_MODEL, settings.OPENCODE_ALT_MODEL):
        try:
            proc = subprocess.run(
                ["opencode", "run", query, "-m", model, "--format", "default"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = (proc.stdout or proc.stderr or "").strip()
            if output:
                return output
        except subprocess.TimeoutExpired:
            logger.warning("Opencode CLI timed out for model %s", model)
            continue
        except FileNotFoundError:
            logger.warning("opencode CLI not found in PATH")
            return ""
        except Exception as exc:
            logger.warning("Opencode CLI error (model=%s): %s", model, exc)
            continue
    return ""


def _tier_qwen(query: str, timeout_s: int) -> str:
    """
    Tier 3: qwen -p "query" (headless mode).
    """
    try:
        proc = subprocess.run(
            ["qwen", "-p", query],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        return output
    except subprocess.TimeoutExpired:
        logger.warning("Qwen headless timed out")
        return ""
    except FileNotFoundError:
        logger.warning("qwen not found in PATH")
        return ""
    except Exception as exc:
        logger.warning("Qwen headless error: %s", exc)
        return ""


def _check_gemini_quota() -> bool:
    """Check if we can still use Gemini this hour. Returns True if allowed."""
    try:
        telem = get_telemetry()
        # Quick count from conversations table where model=gemini in last hour
        rows = telem._db.execute(
            "SELECT COUNT(*) as c FROM conversations "
            "WHERE model = 'gemini' AND ts > strftime('%Y-%m-%dT%H:%M:%fZ','now', '-1 hour')"
        ).fetchone()
        count = rows["c"] if rows else 0
        return count < GEMINI_MAX_PER_HOUR
    except Exception:
        return True  # don't block on telemetry error


def _tier_gemini(query: str, timeout_s: int) -> str:
    """
    Tier 4: Gemini HTTP API (free tier, quota-limited).
    Falls back to CLI if no API key configured.
    """
    import httpx

    api_key = settings.GEMINI_API_KEY
    if not api_key:
        return _tier_gemini_cli(query, timeout_s)

    for model in ("gemini-2.0-flash", "gemini-1.5-flash"):
        try:
            r = httpx.post(
                f"{settings.GEMINI_BASE_URL}/models/{model}:generateContent",
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": query}]}],
                    "generationConfig": {"maxOutputTokens": 4096},
                },
                timeout=timeout_s,
            )
            if r.status_code == 200:
                body = r.json()
                content = body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if content:
                    return content
            elif r.status_code == 429:
                logger.warning("Gemini API quota exceeded (model=%s), trying alt model", model)
                continue
            elif r.status_code == 400:
                err = r.text[:120]
                logger.warning("Gemini API bad request (model=%s): %s", model, err)
                continue
            else:
                logger.warning("Gemini API HTTP %d (model=%s): %s", r.status_code, model, r.text[:200])
        except httpx.TimeoutException:
            logger.warning("Gemini API timed out for model %s", model)
            continue
        except Exception as exc:
            logger.warning("Gemini API error (model=%s): %s", model, exc)
            continue

    # All HTTP models failed → try CLI
    return _tier_gemini_cli(query, timeout_s)


def _tier_gemini_cli(query: str, timeout_s: int) -> str:
    """
    Fallback: gemini -p CLI subprocess.
    """
    if not _check_gemini_quota():
        logger.warning("Gemini CLI quota exceeded for this hour")
        return ""

    try:
        proc = subprocess.run(
            ["gemini", "-p", query],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        return output
    except subprocess.TimeoutExpired:
        logger.warning("Gemini CLI timed out")
        return ""
    except FileNotFoundError:
        logger.warning("Gemini CLI not found in PATH")
        return ""
    except Exception as exc:
        logger.warning("Gemini CLI error: %s", exc)
        return ""


def _tier_telegram(chat_id: int, query: str, timeout_s: int = 10) -> str:
    """
    Tier 5: Ask the user via Telegram if they want to try Copilot manually.
    Returns the user's decision or empty string on timeout.

    This requires the Telegram bot application to be passed in.
    Since this runs in the same process, we use a thread-safe event queue.
    """
    from src.utils.fallback_queue import get_fallback_queue

    fq = get_fallback_queue()

    # Build the message
    copilot_remaining = _get_copilot_remaining()
    msg = (
        f"⚠️ Todos los tiers automáticos fallaron.\n\n"
        f"📋 Query: \"{query[:200]}\"\n\n"
        f"Opciones:\n"
        f"1️⃣  Abrir Copilot TUI → responde `copilot` ({copilot_remaining}/10 restantes)\n"
        f"2️⃣  Armar prompt para Qwen → responde `qwen`\n"
        f"3️⃣  Cancelar → responde `no` o ignora este mensaje"
    )

    # Send the message
    try:
        from src.utils.fallback_queue import get_telegram_app
        app = get_telegram_app()
        if app:
            # Use bot.send_message via the app's bot
            import asyncio
            async def _send():
                await app.bot.send_message(chat_id=chat_id, text=msg)
            asyncio.get_event_loop().run_until_complete(_send())
        else:
            logger.warning("Telegram app not available for fallback")
            return ""
    except Exception as exc:
        logger.warning("Could not send Telegram fallback message: %s", exc)
        return ""

    # Wait for reply
    reply = fq.wait_for(chat_id, timeout_s)
    reply_lower = reply.lower().strip() if reply else ""

    if reply_lower in ("copilot", "copi", "1"):
        return _exec_copilot_tui_hint(query)
    elif reply_lower in ("qwen", "qwen prompt", "2"):
        return _build_qwen_prompt(query)
    else:
        return "No pude procesar tu consulta automáticamente. Intenta de nuevo con otra pregunta."


def _get_copilot_remaining() -> int:
    """Get remaining Copilot calls this hour (max 10)."""
    try:
        telem = get_telemetry()
        rows = telem._db.execute(
            "SELECT COALESCE(MAX(count), 0) as c FROM copilot_usage"
        ).fetchone()
        count = rows["c"] if rows else 0
        return max(0, 10 - count)
    except Exception:
        return 10


def _exec_copilot_tui_hint(query: str) -> str:
    """Return instructions to open Copilot TUI."""
    return (
        f"📋 Abre Copilot en tu terminal y pega tu pregunta:\n\n"
        f"```\ncopilot\n```\n\n"
        f"Luego escribe:\n"
        f"> {query[:300]}\n\n"
        f"Copilot procesará tu solicitud. Cuando tengas la respuesta, "
        f"pégala aquí si quieres que la formatee."
    )


def _build_qwen_prompt(query: str) -> str:
    """Build a ready-to-paste prompt for Qwen Code."""
    return (
        f"📋 Abre Qwen y pega esto:\n\n"
        f"```\n"
        f"Eres un Senior AI Engineer. Responde la siguiente pregunta:\n\n"
        f"{query}\n\n"
        f"Genera una respuesta completa, con código si aplica.\n"
        f"```\n\n"
        f"Copia el bloque completo y pégalo en `qwen -p`. "
        f"Luego pega la respuesta aquí si quieres que la formatee."
    )


# ---- Main entry point ----

def call_with_fallback(
    query: str,
    task_type: str = "general",
    chat_id: int | None = None,
) -> dict[str, str]:
    """
    Execute the full fallback chain.

    Returns dict with:
      - response: the answer text
      - tier: which tier responded ("ollama" | "opencode" | "qwen" | "gemini" | "telegram")

    Args:
        query: the user query
        task_type: "code", "general", "search" — affects timeout
        chat_id: Telegram chat ID for Tier 5 interrupt (optional)
    """
    timeout = settings.FALLBACK_TIMEOUT
    if task_type == "code":
        timeout = max(timeout, 45)
    elif task_type == "search":
        timeout = max(timeout, 60)

    start = time.monotonic()
    tiers_tried: list[str] = []

    # ---- Tier 1: Ollama ----
    tiers_tried.append(TIER_OLLAMA)
    model = settings.CODE_MODEL if task_type == "code" else settings.GENERAL_MODEL
    ollama_timeout = _get_model_timeout(model)
    try:
        response = _tier_ollama(query, model, ollama_timeout)
        elapsed = time.monotonic() - start
        return {"response": response, "tier": TIER_OLLAMA}
    except TierTimeout as exc:
        elapsed = time.monotonic() - start
        _log_fallback(TIER_OLLAMA, TIER_OPENCODE, query, f"timeout({ollama_timeout}s)", elapsed)
        logger.warning("Ollama timeout (model=%s, %ds), falling to Opencode Zen", model, ollama_timeout)
    except VRAMError as exc:
        elapsed = time.monotonic() - start
        _log_fallback(TIER_OLLAMA, TIER_OPENCODE, query, f"VRAM error: {exc}", elapsed)
        logger.warning("Ollama VRAM error (model=%s), falling to Opencode Zen", model)
    except Exception as exc:
        elapsed = time.monotonic() - start
        _log_fallback(TIER_OLLAMA, TIER_OPENCODE, query, str(exc)[:100], elapsed)
        logger.warning("Ollama error (model=%s), falling to Opencode Zen: %s", model, exc)

    # ---- Tier 2: Opencode ----
    tiers_tried.append(TIER_OPENCODE)
    try:
        response = _tier_opencode(query, timeout + 15)
        if response:
            elapsed = time.monotonic() - start
            return {"response": response, "tier": TIER_OPENCODE}
    except Exception as exc:
        elapsed = time.monotonic() - start
        _log_fallback(TIER_OPENCODE, TIER_QWEN, query, str(exc)[:100], elapsed)

    # ---- Tier 3: Qwen headless ----
    tiers_tried.append(TIER_QWEN)
    try:
        response = _tier_qwen(query, timeout + 15)
        if response:
            elapsed = time.monotonic() - start
            return {"response": response, "tier": TIER_QWEN}
    except Exception as exc:
        elapsed = time.monotonic() - start
        _log_fallback(TIER_QWEN, TIER_GEMINI, query, str(exc)[:100], elapsed)

    # ---- Tier 4: Gemini ----
    tiers_tried.append(TIER_GEMINI)
    try:
        response = _tier_gemini(query, timeout)
        if response:
            elapsed = time.monotonic() - start
            return {"response": response, "tier": TIER_GEMINI}
    except Exception as exc:
        elapsed = time.monotonic() - start
        _log_fallback(TIER_GEMINI, TIER_TELEGRAM, query, str(exc)[:100], elapsed)

    # ---- Tier 5: Telegram interrupt ----
    if chat_id:
        tiers_tried.append(TIER_TELEGRAM)
        try:
            response = _tier_telegram(chat_id, query)
            if response:
                elapsed = time.monotonic() - start
                return {"response": response, "tier": TIER_TELEGRAM}
        except Exception as exc:
            logger.warning("Tier 5 Telegram fallback failed: %s", exc)

    # ---- All tiers exhausted ----
    elapsed = time.monotonic() - start
    _log_fallback("ALL", "NONE", query, "all tiers failed", elapsed)
    return {
        "response": (
            "⚠️ No pude procesar tu consulta automáticamente. "
            "Todos los tiers fallaron. Intenta reformular tu pregunta o intenta más tarde."
        ),
        "tier": "none",
    }
