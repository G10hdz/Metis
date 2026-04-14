"""Telegram bot integration for Metis — python-telegram-bot v21+."""

from __future__ import annotations

import atexit
import asyncio
import fcntl
import inspect
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import urllib3
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from src.config import settings
from src.graph.orchestrator import get_graph
from src.graph.state import MetisState
from src.utils.fallback_queue import set_telegram_app, submit_reply

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton single-instance enforcement — OS-level fcntl.flock
# The kernel auto-releases the lock if the process dies (even SIGKILL)
# ---------------------------------------------------------------------------

_LOCK_HANDLE = None
_BOT_APP = None  # global reference for graceful shutdown

# Async background jobs for mobile-first command workflow (/run, /status <id>, /cancel <id>)
_ASYNC_JOBS: dict[str, dict[str, Any]] = {}
_ASYNC_JOB_TASKS: dict[str, asyncio.Task] = {}
_ASYNC_JOBS_BY_CHAT: dict[int, list[str]] = {}
_ASYNC_MAX_ACTIVE_PER_CHAT = int(os.getenv("METIS_ASYNC_MAX_ACTIVE_PER_CHAT", "3"))
_ASYNC_HISTORY_PER_CHAT = int(os.getenv("METIS_ASYNC_HISTORY_PER_CHAT", "50"))


def _is_chat_allowed(chat_id: int | None) -> bool:
    if chat_id is None:
        return False
    if settings.ALLOWED_CHAT_IDS_SET and chat_id not in settings.ALLOWED_CHAT_IDS_SET:
        return False
    return True


def _active_jobs_for_chat(chat_id: int) -> int:
    job_ids = _ASYNC_JOBS_BY_CHAT.get(chat_id, [])
    active = 0
    for job_id in job_ids:
        status = _ASYNC_JOBS.get(job_id, {}).get("status")
        if status in {"queued", "running"}:
            active += 1
    return active


def _trim_chat_history(chat_id: int) -> None:
    job_ids = _ASYNC_JOBS_BY_CHAT.get(chat_id, [])
    if len(job_ids) <= _ASYNC_HISTORY_PER_CHAT:
        return
    overflow = len(job_ids) - _ASYNC_HISTORY_PER_CHAT
    kept: list[str] = []
    removed = 0
    for job_id in job_ids:
        if removed < overflow and _ASYNC_JOBS.get(job_id, {}).get("status") in {"done", "failed", "cancelled"}:
            _ASYNC_JOB_TASKS.pop(job_id, None)
            _ASYNC_JOBS.pop(job_id, None)
            removed += 1
            continue
        kept.append(job_id)
    _ASYNC_JOBS_BY_CHAT[chat_id] = kept


def _register_job(chat_id: int, query: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:8]
    now = time.time()
    job = {
        "job_id": job_id,
        "chat_id": chat_id,
        "query": query,
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "response": "",
        "error": "",
    }
    _ASYNC_JOBS[job_id] = job
    _ASYNC_JOBS_BY_CHAT.setdefault(chat_id, []).append(job_id)
    _trim_chat_history(chat_id)
    return job


def _format_job_status(job: dict[str, Any]) -> str:
    status = job.get("status", "unknown")
    query = str(job.get("query", ""))
    query_preview = query if len(query) <= 100 else query[:97] + "..."
    lines = [
        f"🧩 Job `{job.get('job_id', '-')}`",
        f"Estado: {status}",
        f"Query: {query_preview}",
    ]

    started = job.get("started_at")
    finished = job.get("finished_at")
    if isinstance(started, (int, float)):
        lines.append(f"Iniciado: {time.strftime('%H:%M:%S', time.localtime(started))}")
    if isinstance(finished, (int, float)):
        lines.append(f"Terminado: {time.strftime('%H:%M:%S', time.localtime(finished))}")

    if status == "done":
        response = str(job.get("response", ""))
        if response:
            if len(response) > 1000:
                response = response[:990] + "\n\n...(truncated)"
            lines.extend(["", "Respuesta:", response])
    elif status in {"failed", "cancelled"}:
        error = str(job.get("error", "")).strip() or "No details"
        lines.extend(["", f"Detalle: {error}"])

    return "\n".join(lines)


async def _run_job(job_id: str) -> None:
    job = _ASYNC_JOBS.get(job_id)
    if not job:
        return

    chat_id = int(job["chat_id"])
    query = str(job["query"])
    app = _BOT_APP

    if app is None:
        job["status"] = "failed"
        job["error"] = "Telegram app not initialized"
        job["finished_at"] = time.time()
        return

    try:
        job["status"] = "running"
        job["started_at"] = time.time()

        await app.bot.send_message(
            chat_id=chat_id,
            text=f"🚀 Ejecutando job `{job_id}`...",
            parse_mode="Markdown",
        )

        loop = asyncio.get_running_loop()

        def _invoke() -> dict[str, Any]:
            graph = get_graph()
            state = MetisState.from_query(query)
            state.source = "telegram"
            return graph.invoke(state.model_dump())

        result = await loop.run_in_executor(None, _invoke)
        response = str(result.get("response", "No response generated."))
        job["response"] = response
        job["status"] = "done"
        job["finished_at"] = time.time()

        if len(response) > 3900:
            response = response[:3890] + "\n\n...(truncated)"

        await app.bot.send_message(chat_id=chat_id, text=f"✅ Job `{job_id}` completado.", parse_mode="Markdown")
        await app.bot.send_message(chat_id=chat_id, text=response, parse_mode=None)
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["error"] = "Cancelled by user"
        job["finished_at"] = time.time()
        await app.bot.send_message(chat_id=chat_id, text=f"🛑 Job `{job_id}` cancelado.", parse_mode="Markdown")
        raise
    except Exception as exc:
        logger.exception("Async job %s failed: %s", job_id, exc)
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = time.time()
        await app.bot.send_message(chat_id=chat_id, text=f"⚠️ Job {job_id} falló: {exc}", parse_mode=None)
    finally:
        _ASYNC_JOB_TASKS.pop(job_id, None)


def _parse_job_id_from_status(cmd_text: str) -> str | None:
    parts = cmd_text.strip().split(maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    return None


def _extract_job_id_arg(context: ContextTypes.DEFAULT_TYPE) -> str:
    return " ".join(context.args).strip()


async def run_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /run <query> command to execute in background."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("🔒 Access denied.")
        return

    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text(
            "Uso: /run <query>\n"
            "Ejemplo: /run busca últimas noticias de python 3.13"
        )
        return

    assert chat_id is not None
    if _active_jobs_for_chat(chat_id) >= _ASYNC_MAX_ACTIVE_PER_CHAT:
        await update.message.reply_text(
            f"⏳ Ya tienes {_ASYNC_MAX_ACTIVE_PER_CHAT} jobs activos. "
            "Espera a que termine uno o cancela con /cancel <job_id>."
        )
        return

    job = _register_job(chat_id, query)
    task = asyncio.create_task(_run_job(job["job_id"]))
    _ASYNC_JOB_TASKS[job["job_id"]] = task

    await update.message.reply_text(
        f"🧩 Job creado: `{job['job_id']}`\n"
        "Usa /status <job_id> para ver progreso y /cancel <job_id> para cancelar.",
        parse_mode="Markdown",
    )


async def job_status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status and /status <job_id> (bot+jobs)."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("🔒 Access denied.")
        return

    cmd_text = update.message.text if update.message and update.message.text else "/status"
    requested_job_id = _extract_job_id_arg(context) or _parse_job_id_from_status(cmd_text)

    if requested_job_id:
        job = _ASYNC_JOBS.get(requested_job_id)
        if not job or job.get("chat_id") != chat_id:
            await update.message.reply_text(f"No encontré job `{requested_job_id}`.", parse_mode="Markdown")
            return
        await update.message.reply_text(_format_job_status(job), parse_mode=None)
        return

    # Default /status => global + last jobs for this chat
    try:
        from src.telemetry.store import get_telemetry

        telem = get_telemetry()
        stats = telem.stats()
        status_text = (
            f"🟢 *Bot Online*\n\n"
            f"📊 Stats:\n"
            f"  Total queries: {stats['total']}\n"
            f"  Errores: {stats['errors']}\n"
            f"  Latencia promedio: {stats['avg_latency_ms']}ms\n\n"
            f"🔧 Modelos:\n"
            f"  Router: {settings.ROUTER_MODEL}\n"
            f"  Code: {settings.CODE_MODEL}\n"
            f"  General: {settings.GENERAL_MODEL}\n"
            f"  Embeddings: {settings.EMBEDDING_MODEL}\n\n"
            f"📡 Fallback Chain: Ollama → Opencode Zen API → Qwen → Gemini API → Telegram"
        )
    except Exception:
        status_text = "🟢 Bot Online\nFallback Chain: Ollama → Opencode Zen API → Qwen → Gemini API → Telegram"

    lines = [status_text]
    if chat_id is not None:
        job_ids = _ASYNC_JOBS_BY_CHAT.get(chat_id, [])[-5:]
        if job_ids:
            lines.append("\n🧩 Últimos jobs:")
            for job_id in reversed(job_ids):
                job = _ASYNC_JOBS.get(job_id)
                if not job:
                    continue
                lines.append(f"  - {job_id}: {job.get('status', 'unknown')}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel <job_id>."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("🔒 Access denied.")
        return

    job_id = " ".join(context.args).strip()
    if not job_id:
        await update.message.reply_text("Uso: /cancel <job_id>")
        return

    job = _ASYNC_JOBS.get(job_id)
    if not job or job.get("chat_id") != chat_id:
        await update.message.reply_text(f"No encontré job `{job_id}`.", parse_mode="Markdown")
        return

    status = job.get("status")
    if status in {"done", "failed", "cancelled"}:
        await update.message.reply_text(f"Job `{job_id}` ya terminó con estado: {status}.", parse_mode="Markdown")
        return

    task = _ASYNC_JOB_TASKS.get(job_id)
    if task and not task.done():
        task.cancel()
        await update.message.reply_text(f"🛑 Cancelando job `{job_id}`...", parse_mode="Markdown")
        return

    job["status"] = "cancelled"
    job["error"] = "Cancelled by user"
    job["finished_at"] = time.time()
    await update.message.reply_text(f"🛑 Job `{job_id}` cancelado.", parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Scheduled tasks (proactive automation)
# ---------------------------------------------------------------------------

_SCHEDULE_USAGE = (
    "📅 *Scheduled Tasks*\n\n"
    "Uso: `/schedule <name> | <expression>`\n\n"
    "Ejemplos:\n"
    "  `/schedule devops-news | daily 10:00 search latest devops articles`\n"
    "  `/schedule status-check | every 2h check ollama status`\n"
    "  `/schedule weekly-review | weekly fri 18:00 resume mi semana de trabajo`\n"
    "  `/schedule hourly-ping | hourly ping ollama`\n"
    '  `/schedule custom | cron "0 9 * * 1-5" buenos dias resumen del dia`\n\n'
    "Tipos de schedule:\n"
    "  • `daily HH:MM` — todos los días a esa hora\n"
    "  • `hourly` — cada hora\n"
    "  • `every Nh|Nm|Ns` — intervalo (ej. `every 30m`)\n"
    "  • `weekly DAY HH:MM` — DAY = mon|tue|wed|thu|fri|sat|sun\n"
    '  • `cron "EXPR"` — crontab clásico (5 campos)\n\n'
    "Otros comandos:\n"
    "  `/schedules` — listar tareas programadas\n"
    "  `/unschedule <job_id>` — eliminar una tarea\n"
    "  `/runschedule <job_id>` — ejecutar manualmente ahora"
)


async def schedule_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /schedule <name> | <expression>."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("🔒 Access denied.")
        return

    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text(_SCHEDULE_USAGE, parse_mode="Markdown")
        return

    if "|" not in raw:
        await update.message.reply_text(
            "❌ Falta separador `|` entre nombre y expresión.\n\n" + _SCHEDULE_USAGE,
            parse_mode="Markdown",
        )
        return

    name_part, expr_part = raw.split("|", 1)
    name = name_part.strip()
    expr = expr_part.strip()
    if not name or not expr:
        await update.message.reply_text(
            "❌ Nombre o expresión vacíos.\n\n" + _SCHEDULE_USAGE,
            parse_mode="Markdown",
        )
        return

    try:
        from src.scheduler import register_schedule, ScheduleParseError
        record = register_schedule(chat_id=int(chat_id), name=name, expr=expr)
    except ScheduleParseError as exc:
        await update.message.reply_text(f"❌ {exc}", parse_mode=None)
        return
    except Exception as exc:
        logger.exception("Failed to register schedule: %s", exc)
        await update.message.reply_text(f"⚠️ Error: {exc}", parse_mode=None)
        return

    await update.message.reply_text(
        f"✅ Schedule creado: `{record['job_id']}`\n"
        f"  Nombre: *{record['name']}*\n"
        f"  Expresión: `{record['schedule_expr']}`\n"
        f"  Query: _{record['query'][:120]}_\n\n"
        f"Ver con /schedules, eliminar con /unschedule {record['job_id']}",
        parse_mode="Markdown",
    )


async def schedules_list_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /schedules — list all scheduled tasks for this chat."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("🔒 Access denied.")
        return

    from src.scheduler import get_schedule_store
    store = get_schedule_store()
    schedules = store.list_all(chat_id=int(chat_id))

    if not schedules:
        await update.message.reply_text(
            "📭 No tienes tareas programadas.\n\nUsa /schedule para crear una.",
            parse_mode="Markdown",
        )
        return

    lines = [f"📅 *Tareas programadas* ({len(schedules)})", ""]
    for s in schedules:
        enabled_icon = "🟢" if s.get("enabled") else "⚪"
        last_status = s.get("last_status") or "—"
        run_count = s.get("run_count", 0)
        lines.append(
            f"{enabled_icon} `{s['job_id']}` *{s['name']}*\n"
            f"   ⏰ {s['schedule_expr']}\n"
            f"   💬 _{s['query'][:80]}_\n"
            f"   📊 runs: {run_count} · last: {last_status}"
        )
    lines.append("\n_Eliminar:_ `/unschedule <job_id>`")
    lines.append("_Ejecutar ya:_ `/runschedule <job_id>`")
    await update.message.reply_text("\n\n".join(lines), parse_mode="Markdown")


async def unschedule_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /unschedule <job_id>."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("🔒 Access denied.")
        return

    job_id = " ".join(context.args).strip()
    if not job_id:
        await update.message.reply_text("Uso: /unschedule <job_id>")
        return

    from src.scheduler import get_schedule_store, unregister_schedule
    store = get_schedule_store()
    record = store.get(job_id)
    if not record or record.get("chat_id") != chat_id:
        await update.message.reply_text(f"No encontré schedule `{job_id}`.", parse_mode="Markdown")
        return

    if unregister_schedule(job_id):
        await update.message.reply_text(
            f"🗑️ Schedule `{job_id}` (*{record['name']}*) eliminado.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(f"⚠️ No pude eliminar `{job_id}`.", parse_mode="Markdown")


async def run_schedule_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /runschedule <job_id> — fire a scheduled task immediately."""
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("🔒 Access denied.")
        return

    job_id = " ".join(context.args).strip()
    if not job_id:
        await update.message.reply_text("Uso: /runschedule <job_id>")
        return

    from src.scheduler import get_schedule_store
    from src.scheduler.runner import _execute_scheduled_job
    store = get_schedule_store()
    record = store.get(job_id)
    if not record or record.get("chat_id") != chat_id:
        await update.message.reply_text(f"No encontré schedule `{job_id}`.", parse_mode="Markdown")
        return

    await update.message.reply_text(
        f"▶️ Ejecutando `{job_id}` ahora...",
        parse_mode="Markdown",
    )
    asyncio.create_task(_execute_scheduled_job(job_id))


def _acquire_singleton_lock() -> None:
    """
    Prevent multiple local Metis polling instances for the same token.

    Uses fcntl.flock (advisory lock at kernel level).
    The lock is automatically released by the OS if the process is killed
    (SIGKILL, segfault, OOM killer, etc.).

    If the lock is already held, reads the PID from the lock file and
    reports it to the user before exiting.

    If the PID is dead (stale lock from crash), removes the stale lock
    and retries.
    """
    global _LOCK_HANDLE

    if _LOCK_HANDLE is not None:
        return

    lock_path = Path(
        os.getenv("METIS_TELEGRAM_LOCK_FILE", str(Path.home() / ".metis" / "telegram_bot.lock"))
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing PID BEFORE opening (opening with "w" truncates!)
    old_pid_str = ""
    if lock_path.exists():
        try:
            old_pid_str = lock_path.read_text(encoding="utf-8").strip()
        except Exception:
            old_pid_str = ""

    handle = lock_path.open("w", encoding="utf-8")

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        # Lock is held by another process — check if it's still alive
        handle.close()

        if old_pid_str.isdigit():
            old_pid = int(old_pid_str)
            try:
                os.kill(old_pid, 0)  # signal 0 = check if alive
                handle.close() if not handle.closed else None
                logger.error(
                    "Metis Telegram bot is already running (PID %d).\n"
                    "Stop it first:  kill %d\n"
                    "Force kill:      kill -9 %d",
                    old_pid, old_pid, old_pid,
                )
                sys.exit(1)
            except OSError:
                # Process is dead — stale lock, remove and retry
                lock_path.unlink(missing_ok=True)
                logger.warning("Stale lock file (PID %d is dead). Retrying...", old_pid)
                _acquire_singleton_lock()
                return
        else:
            logger.error(
                "Metis Telegram bot is already running.\n"
                "Stop the other instance before starting a new one."
            )
            sys.exit(1)

    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _LOCK_HANDLE = handle

    logger.info("Singleton lock acquired (PID %d, lock: %s)", os.getpid(), lock_path)


def _release_singleton_lock() -> None:
    """Release polling singleton lock."""
    global _LOCK_HANDLE

    if _LOCK_HANDLE is None:
        return

    lock_path = Path(
        os.getenv("METIS_TELEGRAM_LOCK_FILE", str(Path.home() / ".metis" / "telegram_bot.lock"))
    )

    try:
        fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
        _LOCK_HANDLE.close()
        lock_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Failed to release lock: %s", exc)
    finally:
        _LOCK_HANDLE = None
        logger.info("Singleton lock released")


# ---------------------------------------------------------------------------
# Graceful shutdown via SIGTERM / SIGINT
# ---------------------------------------------------------------------------

def _register_signal_handlers() -> None:
    """Register signal handlers for graceful bot shutdown."""

    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully...", sig_name)
        _graceful_shutdown()

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    logger.debug("Signal handlers registered for SIGTERM, SIGINT")


def _graceful_shutdown() -> None:
    """Stop the Telegram bot, release lock, and exit cleanly."""
    global _BOT_APP

    # Stop the PTB application
    if _BOT_APP is not None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_BOT_APP.stop())
            loop.close()
            logger.info("Telegram bot stopped")
        except Exception as exc:
            logger.warning("Error during bot stop: %s", exc)

    # Release the singleton lock
    _release_singleton_lock()

    logger.info("Graceful shutdown complete")
    sys.exit(0)


START_TEXT = (
    "🧠 *Metis v2.5* — AI Agent Orchestrator\n\n"
    "¿Qué puedo hacer?\n\n"
    "💬 *Chatea conmigo* — escribe cualquier pregunta\n"
    "🔍 *Buscar en la web* — di 'busca...' o 'últimas noticias de...'\n"
    "💻 *Generar código* — 'escribe un script que...'\n"
    "📚 *RAG local* — '¿qué es...?' (usa mi base de conocimiento)\n"
    "🔎 *Investigación profunda* — responde 'deeper' después de una búsqueda\n"
    "📄 *Leer archivos* — 'lee el config.py en ~/Vscode-projects/...'\n"
    "📑 *PDFs y DOCX* — 'lee el reporte.pdf' (extrae texto automáticamente)\n"
    "✏️ *Editar archivos* — 'edita X y cambia A por B'\n"
    "🗑️ *Eliminar archivos* — 'borra el archivo X' (pide confirmación)\n"
    "🖥️ *Comandos bash* — 'ejecuta ls -la ~/Vscode-projects/...'\n"
    "🗣️ *Voz TTS* — '/speak texto --lang es' (ES/EN/ZH)\n"
    "🎤 *Echo Práctica* — '/practice' para practicar pronunciación\n\n"
    "⏰ *Tareas programadas* — '/schedule devops | daily 10:00 search devops news'\n\n"
    "Comandos:\n"
    "/run <query> — Ejecutar en background (async)\n"
    "/status <job_id> — Estado de un job async\n"
    "/cancel <job_id> — Cancelar job async\n"
    "/schedule <name> | <expr> — Crear tarea programada\n"
    "/schedules — Listar tareas programadas\n"
    "/unschedule <job_id> — Eliminar tarea programada\n"
    "/runschedule <job_id> — Ejecutar tarea programada ahora\n"
    "/speak <text> — Texto a voz (ES/EN/ZH)\n"
    "/practice [nivel] [idioma] — Práctica de pronunciación\n"
    "/progress — Ver tu progreso de práctica\n"
    "/status — Estado del bot y modelos\n"
    "/ping — Verificar Ollama y sistema\n"
    "/capabilities — Manual completo de capacidades\n"
    "/help — Esta ayuda"
)


# ---------------------------------------------------------------------------
# Capability pages — each is one Telegram message (~4000 char limit)
# ---------------------------------------------------------------------------

_CAPABILITY_PAGES = {
    "cap_overview": {
        "title": "🧠 Metis v2.4 — Architecture Overview",
        "body": (
            "*9-Route LangGraph StateGraph*\n\n"
            "Every message you send goes through a *router* that classifies it into one of 9 specialized agents:\n\n"
            "1️⃣ *rag* — Retrieves from local ChromaDB KB\n"
            "2️⃣ *code* — Generates code + validates syntax\n"
            "3️⃣ *search* — DuckDuckGo web search (recursive)\n"
            "4️⃣ *search_continue* — \"deeper\" follow-up research\n"
            "5️⃣ *file* — Reads files (text, PDF, DOCX)\n"
            "6️⃣ *file_edit* — Edits files with 2-step confirmation\n"
            "7️⃣ *file_delete* — Deletes files with confirmation\n"
            "8️⃣ *bash* — Runs whitelisted shell commands\n"
            "9️⃣ *general* — Fallback for everything else\n\n"
            "*Fallback Chain (5 tiers):*\n"
            "Ollama → Opencode Zen API → Qwen → Gemini API → Telegram (you decide)\n\n"
            "*Source Tracking:* Telegram / Web / CLI — all logged to SQLite telemetry\n\n"
            "Tap a category below for details ↓"
        ),
    },
    "cap_files": {
        "title": "📄 File Operations — Read, Edit, Delete",
        "body": (
            "*READ Files* (text, PDF, DOCX)\n\n"
            "Just describe the file in natural language:\n"
            "• \"Lee el config.py en ~/Vscode-projects/Metis\"\n"
            "• \"Muestra el reporte.pdf en ~/Vscode-projects/docs\"\n"
            "• \"Abre el documento thesis.docx en ~/...\"\n"
            "• \"Show me ~/projects/my-app/data.json\"\n\n"
            "*Format detection:*\n"
            "• `.py, .js, .ts, .md, .txt, .yaml, .json...` → plain text\n"
            "• `.pdf` → PyMuPDF (fitz) → fallback to pypdf\n"
            "• `.docx` → python-docx (paragraphs + tables)\n\n"
            "*Security:*\n"
            "• Only files under `~/Vscode-projects/`\n"
            "• No symlinks, no binaries (except PDF/DOCX)\n"
            "• Max 50KB file size\n"
            "• 3500-char preview for Telegram (full content stored)\n\n"
            "*EDIT Files* (2-step confirmation)\n"
            "• \"Edita config.py y cambia 'debug=False' por 'True'\"\n"
            "• \"Agrega al final de notes.md: ...\"\n"
            "• \"Inserta en línea 5 de main.py: ...\"\n"
            "Metis shows the change first, asks *sí/no* before executing\n\n"
            "*DELETE Files* (mandatory confirmation)\n"
            "• \"Borra el archivo temp.txt en ~/Vscode-projects/Metis\"\n"
            "• \"Delete ~/Vscode-projects/test/old_file.py\"\n"
            "Always asks before deleting — never instant"
        ),
    },
    "cap_bash": {
        "title": "🖥️ Bash Command Execution",
        "body": (
            "Run safe shell commands from Telegram:\n\n"
            "• \"Ejecuta ls -la ~/Vscode-projects/Metis\"\n"
            "• \"Run grep -r 'def test' ~/Vscode-projects/tests\"\n"
            "• \"Correr pwd\"\n"
            "• \"Shell command: find ~/Vscode-projects -name '*.py'\"\n\n"
            "*Allowed commands (20+):*\n"
            "`ls, cat, head, tail, wc, find, grep, du, df, pwd, tree, stat, diff, sort, uniq, cut, awk, sed, tr, echo, date, whoami, uname, uptime, free, ps, top, md5sum, sha256sum, file, which, whereis, basename, dirname`\n\n"
            "*Blocked patterns (never allowed):*\n"
            "`rm, sudo, chmod, chown, kill, dd, mkfs, wget, curl, pip install, apt, |, &&, ||, ;, >, >>, $(, `, eval, exec, source, export, alias`\n\n"
            "*Limits:*\n"
            "• 10-second timeout per command\n"
            "• 20KB max output (truncated with notice)\n"
            "• Working directory: `~/Vscode-projects/`\n"
            "• Exit code shown if non-zero"
        ),
    },
    "cap_search": {
        "title": "🔍 Web Search & Deep Research",
        "body": (
            "*WEB SEARCH* (DuckDuckGo, no API key)\n\n"
            "• \"Busca últimas noticias de IA\"\n"
            "• \"Find recent updates on Rust async\"\n"
            "• \"What's new in Python 3.13\"\n\n"
            "Search generates 3-5 queries, fetches results, synthesizes a summary with sources.\n\n"
            "*PROGRESSIVE DEEP RESEARCH*\n"
            "After a search, reply with:\n"
            "• \"deeper\" / \"go deeper\" / \"tell me more\" / \"dig deeper\"\n\n"
            "Each \"deeper\" adds 2 more research layers:\n"
            "  Layer 1: 5 queries → summary (~5-10s)\n"
            "  Layer 2: 2 more → detailed findings (~10-20s)\n"
            "  Layer 3: 2 more → full research report\n\n"
            "Context accumulates — you can go deeper multiple times for a comprehensive report with citations.\n\n"
            "*Router keywords:* `search, find, latest, news, current, trending, update, breaking, announced, released`"
        ),
    },
    "cap_code": {
        "title": "💻 Code Generation",
        "body": (
            "Generate production-ready code with syntax validation:\n\n"
            "• \"Write a Python decorator @retry with backoff\"\n"
            "• \"Implement rate limiting with Redis\"\n"
            "• \"Create a docker-compose for FastAPI + Postgres\"\n"
            "• \"Debug this error: IndexError list index out of range\"\n\n"
            "*How it works:*\n"
            "1. Router detects code keywords (`code, function, script, python, debug, api, docker, deploy...`)\n"
            "2. Code agent generates via fallback chain\n"
            "3. Every Python block validated with `ast.parse`\n"
            "4. Syntax errors appended as warnings\n\n"
            "*Router keywords:* `code, function, script, python, debug, error, bug, implement, refactor, algorithm, api, endpoint, class, def, import, pip, install, docker, deploy, server, http, json, yaml, lambda, terraform, aws, s3, ec2`\n\n"
            "Works for any language — Python gets syntax validation, others get formatted code blocks."
        ),
    },
    "cap_rag": {
        "title": "📚 RAG — Local Knowledge Base",
        "body": (
            "Answer questions from your personal knowledge base stored in ChromaDB:\n\n"
            "• \"¿Qué es la ley de Newton?\"\n"
            "• \"Explain backpropagation\"\n"
            "• \"What is the CAP theorem?\"\n\n"
            "*How it works:*\n"
            "1. Query embedded locally (nomic-embed-text)\n"
            "2. Top 3 chunks retrieved from ChromaDB\n"
            "3. Only chunks with similarity score > 0.65 used\n"
            "4. LLM answers based on context\n"
            "5. If no relevant chunks → falls back to general\n\n"
            "*Ingest PDFs:*\n"
            "`python -m src.memory.ingest /ruta/` — scans 150+ PDFs in ~1 min\n"
            "`python -m src.memory.ingest /ruta/ --marker` — preserves LaTeX equations (~5 min/PDF)\n\n"
            "Router keywords: `what is, how to, explain, describe, define, concept, documentation, docs, manual, guide, tutorial, learn, understand, why, when, where, which, history, difference between`"
        ),
    },
    "cap_fallback": {
        "title": "⚡ Fallback Chain — 5-Tier Reliability",
        "body": (
            "Every LLM call goes through an automatic 5-tier fallback chain:\n\n"
            "*Tier 1: Ollama (local, ROCm)*\n"
            "• phi3:mini (45s timeout), qwen2.5-coder:7b (90s)\n"
            "• VRAM errors auto-detected → cascades to Tier 2\n\n"
            "*Tier 2: Opencode Zen API (HTTP, free models)*\n"
            "• big-pickle, minimax-m2.5-free\n"
            "• OpenAI-compatible endpoint, no CLI needed\n\n"
            "*Tier 3: Qwen headless*\n"
            "• Local CLI invocation\n\n"
            "*Tier 4: Gemini API (free tier)*\n"
            "• gemini-2.0-flash, gemini-1.5-flash\n"
            "• Quota-guarded (20/hr limit), falls back to CLI\n\n"
            "*Tier 5: Telegram interrupt*\n"
            "• Bot messages YOU asking what to do\n"
            "• You reply \"copilot\" → instructions for Copilot TUI\n"
            "• You reply \"qwen\" → bot prepares the prompt for you\n\n"
            "All fallbacks logged to:\n"
            "`~/.metis/ergane-logs.md`"
        ),
    },
    "cap_security": {
        "title": "🔒 Security Architecture",
        "body": (
            "*File Access:*\n"
            "• Root: `~/Vscode-projects/` only (configurable via METIS_FILE_ROOT)\n"
            "• Symlinks: rejected\n"
            "• Extensions: whitelist only (.py, .pdf, .docx, etc.)\n"
            "• Max size: 50KB\n"
            "• Text-only detection for unknown formats\n\n"
            "*Bash Execution:*\n"
            "• 35 commands whitelisted\n"
            "• 20+ dangerous patterns blocked\n"
            "• No pipes, redirects, or command chaining\n"
            "• 10-second timeout\n"
            "• Working directory locked to allowed root\n\n"
            "*Chat Access:*\n"
            "• ALLOWED_CHAT_IDS — only your Telegram user\n"
            "• Blocked IDs logged\n\n"
            "*No API Key Exposure:*\n"
            "• All secrets in .env (gitignored)\n"
            "• No keys in logs or responses\n\n"
            "*Singleton Lock:*\n"
            "• Only one bot instance per token\n"
            "• OS-level fcntl.flock (auto-released on crash)\n"
            "• Stale lock detection (dead PID cleanup)\n"
            "• SIGTERM/SIGINT → graceful shutdown"
        ),
    },
    "cap_telemetry": {
        "title": "📊 Telemetry & Monitoring",
        "body": (
            "*Every query logged to SQLite* (`~/.metis/metis.db`):\n\n"
            "• Query text (truncated to 2000 chars)\n"
            "• Route classification (rag/code/search/file/etc.)\n"
            "• Response (truncated to 2000 chars)\n"
            "• Model used (phi3, qwen2.5, opencode, etc.)\n"
            "• Latency in milliseconds\n"
            "• Error messages if any\n"
            "• Source: telegram / web / cli\n\n"
            "*Commands:*\n"
            "`/status` — Total queries, errors, avg latency, active models\n"
            "`/ping` — Ollama API health, loaded models, VRAM status, config\n\n"
            "*Web UI Dashboard* (`localhost:7860`):\n"
            "• Route distribution pie chart\n"
            "• Latency trends over time\n"
            "• Recent conversations\n"
            "• Error rates\n\n"
            "*Fallback logs:* `~/.metis/ergane-logs.md`\n"
            "*VRAM logs:* `~/.metis/vram.log`"
        ),
    },
}

_CAPABILITY_ORDER = [
    "cap_overview",
    "cap_files",
    "cap_bash",
    "cap_search",
    "cap_code",
    "cap_rag",
    "cap_fallback",
    "cap_security",
    "cap_telemetry",
]


def _build_capability_keyboard(current_idx: int) -> InlineKeyboardMarkup:
    """Build navigation buttons for capability pages."""
    page = _CAPABILITY_ORDER[current_idx]
    emoji_map = {
        "cap_overview": "🧠",
        "cap_files": "📄",
        "cap_bash": "🖥️",
        "cap_search": "🔍",
        "cap_code": "💻",
        "cap_rag": "📚",
        "cap_fallback": "⚡",
        "cap_security": "🔒",
        "cap_telemetry": "📊",
    }
    buttons = []
    # Quick nav row (3 per row)
    row = []
    for i, p in enumerate(_CAPABILITY_ORDER):
        label = emoji_map.get(p, "📌")
        btn_text = "●" if i == current_idx else label
        row.append(InlineKeyboardButton(btn_text, callback_data=f"cap_{i}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    # Prev / Next / Close
    nav_row = []
    if current_idx > 0:
        nav_row.append(InlineKeyboardButton("◀️ Prev", callback_data=f"cap_{current_idx - 1}"))
    if current_idx < len(_CAPABILITY_ORDER) - 1:
        nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"cap_{current_idx + 1}"))
    nav_row.append(InlineKeyboardButton("✖️ Cerrar", callback_data="cap_close"))
    buttons.append(nav_row)

    return InlineKeyboardMarkup(buttons)


async def capabilities_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /capabilities command — full documentation manual."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id and chat_id not in settings.ALLOWED_CHAT_IDS_SET and settings.ALLOWED_CHAT_IDS_SET:
        await update.message.reply_text("🔒 Access denied.")
        return

    await update.message.chat.send_action("typing")

    # Show first page
    page_key = _CAPABILITY_ORDER[0]
    page = _CAPABILITY_PAGES[page_key]
    text = f"*{page['title']}*\n\n{page['body']}"
    keyboard = _build_capability_keyboard(0)

    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def capability_nav_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle capability page navigation from inline keyboard."""
    query = update.callback_query
    if not query:
        return

    data = query.data

    # Close button
    if data == "cap_close":
        await query.answer()
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        return

    # Navigation: cap_0, cap_1, etc.
    if data.startswith("cap_"):
        try:
            idx = int(data.split("_")[1])
            if 0 <= idx < len(_CAPABILITY_ORDER):
                page_key = _CAPABILITY_ORDER[idx]
                page = _CAPABILITY_PAGES[page_key]
                text = f"*{page['title']}*\n\n{page['body']}"
                keyboard = _build_capability_keyboard(idx)
                await query.answer()
                await query.edit_message_text(text, parse_mode="Markdown", reply_markup=keyboard)
            else:
                await query.answer()
        except (ValueError, IndexError):
            await query.answer()


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command with interactive buttons."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id and chat_id not in settings.ALLOWED_CHAT_IDS_SET and settings.ALLOWED_CHAT_IDS_SET:
        await update.message.reply_text("🔒 Access denied.")
        return

    keyboard = [
        [InlineKeyboardButton("💬 Preguntar", callback_data="help_chat"),
         InlineKeyboardButton("🔍 Buscar", callback_data="help_search")],
        [InlineKeyboardButton("💻 Código", callback_data="help_code"),
         InlineKeyboardButton("📚 RAG Local", callback_data="help_rag")],
        [InlineKeyboardButton("📋 Capabilities", callback_data="cap_0"),
         InlineKeyboardButton("📊 Status", callback_data="help_status")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(START_TEXT, parse_mode="Markdown", reply_markup=reply_markup)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await start_handler(update, context)


async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Backwards-compatible wrapper for /status command."""
    await job_status_handler(update, context)


async def ping_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ping command — check Ollama and system health."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id and chat_id not in settings.ALLOWED_CHAT_IDS_SET and settings.ALLOWED_CHAT_IDS_SET:
        await update.message.reply_text("🔒 Access denied.")
        return

    await update.message.chat.send_action("typing")
    
    ping_results = []
    
    # Check Ollama
    try:
        start = time.monotonic()
        http = urllib3.PoolManager()
        resp = http.request("GET", f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_latency = (time.monotonic() - start) * 1000
        
        if resp.status == 200:
            import json
            data = json.loads(resp.data)
            models = data.get("models", [])
            model_names = [m["name"] for m in models] if models else ["none loaded"]
            ping_results.append(f"🟢 Ollama: OK ({ollama_latency:.0f}ms)\n   Models: {', '.join(model_names)}")
        else:
            ping_results.append(f"🟡 Ollama: responded {resp.status} ({ollama_latency:.0f}ms)")
    except Exception as exc:
        ping_results.append(f"🔴 Ollama: DOWN - {exc}")
    
    # Check loaded models via ollama ps
    try:
        start = time.monotonic()
        resp = http.request("GET", f"{settings.OLLAMA_BASE_URL}/api/ps", timeout=5)
        ps_latency = (time.monotonic() - start) * 1000
        
        if resp.status == 200:
            import json
            data = json.loads(resp.data)
            running = data.get("models", [])
            if running:
                running_names = [m["name"] for m in running]
                ping_results.append(f"🟢 Loaded: {', '.join(running_names)} ({ps_latency:.0f}ms)")
            else:
                ping_results.append(f"⚪ No models currently loaded in VRAM")
        else:
            ping_results.append(f"🟡 /api/ps: {resp.status} ({ps_latency:.0f}ms)")
    except Exception as exc:
        ping_results.append(f"🔴 /api/ps: {exc}")
    
    # Config summary
    ping_results.append(
        f"⚙️ Config:\n"
        f"  Timeout: {settings.FALLBACK_TIMEOUT}s\n"
        f"  Keep-Alive: {settings.OLLAMA_KEEP_ALIVE}\n"
        f"  Router: {settings.ROUTER_MODEL}\n"
        f"  Code: {settings.CODE_MODEL}"
    )
    
    await update.message.reply_text("\n\n".join(ping_results), parse_mode=None)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses (delegates to capability nav if applicable)."""
    query = update.callback_query
    if not query:
        return

    data = query.data

    # Delegate to capability navigation
    if data.startswith("cap_"):
        await capability_nav_handler(update, context)
        return

    action = query.data
    responses = {
        "help_chat": (
            "💬 *Chat*\n\n"
            "Escribe cualquier pregunta y te respondo.\n"
            "Uso el chain de fallback para asegurar la mejor respuesta posible.\n\n"
            "Ejemplo: '¿qué es la fotosíntesis?'"
        ),
        "help_search": (
            "🔍 *Búsqueda Web*\n\n"
            "Busco en DuckDuckGo y sintetizo resultados.\n"
            "Después puedes decir 'deeper' para investigar más a fondo.\n\n"
            "Ejemplo: 'busca últimas noticias de IA'"
        ),
        "help_code": (
            "💻 *Generación de Código*\n\n"
            "Genero código production-ready y valido syntax con ast.parse.\n\n"
            "Ejemplo: 'escribe un decorator @retry en Python'"
        ),
        "help_rag": (
            "📚 *RAG Local*\n\n"
            "Busco en mi base de conocimiento (ChromaDB).\n"
            "Ingesta PDFs con: python -m src.memory.ingest /ruta/\n\n"
            "Ejemplo: '¿qué es la ley de Newton?'"
        ),
        "help_status": (
            "📊 *Status*\n\n"
            "Escribe /status para ver estadísticas completas."
        ),
    }
    await query.answer()
    await query.edit_message_text(
        responses.get(action, "No sé qué hacer con eso 😅"),
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route incoming text messages through the Metis graph."""
    chat_id = update.effective_chat.id if update.effective_chat else None

    # Access control
    if settings.ALLOWED_CHAT_IDS_SET and chat_id not in settings.ALLOWED_CHAT_IDS_SET:
        logger.warning("Blocked chat_id=%s", chat_id)
        await update.message.reply_text("🔒 Access denied.")
        return

    text = update.message.text
    if not text:
        return

    # Check if this is a fallback permission reply
    from src.utils.fallback_queue import submit_reply as _submit, _pending
    if chat_id in _pending:
        _submit(chat_id, text)
        return

    logger.info("Telegram message received chat_id=%s text='%s'", chat_id, text[:100])

    # Show typing indicator
    await update.message.chat.send_action("typing")

    # Run the graph
    try:
        graph = get_graph()
        initial_state = MetisState.from_query(text)
        initial_state.source = "telegram"

        result: dict[str, Any] = graph.invoke(initial_state.model_dump())

        response = result.get("response", "No response generated.")
        if len(response) > 4000:
            response = response[:3990] + "\n\n...(truncated)"

        # Escape markdown special chars for Telegram
        await update.message.reply_text(response, parse_mode=None)
    except Exception as exc:
        logger.exception("Graph execution failed for query='%s': %s", text[:80], exc)
        await update.message.reply_text(f"⚠️ Error processing your query: {exc}")


async def speak_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Text-to-speech command: /speak <text> [lang]."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("⚠️ Not authorized.")
        return

    if not context.args:
        await update.message.reply_text(
            "🗣️ *TTS — Text to Speech*\n\n"
            "Usage: `/speak <text>` or `/speak <text> --lang es`\n\n"
            "Languages: `en` (English), `es` (Español), `zh` (中文)\n"
            "Engines: `kokoro` (female voices, default), `piper` (fast, male)\n\n"
            "Example: `/speak Hola Metis --lang es`",
            parse_mode="Markdown",
        )
        return

    # Parse args: extract --lang and --engine flags
    text_parts = []
    lang = "en"
    engine = "kokoro"
    for arg in context.args:
        if arg.startswith("--lang="):
            lang = arg.split("=", 1)[1]
        elif arg.startswith("--engine="):
            engine = arg.split("=", 1)[1]
        else:
            text_parts.append(arg)

    text = " ".join(text_parts)
    if not text:
        await update.message.reply_text("⚠️ No text provided.")
        return

    if lang not in ("en", "es", "zh"):
        await update.message.reply_text(f"⚠️ Unsupported language: `{lang}`", parse_mode="Markdown")
        return

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")

    try:
        from src.tts import synthesize, VoiceEngine

        ve = VoiceEngine.PIPER if engine == "piper" else VoiceEngine.KOKORO
        wav_path = synthesize(text, lang=lang, engine=ve)

        if wav_path:
            with open(wav_path, "rb") as audio:
                await context.bot.send_voice(chat_id=chat_id, voice=audio)
            # Cleanup temp file
            try:
                os.remove(wav_path)
            except OSError:
                pass
        else:
            await update.message.reply_text("⚠️ TTS failed — could not generate audio.")
    except ImportError:
        await update.message.reply_text("⚠️ TTS module not installed. Run: pip install kokoro piper-tts")
    except Exception as exc:
        logger.exception("TTS failed: %s", exc)
        await update.message.reply_text(f"⚠️ TTS error: {exc}")


# --- Echo Pronunciation Practice Handlers ---

async def practice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start pronunciation practice: /practice [level] [language] or /practice: <custom text>."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("⚠️ Not authorized.")
        return

    try:
        from src.echo import EchoDatabase, EchoScorer, WhisperSTT, EchoTTS
        from src.graph.nodes import echo_agent
        from src.graph.state import MetisState
        from src.graph.orchestrator import get_graph
        
        user_id = str(chat_id)
        db = EchoDatabase()
        
        # Check for custom sentence
        custom_text = " ".join(context.args) if context.args else ""
        
        if custom_text and ":" in custom_text:
            # Custom sentence mode: /practice: The quick brown fox
            parts = custom_text.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                target = parts[1].strip()
                language = "en"  # Default
                
                # Check for language hint
                if "--lang es" in custom_text or "--lang=es" in custom_text:
                    language = "es"
                    target = target.replace("--lang es", "").replace("--lang=es", "").strip()
                
                response = f"📖 **Read this:**\n\n*{target}*\n\nNow send a 🎤 voice message reading it aloud!"
                await update.message.reply_text(response, parse_mode="Markdown")
                
                # Store target in user data for voice message handler
                context.user_data["echo_target"] = target
                context.user_data["echo_language"] = language
                context.user_data["echo_mode"] = "custom"
                return
        
        # Database sentence mode
        level = "A1"
        language = "en"
        
        # Parse level from args
        if context.args:
            for arg in context.args:
                if arg.upper() in ["A1", "A2", "B1", "B2", "C1"]:
                    level = arg.upper()
                elif arg.lower() in ["spanish", "español", "es"]:
                    language = "es"
                elif arg.lower() in ["english", "en"]:
                    language = "en"
        
        # Get sentence from database
        sentence_data = db.get_sentence(level=level, language=language)
        
        if not sentence_data:
            await update.message.reply_text(
                f"📭 No sentences found for level {level} in {language.upper()}.\n"
                f"Try: `/practice A1 spanish` or `/practice: <your text>`",
                parse_mode="Markdown",
            )
            return
        
        target = sentence_data["text"]
        language = sentence_data["language"]
        
        response = f"📖 **Read this:**\n\n*{target}*\n\nNow send a 🎤 voice message reading it aloud!"
        await update.message.reply_text(response, parse_mode="Markdown")
        
        # Store for voice message handler
        context.user_data["echo_target"] = target
        context.user_data["echo_language"] = language
        context.user_data["echo_mode"] = "database"
        context.user_data["echo_level"] = level
        
    except Exception as exc:
        logger.exception("Practice handler failed: %s", exc)
        await update.message.reply_text(f"⚠️ Echo practice failed: {exc}")


async def progress_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show practice progress: /progress."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("⚠️ Not authorized.")
        return

    try:
        from src.echo import EchoDatabase
        
        user_id = str(chat_id)
        db = EchoDatabase()
        
        progress = db.get_user_progress(user_id)
        
        if not progress:
            await update.message.reply_text(
                "📊 **No practice sessions yet!**\n\n"
                "Start practicing with `/practice` to see your progress here.",
                parse_mode="Markdown",
            )
            return
        
        total = progress.get("total_sessions", 0)
        avg = progress.get("avg_score", 0)
        streak = progress.get("streak_days", 0)
        level = progress.get("level", "A1")
        last = progress.get("last_practice", "Never")
        
        response = (
            f"📊 **Your Echo Progress**\n\n"
            f"**Level:** {level}\n"
            f"**Sessions completed:** {total}\n"
            f"**Average score:** {avg:.1f}%\n"
            f"**Streak:** {streak} days\n"
            f"**Last practice:** {last}\n\n"
            f"Keep practicing! 💪"
        )
        
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as exc:
        logger.exception("Progress handler failed: %s", exc)
        await update.message.reply_text(f"⚠️ Failed to get progress: {exc}")


async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages for Echo pronunciation practice."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    if not _is_chat_allowed(chat_id):
        await update.message.reply_text("⚠️ Not authorized.")
        return

    # Check if user is in Echo practice mode
    if not context.user_data.get("echo_target"):
        # Not in practice mode, ignore voice message
        return

    try:
        from src.echo import EchoScorer, WhisperSTT, EchoTTS, EchoDatabase
        
        user_id = str(chat_id)
        target_text = context.user_data["echo_target"]
        language = context.user_data.get("echo_language", "en")
        
        # Download voice message
        voice_file = await update.message.voice.get_file()
        voice_path = f"/tmp/echo_voice_{chat_id}_{int(time.time())}.ogg"
        await voice_file.download_to_drive(voice_path)
        
        # Send typing indicator
        await update.message.chat.send_action("typing")
        
        # Transcribe voice
        stt = WhisperSTT()
        transcription = stt.transcribe_telegram_voice(voice_path, language=language)
        
        if transcription["error"]:
            await update.message.reply_text(f"⚠️ Transcription failed: {transcription['error']}")
            return
        
        actual_text = transcription["text"]
        
        if not actual_text:
            await update.message.reply_text("⚠️ Couldn't detect speech in your voice message. Try again!")
            return
        
        # Score pronunciation
        scorer = EchoScorer()
        score_result = scorer.score(target_text, actual_text)
        
        # Format feedback
        feedback = scorer.format_feedback(score_result)
        
        # Generate correct pronunciation audio
        tts = EchoTTS(language=language)
        correct_audio_path = tts.generate(target_text)
        
        # Send feedback
        await update.message.reply_text(feedback, parse_mode="Markdown")
        
        # Send correct pronunciation audio
        if correct_audio_path and Path(correct_audio_path).exists():
            with open(correct_audio_path, "rb") as audio:
                await update.message.reply_voice(
                    voice=audio,
                    caption="🔊 Correct pronunciation",
                )
        
        # Flagged words - offer retry
        if score_result["flagged"]:
            await update.message.reply_text(
                f"🔄 Want to try again? Send another voice message, or type `/sentence` for a new practice.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"🌟 Excellent! All words pronounced correctly!\n"
                f"Type `/sentence` for a new practice or `/progress` to see your stats.",
                parse_mode="Markdown",
            )
        
        # Save session to database
        db = EchoDatabase()
        flagged_words = ",".join(f["word"] for f in score_result["flagged"])
        db.save_session(
            user_id=user_id,
            target_sentence=target_text,
            actual_transcription=actual_text,
            score=score_result["overall_score"],
            grade=score_result["grade"],
            flagged_words=flagged_words,
            language=language,
        )
        
        # Cleanup voice file
        try:
            os.remove(voice_path)
        except OSError:
            pass
        
        # Clear echo mode
        context.user_data.pop("echo_target", None)
        context.user_data.pop("echo_language", None)
        context.user_data.pop("echo_mode", None)
        context.user_data.pop("echo_level", None)
        
    except Exception as exc:
        logger.exception("Voice message handler failed: %s", exc)
        await update.message.reply_text(f"⚠️ Error processing voice message: {exc}")


def build_application() -> Any:
    """Build and return the telegram Application (does NOT start it)."""
    global _BOT_APP

    if not settings.TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is not set in .env")

    async def _post_init(application: Any) -> None:
        """Start scheduler inside the bot's event loop after init."""
        try:
            from src.scheduler import start_scheduler
            restored = start_scheduler()
            logger.info("Scheduler started (restored %d job(s))", restored)
        except Exception as exc:
            logger.exception("Failed to start scheduler: %s", exc)

    async def _post_shutdown(application: Any) -> None:
        try:
            from src.scheduler import shutdown_scheduler
            shutdown_scheduler()
        except Exception as exc:
            logger.warning("Error during scheduler shutdown: %s", exc)

    app = (
        ApplicationBuilder()
        .token(settings.TELEGRAM_TOKEN)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("run", run_handler))
    app.add_handler(CommandHandler("status", status_handler))
    app.add_handler(CommandHandler("cancel", cancel_handler))
    app.add_handler(CommandHandler("ping", ping_handler))
    app.add_handler(CommandHandler("capabilities", capabilities_handler))
    app.add_handler(CommandHandler("schedule", schedule_handler))
    app.add_handler(CommandHandler("schedules", schedules_list_handler))
    app.add_handler(CommandHandler("unschedule", unschedule_handler))
    app.add_handler(CommandHandler("runschedule", run_schedule_handler))
    app.add_handler(CommandHandler("speak", speak_handler))
    app.add_handler(CommandHandler("practice", practice_handler))
    app.add_handler(CommandHandler("progress", progress_handler))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, voice_message_handler))

    set_telegram_app(app)

    # Store global reference for graceful shutdown
    _BOT_APP = app

    logger.info("Telegram bot handlers registered")
    return app


def run_polling() -> None:
    """Start the bot in polling mode (blocking). Uses OS-level singleton lock."""
    # Step 1: Acquire the singleton lock — exits if already running
    _acquire_singleton_lock()

    # Step 2: Register signal handlers for graceful shutdown
    _register_signal_handlers()

    # Step 3: Build and start the bot
    app = build_application()
    logger.info("Starting Telegram bot (polling mode)...")

    run_kwargs: dict[str, Any] = {
        "drop_pending_updates": True,
        "allowed_updates": Update.ALL_TYPES,
    }

    if "error_callback" in inspect.signature(app.run_polling).parameters:
        from telegram.error import Conflict, TelegramError

        def _polling_error_callback(exc: TelegramError) -> None:
            if isinstance(exc, Conflict):
                logger.error(
                    "Telegram conflict detected (409). Another process is polling this bot token. "
                    "Verify only one Metis instance runs for this token."
                )
                _graceful_shutdown()
                return
            logger.warning("Polling error: %s", exc)

        run_kwargs["error_callback"] = _polling_error_callback

    try:
        app.run_polling(**run_kwargs)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except SystemExit:
        # Signal handler triggered sys.exit — this is expected
        raise
    finally:
        _graceful_shutdown()
