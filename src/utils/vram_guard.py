"""Synchronous wrapper for Ollama calls that enforces keep_alive=0s and logs usage."""

import time
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import LLMResult

from src.config import ollama as ollama_cfg
from src.config import settings

logger = logging.getLogger(__name__)


def vram_call(model: str | None = None, messages: list[dict] | None = None) -> str:
    """
    Synchronous LLM call with VRAM guard.

    Parameters
    ----------
    model : str | None
        Override model name. Defaults to GENERAL_MODEL.
    messages : list[dict] | None
        List of {"role": ..., "content": ...} dicts. Defaults to [{"role": "user", "content": ""}].

    Returns
    -------
    str — the assistant's text reply.
    """
    model_name = model or settings.GENERAL_MODEL
    if messages is None:
        messages = [{"role": "user", "content": ""}]

    lc_messages: list[BaseMessage] = [
        HumanMessage(content=m["content"]) if m["role"] in ("user", "system")
        else AIMessage(content=m["content"])
        for m in messages
    ]

    llm = ollama_cfg.get_chat_model(model_name)
    start = time.monotonic()

    try:
        result = llm.invoke(lc_messages)
        latency = time.monotonic() - start
        text = result.content if isinstance(result.content, str) else str(result.content)
        token_estimate = len(text.split())  # rough estimate
        ollama_cfg.log_call(model_name, latency, token_estimate)
        return text
    except Exception as exc:
        latency = time.monotonic() - start
        logger.error("vram_call failed model=%s error=%s latency=%.2fs", model_name, exc, latency)
        raise


def vram_call_structured(
    prompt: str,
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Single-turn convenience wrapper."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return vram_call(model=model, messages=messages)
