"""Ollama client wrappers with VRAM guard (keep_alive=0s)."""

import time
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)


def _make_kwargs(model: str) -> dict:
    return {
        "model": model,
        "base_url": settings.OLLAMA_BASE_URL,
        "temperature": 0.0,
    }


def get_chat_model(model: str | None = None) -> ChatOllama:
    """Return a ChatOllama instance with VRAM-safe defaults."""
    model_name = model or settings.GENERAL_MODEL
    kwargs = _make_kwargs(model_name)
    # keep_alive is passed via model_kwargs for langchain-ollama >= 0.2
    kwargs["keep_alive"] = settings.OLLAMA_KEEP_ALIVE
    return ChatOllama(**kwargs)


def get_embedding_model() -> OllamaEmbeddings:
    """Return embedding model (nomic-embed-text)."""
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def log_call(model: str, latency: float, token_count: int = 0):
    """Append a line to the VRAM guard log."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] model={model} latency={latency:.2f}s tokens={token_count}\n"
    try:
        with open(settings.VRAM_GUARD_LOG, "a") as f:
            f.write(line)
    except OSError:
        pass  # non-critical
    logger.info(line.strip())
