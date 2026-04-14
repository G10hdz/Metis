"""Thread-safe message queue for Telegram fallback interrupt.

When the fallback chain needs user permission, it puts a pending request
in this queue and waits for the Telegram handler to push the reply back.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_pending: dict[int, dict[str, Any]] = {}  # chat_id -> {"event": Event, "reply": str}
_lock = threading.Lock()

_telegram_app: Any | None = None


def set_telegram_app(app: Any) -> None:
    """Store reference to the telegram.ext.Application for sending messages."""
    global _telegram_app
    _telegram_app = app


def get_telegram_app() -> Any | None:
    return _telegram_app


def submit_request(chat_id: int) -> threading.Event:
    """Register a pending request for a chat_id. Returns an Event to wait on."""
    event = threading.Event()
    with _lock:
        _pending[chat_id] = {"event": event, "reply": ""}
    return event


def submit_reply(chat_id: int, reply: str) -> None:
    """Submit the user's reply. Wakes up the waiting thread."""
    with _lock:
        if chat_id in _pending:
            _pending[chat_id]["reply"] = reply
            _pending[chat_id]["event"].set()


def wait_for(chat_id: int, timeout: float = 10) -> str:
    """Wait for a reply on a pending request. Returns reply or empty on timeout."""
    event = None
    with _lock:
        if chat_id in _pending:
            event = _pending[chat_id]["event"]

    if event is None:
        return ""

    got = event.wait(timeout)
    with _lock:
        reply = _pending.pop(chat_id, {}).get("reply", "")

    return reply if got else ""


def get_fallback_queue():
    """Return this module as the queue interface."""
    import sys
    return sys.modules[__name__]
