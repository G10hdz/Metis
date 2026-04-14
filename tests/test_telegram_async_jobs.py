"""Tests for Telegram async job helpers (/run, /status <id>, /cancel <id>)."""

import pytest

from src.telegram import bot


@pytest.fixture(autouse=True)
def _reset_async_job_state():
    """Isolate module-level async job globals between tests."""
    bot._ASYNC_JOBS.clear()
    bot._ASYNC_JOB_TASKS.clear()
    bot._ASYNC_JOBS_BY_CHAT.clear()
    yield
    bot._ASYNC_JOBS.clear()
    bot._ASYNC_JOB_TASKS.clear()
    bot._ASYNC_JOBS_BY_CHAT.clear()


def test_parse_job_id_from_status_command():
    assert bot._parse_job_id_from_status("/status abc123ef") == "abc123ef"
    assert bot._parse_job_id_from_status("/status") is None


def test_register_job_and_active_count():
    chat_id = 1001
    j1 = bot._register_job(chat_id, "first task")
    j2 = bot._register_job(chat_id, "second task")

    assert j1["job_id"] != j2["job_id"]
    assert bot._active_jobs_for_chat(chat_id) == 2

    bot._ASYNC_JOBS[j1["job_id"]]["status"] = "done"
    assert bot._active_jobs_for_chat(chat_id) == 1


def test_trim_history_discards_finished_jobs_first(monkeypatch):
    chat_id = 2002
    monkeypatch.setattr(bot, "_ASYNC_HISTORY_PER_CHAT", 2)

    j1 = bot._register_job(chat_id, "old done")
    bot._ASYNC_JOBS[j1["job_id"]]["status"] = "done"

    j2 = bot._register_job(chat_id, "running")
    bot._ASYNC_JOBS[j2["job_id"]]["status"] = "running"

    j3 = bot._register_job(chat_id, "new queued")

    ids = bot._ASYNC_JOBS_BY_CHAT[chat_id]
    assert j1["job_id"] not in ids
    assert j1["job_id"] not in bot._ASYNC_JOBS
    assert j2["job_id"] in ids
    assert j3["job_id"] in ids
