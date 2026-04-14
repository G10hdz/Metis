"""SQLite-backed persistence for scheduled tasks."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

_DB_PATH: Path = settings.CHROMA_DIR.parent / "schedules.db"  # ~/.metis/schedules.db

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS schedules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT    UNIQUE NOT NULL,
    chat_id         INTEGER NOT NULL,
    name            TEXT    NOT NULL,
    schedule_type   TEXT    NOT NULL,   -- 'cron' | 'interval' | 'date'
    schedule_expr   TEXT    NOT NULL,   -- raw user expression
    trigger_kwargs  TEXT    NOT NULL,   -- JSON-encoded APScheduler kwargs
    query           TEXT    NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    last_run        TEXT,
    last_status     TEXT,
    last_response   TEXT,
    run_count       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sched_chat ON schedules(chat_id);
CREATE INDEX IF NOT EXISTS idx_sched_enabled ON schedules(enabled);
"""


class ScheduleStore:
    """Thread-safe SQLite wrapper for persisted schedules."""

    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_INIT_SQL)
        self._db.commit()
        logger.info("ScheduleStore initialized at %s", _DB_PATH)

    def add(
        self,
        chat_id: int,
        name: str,
        schedule_type: str,
        schedule_expr: str,
        trigger_kwargs_json: str,
        query: str,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:8]
        cur = self._db.execute(
            """
            INSERT INTO schedules
                (job_id, chat_id, name, schedule_type, schedule_expr, trigger_kwargs, query)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, chat_id, name, schedule_type, schedule_expr, trigger_kwargs_json, query),
        )
        self._db.commit()
        row = self._db.execute("SELECT * FROM schedules WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row) if row else {}

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM schedules WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_all(self, chat_id: int | None = None, only_enabled: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM schedules WHERE 1=1"
        params: list[Any] = []
        if chat_id is not None:
            sql += " AND chat_id = ?"
            params.append(chat_id)
        if only_enabled:
            sql += " AND enabled = 1"
        sql += " ORDER BY created_at DESC"
        rows = self._db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def delete(self, job_id: str) -> bool:
        cur = self._db.execute("DELETE FROM schedules WHERE job_id = ?", (job_id,))
        self._db.commit()
        return cur.rowcount > 0

    def set_enabled(self, job_id: str, enabled: bool) -> bool:
        cur = self._db.execute(
            "UPDATE schedules SET enabled = ? WHERE job_id = ?",
            (1 if enabled else 0, job_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    def record_run(
        self,
        job_id: str,
        status: str,
        response: str = "",
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._db.execute(
            """
            UPDATE schedules
            SET last_run = ?, last_status = ?, last_response = ?, run_count = run_count + 1
            WHERE job_id = ?
            """,
            (ts, status, response[:2000], job_id),
        )
        self._db.commit()


_store: ScheduleStore | None = None


def get_schedule_store() -> ScheduleStore:
    global _store
    if _store is None:
        _store = ScheduleStore()
    return _store
