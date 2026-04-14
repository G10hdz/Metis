"""SQLite telemetry store — unified conversation log across all interaction channels."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

_DB_PATH: Path = settings.CHROMA_DIR.parent / "metis.db"  # ~/.metis/metis.db

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    query       TEXT    NOT NULL,
    route       TEXT    NOT NULL,
    model       TEXT    DEFAULT '',
    latency_ms  REAL    DEFAULT 0,
    response    TEXT    DEFAULT '',
    error       TEXT    DEFAULT '',
    source      TEXT    DEFAULT 'cli'  -- 'telegram' | 'web' | 'cli'
);

CREATE TABLE IF NOT EXISTS copilot_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    hour    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:00:00Z','now')),
    count   INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversations(ts);
CREATE INDEX IF NOT EXISTS idx_conv_route ON conversations(route);
CREATE INDEX IF NOT EXISTS idx_conv_source ON conversations(source);
CREATE INDEX IF NOT EXISTS idx_copilot_hour ON copilot_usage(hour);
"""


class TelemetryStore:
    """Thread-safe SQLite telemetry wrapper."""

    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_INIT_SQL)
        self._db.commit()
        logger.info("TelemetryStore initialized at %s", _DB_PATH)

    # ---- write ----

    def log(
        self,
        query: str,
        route: str,
        response: str = "",
        model: str = "",
        latency_ms: float = 0,
        error: str = "",
        source: str = "cli",
    ) -> int:
        """Insert a conversation record. Returns the new row id."""
        cur = self._db.execute(
            "INSERT INTO conversations (query, route, model, latency_ms, response, error, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query, route, model, latency_ms, response, error, source),
        )
        self._db.commit()
        return cur.lastrowid

    # ---- read ----

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get most recent conversations."""
        rows = self._db.execute(
            "SELECT * FROM conversations ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard."""
        total = self._db.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
        errors = self._db.execute("SELECT COUNT(*) as c FROM conversations WHERE error != ''").fetchone()["c"]
        avg_latency = self._db.execute(
            "SELECT AVG(latency_ms) as a FROM conversations WHERE latency_ms > 0"
        ).fetchone()["a"] or 0

        # Route distribution
        route_rows = self._db.execute(
            "SELECT route, COUNT(*) as c FROM conversations GROUP BY route ORDER BY c DESC"
        ).fetchall()
        routes = {r["route"]: r["c"] for r in route_rows}

        # Source distribution
        source_rows = self._db.execute(
            "SELECT source, COUNT(*) as c FROM conversations GROUP BY source ORDER BY c DESC"
        ).fetchall()
        sources = {r["source"]: r["c"] for r in source_rows}

        # Top 10 slowest
        slow_rows = self._db.execute(
            "SELECT query, route, model, latency_ms, ts FROM conversations "
            "WHERE latency_ms > 0 ORDER BY latency_ms DESC LIMIT 10"
        ).fetchall()
        slowest = [dict(r) for r in slow_rows]

        return {
            "total": total,
            "errors": errors,
            "avg_latency_ms": round(avg_latency, 1),
            "routes": routes,
            "sources": sources,
            "slowest": slowest,
        }

    def latency_series(self, limit: int = 100) -> list[dict[str, Any]]:
        """Time-series of latencies for charting."""
        rows = self._db.execute(
            "SELECT ts, latency_ms, route FROM conversations "
            "WHERE latency_ms > 0 ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- copilot usage ----

    def copilot_increment(self) -> None:
        """Increment Copilot usage counter for the current hour."""
        self._db.execute(
            "INSERT INTO copilot_usage (hour, count) VALUES "
            "(strftime('%Y-%m-%dT%H:00:00Z','now'), 1) "
            "ON CONFLICT DO NOTHING"
        )
        # Use UPSERT via hour match
        self._db.execute(
            "UPDATE copilot_usage SET count = count + 1 "
            "WHERE hour = strftime('%Y-%m-%dT%H:00:00Z','now')"
        )
        self._db.commit()

    def copilot_remaining(self) -> int:
        """Get remaining Copilot calls this hour (max 10)."""
        rows = self._db.execute(
            "SELECT COALESCE(SUM(count), 0) as c FROM copilot_usage "
            "WHERE hour = strftime('%Y-%m-%dT%H:00:00Z','now')"
        ).fetchone()
        count = rows["c"] if rows else 0
        return max(0, 10 - count)

    # ---- lifecycle ----

    def close(self) -> None:
        self._db.close()


# --- Singleton ---
_instance: TelemetryStore | None = None


def get_telemetry() -> TelemetryStore:
    global _instance
    if _instance is None:
        _instance = TelemetryStore()
    return _instance


def reset_telemetry() -> None:
    global _instance
    if _instance is not None:
        _instance.close()
    _instance = None
