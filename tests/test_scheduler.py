"""Tests for the scheduler module — parser and persistence."""

from __future__ import annotations

import json

import pytest

from src.scheduler.runner import parse_schedule_expr, ScheduleParseError


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseScheduleExpr:
    def test_daily(self):
        t, raw, kwargs, query = parse_schedule_expr("daily 10:00 search devops news")
        assert t == "cron"
        assert kwargs == {"hour": 10, "minute": 0}
        assert query == "search devops news"
        assert raw == "daily 10:00"

    def test_daily_padded(self):
        t, _, kwargs, _ = parse_schedule_expr("daily 09:05 hola")
        assert kwargs == {"hour": 9, "minute": 5}

    def test_hourly(self):
        t, raw, kwargs, query = parse_schedule_expr("hourly ping ollama")
        assert t == "interval"
        assert kwargs == {"hours": 1}
        assert query == "ping ollama"

    def test_every_minutes(self):
        t, _, kwargs, query = parse_schedule_expr("every 30m check status")
        assert t == "interval"
        assert kwargs == {"minutes": 30}
        assert query == "check status"

    def test_every_hours(self):
        _, _, kwargs, _ = parse_schedule_expr("every 2h fetch news")
        assert kwargs == {"hours": 2}

    def test_every_seconds(self):
        _, _, kwargs, _ = parse_schedule_expr("every 90s heartbeat")
        assert kwargs == {"seconds": 90}

    def test_weekly(self):
        t, raw, kwargs, query = parse_schedule_expr("weekly fri 18:00 weekly review")
        assert t == "cron"
        assert kwargs == {"day_of_week": "fri", "hour": 18, "minute": 0}
        assert query == "weekly review"

    def test_weekly_long_day_truncated(self):
        _, _, kwargs, _ = parse_schedule_expr("weekly monday 09:00 standup")
        assert kwargs["day_of_week"] == "mon"

    def test_cron_double_quotes(self):
        t, _, kwargs, query = parse_schedule_expr('cron "0 9 * * 1-5" buenos dias')
        assert t == "cron"
        assert query == "buenos dias"
        assert kwargs.get("hour") == "9"
        assert kwargs.get("day_of_week") == "1-5"

    def test_cron_single_quotes(self):
        t, _, _, query = parse_schedule_expr("cron '*/5 * * * *' polling")
        assert t == "cron"
        assert query == "polling"

    # --- error cases ---

    def test_empty_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("")

    def test_unknown_keyword_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("foo bar baz")

    def test_daily_invalid_time_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("daily 25:00 query")

    def test_daily_missing_query_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("daily 10:00")

    def test_every_zero_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("every 0h query")

    def test_every_invalid_format_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("every banana query")

    def test_weekly_invalid_day_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("weekly xyz 10:00 query")

    def test_cron_unquoted_raises(self):
        with pytest.raises(ScheduleParseError):
            parse_schedule_expr("cron 0 9 * * * query")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Build a fresh ScheduleStore writing to a temp DB."""
    from src.scheduler import store as store_module

    monkeypatch.setattr(store_module, "_DB_PATH", tmp_path / "schedules.db")
    monkeypatch.setattr(store_module, "_store", None)
    return store_module.ScheduleStore()


class TestScheduleStore:
    def test_add_and_get(self, isolated_store):
        rec = isolated_store.add(
            chat_id=42,
            name="news",
            schedule_type="cron",
            schedule_expr="daily 10:00",
            trigger_kwargs_json=json.dumps({"hour": 10, "minute": 0}),
            query="search news",
        )
        assert rec["job_id"]
        assert rec["chat_id"] == 42
        assert rec["enabled"] == 1
        assert rec["run_count"] == 0

        got = isolated_store.get(rec["job_id"])
        assert got is not None
        assert got["name"] == "news"

    def test_list_all_filters_by_chat(self, isolated_store):
        isolated_store.add(1, "a", "interval", "hourly", "{}", "q1")
        isolated_store.add(2, "b", "interval", "hourly", "{}", "q2")
        isolated_store.add(1, "c", "interval", "hourly", "{}", "q3")

        for_chat_1 = isolated_store.list_all(chat_id=1)
        assert len(for_chat_1) == 2
        assert all(r["chat_id"] == 1 for r in for_chat_1)

    def test_delete_removes_record(self, isolated_store):
        rec = isolated_store.add(1, "x", "interval", "hourly", "{}", "q")
        assert isolated_store.delete(rec["job_id"]) is True
        assert isolated_store.get(rec["job_id"]) is None
        assert isolated_store.delete(rec["job_id"]) is False

    def test_record_run_updates_counters(self, isolated_store):
        rec = isolated_store.add(1, "x", "interval", "hourly", "{}", "q")
        isolated_store.record_run(rec["job_id"], "done", "result text")
        isolated_store.record_run(rec["job_id"], "done", "another")

        got = isolated_store.get(rec["job_id"])
        assert got["run_count"] == 2
        assert got["last_status"] == "done"
        assert got["last_response"] == "another"

    def test_set_enabled(self, isolated_store):
        rec = isolated_store.add(1, "x", "interval", "hourly", "{}", "q")
        assert isolated_store.set_enabled(rec["job_id"], False) is True
        assert isolated_store.get(rec["job_id"])["enabled"] == 0

        only_enabled = isolated_store.list_all(only_enabled=True)
        assert all(r["enabled"] == 1 for r in only_enabled)
