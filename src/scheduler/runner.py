"""APScheduler integration — proactive scheduled task execution."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.scheduler.store import get_schedule_store

logger = logging.getLogger(__name__)


class ScheduleParseError(ValueError):
    """Raised when a schedule expression cannot be parsed."""


_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler


# ---------------------------------------------------------------------------
# Natural language parser
# ---------------------------------------------------------------------------
# Supported syntaxes:
#   daily HH:MM <query>
#   hourly <query>
#   every Nh|Nm|Ns <query>
#   weekly DAY HH:MM <query>             (DAY = mon|tue|wed|thu|fri|sat|sun)
#   cron "EXPR" <query>                  (5-field crontab)

_DAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_INTERVAL_RE = re.compile(r"^(\d+)([hms])$")


def parse_schedule_expr(expr: str) -> tuple[str, str, dict[str, Any], str]:
    """Parse a schedule expression and the trailing query.

    Returns: (schedule_type, raw_expr, trigger_kwargs, query)
    Raises: ScheduleParseError if invalid
    """
    expr = expr.strip()
    if not expr:
        raise ScheduleParseError("Empty expression")

    tokens = expr.split()
    keyword = tokens[0].lower()

    # cron "EXPR" <query>
    if keyword == "cron":
        match = re.match(r'^cron\s+"([^"]+)"\s+(.+)$', expr)
        if not match:
            match = re.match(r"^cron\s+'([^']+)'\s+(.+)$", expr)
        if not match:
            raise ScheduleParseError(
                'Cron syntax: cron "MIN HOUR DOM MON DOW" <query>\n'
                'Example: cron "0 10 * * *" search latest devops news'
            )
        cron_expr, query = match.group(1).strip(), match.group(2).strip()
        try:
            trig = CronTrigger.from_crontab(cron_expr)
        except Exception as exc:
            raise ScheduleParseError(f"Invalid cron expression: {exc}") from exc
        return "cron", f"cron {cron_expr}", _trigger_to_kwargs(trig, "cron"), query

    # daily HH:MM <query>
    if keyword == "daily":
        if len(tokens) < 3:
            raise ScheduleParseError("Usage: daily HH:MM <query>")
        time_match = _TIME_RE.match(tokens[1])
        if not time_match:
            raise ScheduleParseError(f"Invalid time format: {tokens[1]} (use HH:MM)")
        hour, minute = int(time_match.group(1)), int(time_match.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ScheduleParseError(f"Time out of range: {tokens[1]}")
        query = " ".join(tokens[2:]).strip()
        if not query:
            raise ScheduleParseError("Missing query for daily schedule")
        return (
            "cron",
            f"daily {hour:02d}:{minute:02d}",
            {"hour": hour, "minute": minute},
            query,
        )

    # hourly <query>
    if keyword == "hourly":
        if len(tokens) < 2:
            raise ScheduleParseError("Usage: hourly <query>")
        query = " ".join(tokens[1:]).strip()
        return "interval", "hourly", {"hours": 1}, query

    # every Nh|Nm|Ns <query>
    if keyword == "every":
        if len(tokens) < 3:
            raise ScheduleParseError("Usage: every Nh|Nm|Ns <query>  (e.g. every 2h check status)")
        interval_match = _INTERVAL_RE.match(tokens[1])
        if not interval_match:
            raise ScheduleParseError(f"Invalid interval: {tokens[1]} (use 30m, 2h, 90s)")
        n = int(interval_match.group(1))
        unit = interval_match.group(2)
        if n <= 0:
            raise ScheduleParseError("Interval must be positive")
        kwargs_map = {"h": "hours", "m": "minutes", "s": "seconds"}
        kwargs = {kwargs_map[unit]: n}
        query = " ".join(tokens[2:]).strip()
        if not query:
            raise ScheduleParseError("Missing query")
        return "interval", f"every {n}{unit}", kwargs, query

    # weekly DAY HH:MM <query>
    if keyword == "weekly":
        if len(tokens) < 4:
            raise ScheduleParseError("Usage: weekly DAY HH:MM <query>  (DAY = mon..sun)")
        day = tokens[1].lower()[:3]
        if day not in _DAYS:
            raise ScheduleParseError(f"Invalid day: {tokens[1]} (use mon, tue, ..., sun)")
        time_match = _TIME_RE.match(tokens[2])
        if not time_match:
            raise ScheduleParseError(f"Invalid time format: {tokens[2]} (use HH:MM)")
        hour, minute = int(time_match.group(1)), int(time_match.group(2))
        query = " ".join(tokens[3:]).strip()
        if not query:
            raise ScheduleParseError("Missing query")
        return (
            "cron",
            f"weekly {day} {hour:02d}:{minute:02d}",
            {"day_of_week": day, "hour": hour, "minute": minute},
            query,
        )

    raise ScheduleParseError(
        "Unknown schedule type. Supported:\n"
        "  daily HH:MM <query>\n"
        "  hourly <query>\n"
        "  every Nh|Nm|Ns <query>\n"
        "  weekly DAY HH:MM <query>\n"
        '  cron "MIN HOUR DOM MON DOW" <query>'
    )


def _trigger_to_kwargs(trig: Any, _kind: str) -> dict[str, Any]:
    """Best-effort serialization of CronTrigger fields back to kwargs."""
    kwargs: dict[str, Any] = {}
    for field in getattr(trig, "fields", []):
        name = field.name
        rep = str(field)
        if rep != "*":
            kwargs[name] = rep
    return kwargs


def _build_trigger(schedule_type: str, kwargs: dict[str, Any]) -> Any:
    if schedule_type == "cron":
        return CronTrigger(**kwargs)
    if schedule_type == "interval":
        return IntervalTrigger(**kwargs)
    raise ValueError(f"Unsupported schedule_type: {schedule_type}")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


async def _execute_scheduled_job(job_id: str) -> None:
    """Run the graph for a scheduled job and notify the chat."""
    store = get_schedule_store()
    record = store.get(job_id)
    if not record:
        logger.warning("Scheduled job %s not found in store", job_id)
        return
    if not record.get("enabled"):
        logger.info("Scheduled job %s is disabled, skipping", job_id)
        return

    chat_id = int(record["chat_id"])
    query = str(record["query"])
    name = str(record["name"])

    # Lazy import to avoid circular import at module load
    from src.telegram.bot import _BOT_APP  # type: ignore
    from src.graph.orchestrator import get_graph
    from src.graph.state import MetisState

    if _BOT_APP is None:
        logger.error("Cannot run scheduled job %s: Telegram app not initialized", job_id)
        store.record_run(job_id, "failed", "Telegram app not initialized")
        return

    logger.info("Running scheduled job %s ('%s'): %s", job_id, name, query[:80])

    try:
        await _BOT_APP.bot.send_message(
            chat_id=chat_id,
            text=f"⏰ Scheduled task `{job_id}` — *{name}*\nRunning: _{query[:100]}_",
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
        store.record_run(job_id, "done", response)

        truncated = response if len(response) <= 3900 else response[:3890] + "\n\n...(truncated)"
        await _BOT_APP.bot.send_message(
            chat_id=chat_id,
            text=f"✅ Scheduled task `{job_id}` complete:",
            parse_mode="Markdown",
        )
        await _BOT_APP.bot.send_message(chat_id=chat_id, text=truncated, parse_mode=None)

    except Exception as exc:
        logger.exception("Scheduled job %s failed: %s", job_id, exc)
        store.record_run(job_id, "failed", str(exc))
        try:
            await _BOT_APP.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Scheduled task `{job_id}` failed: {exc}",
                parse_mode=None,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_schedule(
    chat_id: int,
    name: str,
    expr: str,
) -> dict[str, Any]:
    """Parse expr, persist the schedule, and add it to the running scheduler."""
    schedule_type, raw_expr, trigger_kwargs, query = parse_schedule_expr(expr)

    store = get_schedule_store()
    record = store.add(
        chat_id=chat_id,
        name=name,
        schedule_type=schedule_type,
        schedule_expr=raw_expr,
        trigger_kwargs_json=json.dumps(trigger_kwargs),
        query=query,
    )

    job_id = record["job_id"]
    scheduler = get_scheduler()
    trigger = _build_trigger(schedule_type, trigger_kwargs)
    scheduler.add_job(
        _execute_scheduled_job,
        trigger=trigger,
        args=[job_id],
        id=job_id,
        replace_existing=True,
        misfire_grace_time=300,
    )
    logger.info("Registered schedule %s: %s -> %s", job_id, raw_expr, query[:80])
    return record


def unregister_schedule(job_id: str) -> bool:
    """Remove a schedule from both the scheduler and the store."""
    scheduler = get_scheduler()
    try:
        scheduler.remove_job(job_id)
    except Exception as exc:
        logger.debug("Job %s not in scheduler (may have been removed): %s", job_id, exc)
    return get_schedule_store().delete(job_id)


def _restore_persisted_jobs() -> int:
    """Re-register all enabled schedules from SQLite into the scheduler."""
    store = get_schedule_store()
    scheduler = get_scheduler()
    schedules = store.list_all(only_enabled=True)
    restored = 0
    for sched in schedules:
        try:
            kwargs = json.loads(sched["trigger_kwargs"])
            trigger = _build_trigger(sched["schedule_type"], kwargs)
            scheduler.add_job(
                _execute_scheduled_job,
                trigger=trigger,
                args=[sched["job_id"]],
                id=sched["job_id"],
                replace_existing=True,
                misfire_grace_time=300,
            )
            restored += 1
        except Exception as exc:
            logger.error("Failed to restore schedule %s: %s", sched.get("job_id"), exc)
    return restored


def start_scheduler() -> int:
    """Start the AsyncIOScheduler and restore persisted jobs. Call from inside the event loop."""
    scheduler = get_scheduler()
    if scheduler.running:
        logger.info("Scheduler already running")
        return 0
    restored = _restore_persisted_jobs()
    scheduler.start()
    logger.info("Scheduler started with %d persisted job(s)", restored)
    return restored


def shutdown_scheduler() -> None:
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        try:
            _scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down")
        except Exception as exc:
            logger.warning("Error shutting down scheduler: %s", exc)
