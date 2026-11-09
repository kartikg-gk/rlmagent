"""Minute-resolution wall-clock schedules for durable worker jobs."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
}


def _field(text: str, low: int, high: int) -> set[int]:
    values = set()
    for part in text.split(","):
        span, slash, stride = part.partition("/")
        if slash and (not stride.isascii() or not stride.isdecimal() or not int(stride)):
            raise ValueError("cron steps must be positive integers")
        step = int(stride) if slash else 1
        if span == "*":
            start, end = low, high
        else:
            first, dash, last = span.partition("-")
            endpoints = (first, last) if dash else (first,)
            if not all(item.isascii() and item.isdecimal() for item in endpoints):
                raise ValueError("cron fields accept numbers, ranges, lists, and steps")
            start = int(first)
            end = int(last) if dash else high if slash else start
        if not low <= start <= end <= high:
            raise ValueError(f"cron field must be within {low}–{high}")
        values.update(range(start, end + 1, step))
    return values


def next_run(expression: str, timezone: str, after: float) -> float:
    """Find the next occurrence, skipping missing times and running both repeated times."""
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("cron expression must be text of at most 256 characters")
    expression = expression.strip()
    fields = _ALIASES.get(expression, expression).split()
    if len(fields) != 5:
        raise ValueError("a cron schedule is 5 space-separated parts (minute hour day-of-month month day-of-week)")
    if not isinstance(timezone, str) or not timezone:
        raise ValueError("cron needs an IANA timezone such as UTC or America/New_York")
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown IANA timezone {timezone!r}") from exc

    bounds = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
    minutes, hours, days, months, weekdays = (
        _field(text, *limits) for text, limits in zip(fields, bounds, strict=True)
    )
    if 7 in weekdays:
        weekdays.add(0)
    times = [(hour, minute) for hour in sorted(hours) for minute in sorted(minutes)]
    start = datetime.fromtimestamp(after, zone).date()
    for offset in range(9 * 366):
        date = start + timedelta(days=offset)
        weekday = (date.weekday() + 1) % 7
        if date.month not in months or date.day not in days or weekday not in weekdays:
            continue

        earliest = None
        for hour, minute in times:
            wall = datetime(date.year, date.month, date.day, hour, minute)
            for fold in (0, 1):
                candidate = wall.replace(tzinfo=zone, fold=fold).astimezone(UTC)
                stamp = candidate.timestamp()
                if stamp <= after or (earliest is not None and stamp >= earliest):
                    continue
                if candidate.astimezone(zone).replace(tzinfo=None) == wall:
                    earliest = stamp
        if earliest is not None:
            return earliest
    raise ValueError("cron expression has no occurrence in the next nine years")
