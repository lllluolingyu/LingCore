"""Bounded recurrence expansion and calendar result formatting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta

from lingcore.bundled_plugins.calendar.icalendar import (
    Stored,
    _duration_end,
    _key,
    _rule_bounds,
)
from lingcore.bundled_plugins.calendar.models import (
    _UTC,
    Event,
    RepeatArgs,
    _advance,
    _ambiguous,
    _length,
)
from lingcore.timezones import WEEKDAYS, Zone, format_offset

_MAX_LISTED = 100
_MAX_OVERLAPS = 3
_MAX_STEPS = 50_000  # repeat-expansion budget per series


def _shift(start: date | datetime, freq: str, steps: int) -> date | datetime | None:
    """The ``steps``-th period after ``start``; ``None`` when that date is invalid."""
    if freq == "DAILY":
        return start + timedelta(days=steps)
    if freq == "WEEKLY":
        return start + timedelta(weeks=steps)
    if freq == "MONTHLY":
        months = start.month - 1 + steps
        year, month = start.year + months // 12, months % 12 + 1
    else:
        year, month = start.year + steps, start.month
    try:
        return start.replace(year=year, month=month)
    except ValueError:  # e.g. the 31st in a 30-day month: skipped (RFC 5545)
        return None


def occurrences(
    event: Stored, zone: Zone, window: tuple[datetime, datetime]
) -> tuple[list[tuple[date | datetime, date | datetime]], bool]:
    """Occurrences overlapping ``window``, and whether expansion was cut off."""
    begin, finish = (_key(value, zone) for value in window)
    length = _length(event.start, event.end)
    # Unknown series are surfaced separately as incomplete coverage, never
    # presented as confirmed occurrences or definite scheduling conflicts.
    if event.recurring and not event.expandable:
        return [], False
    excluded = {_key(value, zone) for value in event.exdates}
    found: dict[datetime, tuple[date | datetime, date | datetime]] = {}

    def collect(start: date | datetime) -> None:
        key = _key(start, zone)
        if event.duration is not None:
            if isinstance(start, datetime) and isinstance(event.start, datetime):
                start = start.astimezone(event.start.tzinfo)
            end = _duration_end(start, event.duration)
        else:
            end = _advance(start, length)
        if (
            key not in excluded
            and key < finish
            and (_key(end, zone) > begin or (not length and key >= begin))
        ):
            found[key] = (start, end)

    def result(
        cut: bool = False,
    ) -> tuple[list[tuple[date | datetime, date | datetime]], bool]:
        return [found[key] for key in sorted(found)], cut

    for extra in event.rdates:
        collect(extra)
    if event.rule is None:
        collect(event.start)
        return result()
    rule = event.rule
    interval, count, limit = _rule_bounds(rule, zone)
    seen = 0
    for n in range(_MAX_STEPS):
        start = _shift(event.start, rule["FREQ"], n * interval)
        if start is None:
            continue
        if isinstance(start, datetime) and (
            start.astimezone(_UTC).astimezone(start.tzinfo).replace(tzinfo=None)
            != start.replace(tzinfo=None)
        ):
            # RFC 5545: nonexistent local times are skipped without consuming
            # COUNT, just like an invalid day of the month.
            continue
        if limit is not None and _key(start, zone) > limit:
            return result()
        seen += 1
        if count is not None and seen > count:
            return result()
        if _key(start, zone) >= finish:
            return result()
        collect(start)
    return result(True)


def _listable(events: Sequence[Stored]) -> list[Stored]:
    """Expand simple series around individual overrides; retain unknown coverage."""
    overrides: dict[str, list[Stored]] = {}
    for event in events:
        if event.is_override and event.uid:
            overrides.setdefault(event.uid, []).append(event)
    kept = []
    for event in events:
        if event.cancelled:
            continue
        changed = overrides.get(event.uid or "", []) if event.recurring else []
        if changed:
            if any(
                other.recurrence_id is None
                or other.range_override
                or isinstance(other.recurrence_id, datetime)
                != isinstance(event.start, datetime)
                for other in changed
            ):
                event = replace(event, expandable=False)
            else:
                event = replace(
                    event,
                    exdates=event.exdates
                    + tuple(
                        e.recurrence_id for e in changed if e.recurrence_id is not None
                    ),
                )
        kept.append(event)
    return kept


def _day(value: date) -> str:
    return f"{WEEKDAYS[value.weekday()][:3]} {value:%Y-%m-%d}"


def _span(start: date | datetime, end: date | datetime, zone: Zone) -> str:
    if not isinstance(start, datetime):
        assert not isinstance(end, datetime)
        last = end - timedelta(days=1)
        if last <= start:
            return f"{_day(start)} (all day)"
        return f"{_day(start)} – {_day(last)} (all day)"
    assert isinstance(end, datetime)
    begin, finish = start.astimezone(zone.tzinfo), end.astimezone(zone.tzinfo)
    offsets = (
        _ambiguous(begin)
        or _ambiguous(finish)
        or begin.utcoffset() != finish.utcoffset()
    )
    begin_time, end_time = f"{begin:%H:%M}", f"{finish:%H:%M}"
    if offsets:
        begin_time += f" ({format_offset(begin.utcoffset())})"
        end_time += f" ({format_offset(finish.utcoffset())})"
    text = f"{_day(begin.date())} {begin_time}"
    if _key(finish, zone) == _key(begin, zone):
        return text
    if finish.date() == begin.date():
        return f"{text}–{end_time}"
    return f"{text} – {_day(finish.date())} {end_time}"


def _repeat_text(repeat: RepeatArgs) -> str:
    unit = {"daily": "day", "weekly": "week", "monthly": "month", "yearly": "year"}[
        repeat.frequency
    ]
    text = (
        f"repeats {repeat.frequency}"
        if repeat.interval == 1
        else f"repeats every {repeat.interval} {unit}s"
    )
    if repeat.count is not None:
        text += f", {repeat.count} times"
    if repeat.until is not None:
        text += f", until {repeat.until:%Y-%m-%d}"
    return text


def _summary(event: Event, uid: str, zone: Zone) -> str:
    line = f"- {_span(event.start, event.end, zone)} {event.title}"
    if event.location:
        line += f" @ {event.location}"
    extras = []
    if event.reminders:
        extras.append(
            "reminder " + ", ".join(f"{m} min before" for m in event.reminders)
        )
    if event.repeat is not None:
        extras.append(_repeat_text(event.repeat))
    return line + (f" ({'; '.join(extras)})" if extras else "") + f"  [id {uid}]"


def _entry(
    event: Stored, start: date | datetime, end: date | datetime, zone: Zone
) -> str:
    line = f"- {_span(start, end, zone)} {event.title}"
    if event.location:
        line += f" @ {event.location}"
    if event.recurring:
        if event.expandable:
            line += " (repeats)"
        else:
            rule = f", rule {event.rule_text}" if event.rule_text else ""
            line += f" (repeating series{rule}; occurrences not expanded)"
    if event.recurrence_id is not None:
        original = event.recurrence_id.isoformat()
        line += f" (original occurrence {original})"
    return line + (f"  [id {event.uid}]" if event.uid else "")


def _unexpanded_notes(events: Sequence[Stored], zone: Zone, limit: int) -> list[str]:
    if not events:
        return []
    lines = [
        f"Coverage incomplete: {len(events)} recurring series could not be expanded; "
        "do not assume unlisted times are free.",
        "Unexpanded series (original start shown; occurrences in the requested "
        "period are unknown):",
        *(_entry(e, e.start, e.end, zone) for e in events[:limit]),
    ]
    if len(events) > limit:
        lines.append(f"... {len(events) - limit} more unexpanded series.")
    return lines
