"""Edits to existing iCalendar events and individual occurrences."""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta

from lingcore.bundled_plugins.calendar.icalendar import (
    Stored,
    _alarms,
    _duration_end,
    _escape,
    _key,
    _properties,
    _rrule,
    _split,
    _stamp,
    _tz,
    _vevent,
    _when,
)
from lingcore.bundled_plugins.calendar.models import (
    _UTC,
    Event,
    EventChanges,
    RepeatArgs,
    _advance,
    _after,
    _length,
    _parse_when,
)
from lingcore.bundled_plugins.calendar.recurrence import occurrences
from lingcore.errors import ToolError
from lingcore.timezones import Zone


def _new_uid(taken: set[str]) -> str:
    while True:
        uid = f"{secrets.token_hex(6)}@lingcore"
        if uid not in taken:
            taken.add(uid)
            return uid


def _rewrite_component(
    component: Sequence[str],
    replacements: Mapping[str, Sequence[str]],
    *,
    alarms: Sequence[str] | None = None,
) -> list[str]:
    """Patch own properties while retaining unknown properties and components."""
    pending = dict(replacements)
    lines = [component[0]]
    depth = skip = 0
    for line in component[1:-1]:
        name, _, value = _split(line)
        if skip:
            skip += (name == "BEGIN") - (name == "END")
            continue
        if depth == 0:
            if name == "BEGIN" and value.upper() == "VALARM" and alarms is not None:
                skip = 1
                continue
            if name in replacements:
                lines.extend(pending.pop(name, ()))
                continue
        lines.append(line)
        depth += (name == "BEGIN") - (name == "END")
    lines += [line for values in pending.values() for line in values]
    if alarms is not None:
        lines.extend(alarms)
    return [*lines, component[-1]]


def _saved_zone(event: Stored, fallback: Zone) -> Zone:
    if event.tzid and (tz := _tz(event.tzid)) is not None:
        return Zone(tz, event.tzid, True)
    if isinstance(event.start, datetime) and event.start.tzinfo is _UTC:
        return Zone(_UTC, "UTC", False)
    return fallback


def _patched_event(event: Stored, changes: EventChanges, zone: Zone) -> Event:
    fields = changes.model_fields_set
    start = (
        _parse_when(changes.start, zone, "start")
        if changes.start is not None
        else event.start
    )
    if changes.end is not None:
        end = _parse_when(changes.end, zone, "end")
        if isinstance(start, datetime) != isinstance(end, datetime):
            raise ToolError("start and end must both be dates or both be date-times")
        if not isinstance(start, datetime):
            end += timedelta(days=1)
    elif changes.duration_minutes is not None:
        if not isinstance(start, datetime):
            raise ToolError("duration_minutes needs a timed start")
        end = _after(start, changes.duration_minutes, zone)
    else:
        if isinstance(start, datetime) != isinstance(event.start, datetime):
            raise ToolError(
                "changing between all-day and timed needs a new end or duration"
            )
        end = (
            _duration_end(start, event.duration)
            if event.duration is not None
            else _advance(start, _length(event.start, event.end))
        )
    if _key(end, zone) < _key(start, zone) or (
        (changes.end is not None or changes.duration_minutes is not None)
        and _key(end, zone) == _key(start, zone)
    ):
        raise ToolError("the updated event ends before it starts")
    repeat = changes.repeat
    if repeat is not None and repeat.until is not None:
        first = start.date() if isinstance(start, datetime) else start
        if repeat.until < first:
            raise ToolError("the updated event repeats until before it starts")
    return Event(
        title=changes.title if changes.title is not None else event.title,
        start=start,
        end=end,
        location=changes.location if "location" in fields else event.location,
        description=changes.description
        if "description" in fields
        else event.description,
        reminders=tuple(changes.reminders or ()),
        repeat=repeat,
    )


def _patch_component(
    component: Sequence[str], event: Stored, changes: EventChanges, zone: Zone
) -> list[str]:
    fields = changes.model_fields_set
    updated = _patched_event(event, changes, zone)
    props = _properties(component)
    stamp = _stamp(datetime.now(_UTC))
    try:
        sequence = max(0, int(props.get("SEQUENCE", ({}, "0"))[1])) + 1
    except ValueError:
        sequence = 1
    patch: dict[str, Sequence[str]] = {
        "DTSTAMP": [f"DTSTAMP:{stamp}"],
        "LAST-MODIFIED": [f"LAST-MODIFIED:{stamp}"],
        "SEQUENCE": [f"SEQUENCE:{sequence}"],
    }
    for field_name, property_name in (
        ("title", "SUMMARY"),
        ("location", "LOCATION"),
        ("description", "DESCRIPTION"),
    ):
        if field_name in fields:
            text = getattr(updated, field_name)
            patch[property_name] = [f"{property_name}:{_escape(text)}"] if text else []
    adding_repeat = (
        "repeat" in fields and changes.repeat is not None and not event.recurring
    )
    write_zone = (
        zone if "start" in fields or adding_repeat else _saved_zone(event, zone)
    )
    if (
        adding_repeat
        and isinstance(updated.start, datetime)
        and isinstance(updated.end, datetime)
    ):
        updated = replace(
            updated,
            start=updated.start.astimezone(write_zone.tzinfo),
            end=updated.end.astimezone(write_zone.tzinfo),
        )
    if fields & {"start", "end", "duration_minutes", "repeat"}:
        # The placeholder only selects TZID-based recurrence serialization.
        # Keep the saved rule verbatim unless repeat was explicitly changed.
        recurring = updated.repeat or (
            RepeatArgs(frequency="daily")
            if event.recurring and "repeat" not in fields
            else None
        )
        generated = _vevent(
            replace(updated, repeat=recurring),
            event.uid or "",
            write_zone,
            datetime.now(_UTC),
        )
        if fields & {"start", "end", "duration_minutes"} or adding_repeat:
            for name in ("DTSTART", "DTEND"):
                patch[name] = [line for line in generated if _split(line)[0] == name]
            if event.duration is not None and not fields & {"end", "duration_minutes"}:
                patch["DTEND"] = []  # retain the nominal DURATION when moving start
            else:
                patch["DURATION"] = []
            if "start" in fields or adding_repeat:
                for name in ("RDATE", "EXDATE"):
                    patch[name] = [
                        line for line in generated if _split(line)[0] == name
                    ]
        if "repeat" in fields:
            patch["RRULE"] = (
                [rule] if (rule := _rrule(updated, write_zone)) is not None else []
            )
            patch["EXRULE"] = []
    return _rewrite_component(
        component,
        patch,
        alarms=_alarms(updated.title, updated.reminders)
        if "reminders" in fields
        else None,
    )


def _occurrence_component(
    master_component: Sequence[str], master: Stored, original: str, zone: Zone
) -> tuple[list[str], date | datetime]:
    start = _parse_when(original, zone, "occurrence")
    if isinstance(start, datetime) != isinstance(master.start, datetime):
        raise ToolError(
            "occurrence must have the same date/date-time form as the series"
        )
    if not master.expandable:
        raise ToolError(
            "this recurrence rule cannot be expanded safely; edit it in a calendar app"
        )
    key = _key(start, zone)
    spans, cut = occurrences(master, zone, (key, key + timedelta(microseconds=1)))
    matches = [(begin, end) for begin, end in spans if _key(begin, zone) == key]
    if len(matches) != 1 or cut:
        raise ToolError(
            "no occurrence at that original start; calendar_list shows the dates"
        )
    begin, end = matches[0]
    write_zone = _saved_zone(master, zone)
    patch: dict[str, Sequence[str]] = {
        "DTSTART": [_when("DTSTART", begin, write_zone)],
        "RECURRENCE-ID": [_when("RECURRENCE-ID", begin, write_zone)],
        "RRULE": [],
        "RDATE": [],
        "EXDATE": [],
        "EXRULE": [],
    }
    if master.duration is None:
        patch["DTEND"] = [_when("DTEND", end, write_zone)]
    return _rewrite_component(master_component, patch), begin
