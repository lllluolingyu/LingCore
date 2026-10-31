"""iCalendar parsing, serialization, and timezone components."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lingcore.bundled_plugins.calendar.models import (
    _UTC,
    Event,
    _advance,
    _ambiguous,
    _key,
    _length,
)
from lingcore.errors import ToolError
from lingcore.timezones import Zone

PRODID = "-//LingCore//calendar plugin 1.0//EN"
_TZ_SPAN = timedelta(days=3653)  # VTIMEZONE coverage per series: ten years
_TZ_TOTAL = timedelta(days=36525)  # and for the whole file: a century
_FREQ_SPAN = {"DAILY": 1, "WEEKLY": 7, "MONTHLY": 31, "YEARLY": 366}
_HEADER_PROPERTIES = frozenset({"VERSION", "PRODID", "CALSCALE", "METHOD"})


def _escape(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> list[str]:
    """Split a content line into 75-octet lines without breaking a character."""
    out: list[str] = []
    current, size, limit = "", 0, 75
    for char in line:
        width = len(char.encode("utf-8"))
        if size + width > limit:
            out.append(current)
            current, size, limit = " ", 1, 75
        current += char
        size += width
    out.append(current)
    return out


def _ical_offset(offset: timedelta | None) -> str:
    seconds = int((offset or timedelta()).total_seconds())
    sign = "-" if seconds < 0 else "+"
    hours, rest = divmod(abs(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{sign}{hours:02d}{minutes:02d}" + (f"{secs:02d}" if secs else "")


def _stamp(value: datetime) -> str:
    return f"{value.astimezone(_UTC):%Y%m%dT%H%M%SZ}"


def _when(name: str, value: date | datetime, zone: Zone) -> str:
    if not isinstance(value, datetime):
        return f"{name};VALUE=DATE:{value:%Y%m%d}"
    local = value.astimezone(zone.tzinfo)
    # RFC 5545 interprets an ambiguous TZID time as its first occurrence.
    # UTC is needed to preserve an explicitly chosen second occurrence.
    if zone.iana and not (local.fold and _ambiguous(local)):
        return f"{name};TZID={zone.name}:{local:%Y%m%dT%H%M%S}"
    return f"{name}:{_stamp(value)}"


def _rrule(event: Event, zone: Zone) -> str | None:
    repeat = event.repeat
    if repeat is None:
        return None
    parts = [f"FREQ={repeat.frequency.upper()}"]
    if repeat.interval != 1:
        parts.append(f"INTERVAL={repeat.interval}")
    if repeat.count is not None:
        parts.append(f"COUNT={repeat.count}")
    if repeat.until is not None:
        if isinstance(event.start, datetime):
            # With a TZID or UTC start, UNTIL must be a UTC date-time. The last
            # occurrence on that day starts at the series' local start time.
            local = datetime.combine(repeat.until, event.start.time())
            parts.append(f"UNTIL={_stamp(local.replace(tzinfo=zone.tzinfo))}")
        else:
            parts.append(f"UNTIL={repeat.until:%Y%m%d}")
    return "RRULE:" + ";".join(parts)


def _vevent(event: Event, uid: str, zone: Zone, now: datetime) -> list[str]:
    start, end = event.start, event.end
    exceptions = []
    if (
        event.repeat is not None
        and isinstance(start, datetime)
        and start.fold
        and _ambiguous(start)
        and zone.iana
    ):
        # Keep the rule in local time across future DST changes, replacing
        # only its first occurrence with the user's explicit UTC instant.
        first = start.replace(fold=0)
        exceptions = [f"EXDATE:{_stamp(first)}", f"RDATE:{_stamp(start)}"]
        end = _advance(first, _length(start, end))
        start = first
    write_zone = zone
    if isinstance(start, datetime) and start.fold and _ambiguous(start):
        # A UTC DTSTART needs no VTIMEZONE; keep its DTEND in UTC too.
        write_zone = Zone(_UTC, "UTC", False)
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_stamp(now)}",
        f"SUMMARY:{_escape(event.title)}",
        _when("DTSTART", start, write_zone),
        _when("DTEND", end, write_zone),
    ]
    rule = _rrule(event, zone)
    if rule is not None:
        lines.append(rule)
    lines += exceptions
    if event.location:
        lines.append(f"LOCATION:{_escape(event.location)}")
    if event.description:
        lines.append(f"DESCRIPTION:{_escape(event.description)}")
    lines += _alarms(event.title, event.reminders)
    lines.append("END:VEVENT")
    return lines


def _alarms(title: str, reminders: Sequence[int]) -> list[str]:
    lines = []
    for minutes in reminders:
        lines += [
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{_escape(title)}",
            f"TRIGGER:{'-PT%dM' % minutes if minutes else 'PT0S'}",
            "END:VALARM",
        ]
    return lines


def _transitions(zone: tzinfo, start: int, end: int) -> Iterator[int]:
    """POSIX seconds at which ``zone``'s UTC offset changes in [start, end)."""

    def offset(ts: int) -> timedelta | None:
        return datetime.fromtimestamp(ts, zone).utcoffset()

    day = 86_400
    t, before = start, offset(start)
    while t < end:
        nxt = min(t + day, end)
        after = offset(nxt)
        if after != before:
            lo, hi = t, nxt  # offset(lo) == before, offset(hi) == after
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if offset(mid) == before:
                    lo = mid
                else:
                    hi = mid
            yield hi
            before = after
        t = nxt


def _observance(moment: datetime, before: timedelta | None) -> list[str]:
    kind = "DAYLIGHT" if moment.dst() else "STANDARD"
    local = (moment.astimezone(_UTC) + (before or timedelta())).replace(tzinfo=None)
    return [
        f"BEGIN:{kind}",
        f"DTSTART:{local:%Y%m%dT%H%M%S}",
        f"TZOFFSETFROM:{_ical_offset(before)}",
        f"TZOFFSETTO:{_ical_offset(moment.utcoffset())}",
        f"TZNAME:{_escape(moment.tzname() or _ical_offset(moment.utcoffset()))}",
        f"END:{kind}",
    ]


def _reach(event: Stored, zone: Zone) -> datetime:
    """How far a timed series' occurrences reach, for VTIMEZONE coverage."""
    assert isinstance(event.start, datetime) and isinstance(event.end, datetime)
    start, end = event.start.astimezone(_UTC), event.end.astimezone(_UTC)
    cap = start + _TZ_SPAN
    rule = event.rule
    if rule is None:
        return min(end, cap)
    try:
        if "UNTIL" in rule:
            until = _key(_ical_time(rule["UNTIL"], {}, zone), zone)
            return min(until + (end - start) + timedelta(days=2), cap)
        if "COUNT" in rule and rule.get("FREQ") in _FREQ_SPAN:
            steps = int(rule["COUNT"]) * max(1, int(rule.get("INTERVAL", "1")))
            return min(end + timedelta(days=steps * _FREQ_SPAN[rule["FREQ"]]), cap)
    except (ValueError, OverflowError):
        pass
    return cap


def vtimezone(zone: Zone, events: Sequence[Stored]) -> list[str]:
    """An explicit VTIMEZONE covering every timed event saved in ``zone``."""
    timed = [e for e in events if isinstance(e.start, datetime) and e.tzid == zone.name]
    if not zone.iana or not timed:
        return []
    first = min(_key(e.start, zone) for e in timed).astimezone(_UTC) - timedelta(days=1)
    last = min(max(_reach(e, zone) for e in timed), first + _TZ_TOTAL)
    opening = first.astimezone(zone.tzinfo)
    lines = ["BEGIN:VTIMEZONE", f"TZID:{zone.name}"]
    lines += _observance(opening, opening.utcoffset())
    before = opening.utcoffset()
    for ts in _transitions(zone.tzinfo, int(first.timestamp()), int(last.timestamp())):
        moment = datetime.fromtimestamp(ts, zone.tzinfo)
        lines += _observance(moment, before)
        before = moment.utcoffset()
    lines.append("END:VTIMEZONE")
    return lines


_TIME = re.compile(r"(\d{8})(?:T(\d{6})(Z?))?\Z")
_DURATION = re.compile(
    r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?\Z"
)
_SIMPLE_RULE_KEYS = frozenset({"FREQ", "INTERVAL", "COUNT", "UNTIL", "WKST"})


def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        elif raw:
            lines.append(raw)
    return lines


def _split(line: str) -> tuple[str, dict[str, str], str]:
    quoted = False
    for index, char in enumerate(line):
        if char == '"':
            quoted = not quoted
        elif char == ":" and not quoted:
            head, value = line[:index], line[index + 1 :]
            break
    else:
        return line.upper(), {}, ""
    name, *raw_params = head.split(";")
    params = {}
    for item in raw_params:
        key, _, val = item.partition("=")
        params[key.upper()] = val.strip('"')
    return name.upper(), params, value


def _unescape(text: str) -> str:
    return re.sub(
        r"\\([\\;,nN])",
        lambda m: "\n" if m.group(1) in "nN" else m.group(1),
        text,
    )


def _tz(name: str) -> tzinfo | None:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _ical_time(value: str, params: Mapping[str, str], zone: Zone) -> date | datetime:
    match = _TIME.fullmatch(value.strip())
    if match is None:
        raise ValueError("bad time")
    day = datetime.strptime(match.group(1), "%Y%m%d")
    if match.group(2) is None or params.get("VALUE") == "DATE":
        return day.date()
    clock = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    if match.group(3):
        return clock.replace(tzinfo=_UTC)
    tzid = params.get("TZID")
    # An unknown TZID (a Windows zone name, a custom definition) and a
    # floating time are read in the calendar's own zone.
    return clock.replace(tzinfo=(_tz(tzid) if tzid else None) or zone.tzinfo)


def _duration_end(start: date | datetime, value: str) -> date | datetime:
    match = _DURATION.fullmatch(value.strip())
    if match is None:
        raise ValueError("bad duration")
    sign, weeks, days, hours, minutes, seconds = match.groups()
    calendar_days = timedelta(weeks=int(weeks or 0), days=int(days or 0))
    elapsed = timedelta(
        hours=int(hours or 0),
        minutes=int(minutes or 0),
        seconds=int(seconds or 0),
    )
    if sign == "-":
        calendar_days, elapsed = -calendar_days, -elapsed
    # DURATION days/weeks are nominal calendar units, but hours/minutes are
    # elapsed time. Keep that distinction for every imported occurrence.
    local = start + calendar_days if calendar_days else start
    return _advance(local, elapsed)


@dataclass(slots=True)
class Stored:
    title: str
    start: date | datetime
    end: date | datetime
    location: str | None
    description: str | None
    rule: dict[str, str] | None
    rule_text: str | None
    expandable: bool
    uid: str | None
    tzid: str | None
    is_override: bool
    cancelled: bool
    has_extra_dates: bool
    rdates: tuple[date | datetime, ...]
    exdates: tuple[date | datetime, ...]
    duration: str | None
    recurrence_id: date | datetime | None
    range_override: bool

    @property
    def recurring(self) -> bool:
        return self.rule is not None or self.has_extra_dates


def _property_items(
    component: Sequence[str],
) -> Iterator[tuple[str, dict[str, str], str]]:
    """The component's own properties, including repeated RDATE/EXDATE lines."""
    depth = 0
    for line in component[1:-1]:
        name, params, value = _split(line)
        if name == "BEGIN":
            depth += 1  # VALARM and friends: not the event's own properties
        elif name == "END":
            depth -= 1
        elif depth == 0:
            yield name, params, value


def _properties(component: Sequence[str]) -> dict[str, tuple[dict[str, str], str]]:
    """The component's own first-seen properties, skipping nested components."""
    props: dict[str, tuple[dict[str, str], str]] = {}
    for name, params, value in _property_items(component):
        props.setdefault(name, (params, value))
    return props


def _stored(component: Sequence[str], zone: Zone) -> Stored | None:
    props = _properties(component)
    if "DTSTART" not in props:
        return None
    duration = None
    try:
        start = _ical_time(props["DTSTART"][1], props["DTSTART"][0], zone)
        end: date | datetime
        if "DTEND" in props:
            end = _ical_time(props["DTEND"][1], props["DTEND"][0], zone)
        elif "DURATION" in props:
            duration = props["DURATION"][1]
            end = _duration_end(start, duration)
        else:
            end = start if isinstance(start, datetime) else start + timedelta(days=1)
    except (ValueError, OverflowError):
        return None
    if isinstance(end, datetime) != isinstance(start, datetime) or _key(
        end, zone
    ) < _key(start, zone):
        end = start if isinstance(start, datetime) else start + timedelta(days=1)
        duration = None
    rule = None
    rule_text = None
    expandable = True
    if "RRULE" in props:
        rule_text = props["RRULE"][1]
        rule = {}
        for part in rule_text.split(";"):
            key, _, val = part.partition("=")
            rule[key.upper()] = val
        expandable = set(rule) <= _SIMPLE_RULE_KEYS and rule.get("FREQ") in _FREQ_SPAN
        try:
            _rule_bounds(rule, zone)
        except (ValueError, OverflowError):
            expandable = False
    dates: dict[str, list[date | datetime]] = {"RDATE": [], "EXDATE": []}
    for name, params, value in _property_items(component):
        if name not in dates:
            continue
        try:
            if params.get("VALUE", "DATE-TIME") not in {"DATE", "DATE-TIME"}:
                raise ValueError("unsupported recurrence date type")
            for text in value.split(","):
                when = _ical_time(text, params, zone)
                if isinstance(when, datetime) != isinstance(start, datetime):
                    raise ValueError("recurrence date type differs from DTSTART")
                dates[name].append(when)
        except (ValueError, OverflowError):
            expandable = False
    recurrence_id = None
    if "RECURRENCE-ID" in props:
        try:
            recurrence_id = _ical_time(
                props["RECURRENCE-ID"][1], props["RECURRENCE-ID"][0], zone
            )
        except (ValueError, OverflowError):
            pass
    return Stored(
        title=_unescape(props.get("SUMMARY", ({}, ""))[1]).strip() or "(untitled)",
        start=start,
        end=end,
        location=_unescape(props.get("LOCATION", ({}, ""))[1]).strip() or None,
        description=_unescape(props.get("DESCRIPTION", ({}, ""))[1]).strip() or None,
        rule=rule,
        rule_text=rule_text,
        expandable=expandable,
        uid=props.get("UID", ({}, ""))[1].strip() or None,
        tzid=props["DTSTART"][0].get("TZID"),
        is_override="RECURRENCE-ID" in props,
        cancelled=props.get("STATUS", ({}, ""))[1].strip().upper() == "CANCELLED",
        has_extra_dates="RDATE" in props or "EXDATE" in props,
        rdates=tuple(dates["RDATE"]),
        exdates=tuple(dates["EXDATE"]),
        duration=duration,
        recurrence_id=recurrence_id,
        range_override="RANGE" in props.get("RECURRENCE-ID", ({}, ""))[0],
    )


def _kind(component: Sequence[str]) -> str:
    return _split(component[0])[2].strip().upper()


@dataclass(slots=True)
class Calendar:
    """One VCALENDAR as unfolded lines: extra calendar properties + components."""

    properties: list[str] = field(default_factory=list)
    components: list[list[str]] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> Calendar:
        """Parse a whole file; ``ValueError`` unless it is exactly one VCALENDAR."""
        lines = _unfold(text.lstrip("﻿"))
        if not lines:
            return cls()
        begin, end = _split(lines[0]), _split(lines[-1])
        if (begin[0], begin[2].strip().upper()) != ("BEGIN", "VCALENDAR") or (
            end[0],
            end[2].strip().upper(),
        ) != ("END", "VCALENDAR"):
            raise ValueError("not a single VCALENDAR")
        calendar = cls()
        current: list[str] | None = None
        depth = 0
        for line in lines[1:-1]:
            name = _split(line)[0]
            if name == "BEGIN":
                if depth == 0:
                    current = []
                depth += 1
            if current is None:
                if name == "END":
                    raise ValueError("unbalanced END")
                if name not in _HEADER_PROPERTIES:
                    calendar.properties.append(line)
                continue
            current.append(line)
            if name == "END":
                depth -= 1
                if depth == 0:
                    calendar.components.append(current)
                    current = None
        if depth:
            raise ValueError("unterminated component")
        return calendar

    def events(self, zone: Zone) -> list[tuple[list[str], Stored | None]]:
        return [(c, _stored(c, zone)) for c in self.components if _kind(c) == "VEVENT"]

    def uids(self) -> set[str]:
        return {
            _properties(c).get("UID", ({}, ""))[1].strip()
            for c in self.components
            if _kind(c) == "VEVENT"
        }

    def render(self, zone: Zone) -> bytes:
        """The file, with ``zone``'s VTIMEZONE regenerated to cover its events."""
        kept = [
            c
            for c in self.components
            if not (
                _kind(c) == "VTIMEZONE"
                and _properties(c).get("TZID", ({}, ""))[1].strip() == zone.name
            )
        ]
        stored = [e for _, e in self.events(zone) if e is not None]
        zones = [c for c in kept if _kind(c) == "VTIMEZONE"]
        others = [c for c in kept if _kind(c) != "VTIMEZONE"]
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            f"PRODID:{PRODID}",
            "CALSCALE:GREGORIAN",
            *self.properties,
            *vtimezone(zone, stored),
            *(line for c in zones for line in c),
            *(line for c in others for line in c),
            "END:VCALENDAR",
        ]
        folded = [part for line in lines for part in _fold(line)]
        return ("\r\n".join(folded) + "\r\n").encode("utf-8")


def parse_calendar(text: str, zone: Zone) -> list[Stored]:
    """The calendar's events, cancelled ones included (``Stored.cancelled``)."""
    try:
        calendar = Calendar.parse(text)
    except ValueError:
        raise ToolError(
            "the calendar file is not a valid iCalendar file; fix or move it"
        ) from None
    return [e for _, e in calendar.events(zone) if e is not None]


def _rule_bounds(
    rule: Mapping[str, str], zone: Zone
) -> tuple[int, int | None, datetime | None]:
    interval = int(rule.get("INTERVAL", "1"))
    count = int(rule["COUNT"]) if "COUNT" in rule else None
    if interval < 1 or (count is not None and count < 1):
        raise ValueError("recurrence interval and count must be positive")
    until = _ical_time(rule["UNTIL"], {}, zone) if "UNTIL" in rule else None
    # A date UNTIL includes that whole day; a date-time UNTIL is an instant.
    limit = (
        None
        if until is None
        else _key(until, zone)
        if isinstance(until, datetime)
        else _key(until + timedelta(days=1), zone) - timedelta(microseconds=1)
    )
    return interval, count, limit
