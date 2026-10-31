"""Calendar options, tool arguments, and local event time validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from lingcore.errors import ToolError
from lingcore.timezones import Zone, resolve_zone

OPTION_PATH = "tool_options.calendar"
MAX_EVENTS_PER_CALL = 20
MAX_REMINDER_MINUTES = 28 * 24 * 60
_LIST_SPAN_DAYS = 3660
_UTC = timezone.utc


class CalendarOptions(BaseModel):
    """Validated ``tool_options.calendar``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Relative to the profile directory, like memory.md.
    path: str = "calendar.ics"
    allow_absolute_path: bool = False
    # IANA name; unset falls back to tool_options.clock.timezone, then the host.
    timezone: str | None = None
    default_duration_minutes: int = Field(default=60, ge=1, le=24 * 60)
    list_days: int = Field(default=90, ge=1, le=_LIST_SPAN_DAYS)


def _errors(exc: ValidationError) -> str:
    return "; ".join(
        ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
        for error in exc.errors()
    )


def parse_calendar_options(
    tool_options: Mapping[str, object],
) -> tuple[CalendarOptions, Zone]:
    raw = tool_options.get("calendar", {})
    if not isinstance(raw, Mapping):
        raise ToolError(f"{OPTION_PATH} must be a mapping")
    try:
        options = CalendarOptions.model_validate(dict(raw))
    except ValidationError as exc:
        raise ToolError(f"invalid {OPTION_PATH}: {_errors(exc)}") from None
    name = options.timezone
    if not (name or "").strip():
        clock = tool_options.get("clock", {})
        candidate = clock.get("timezone") if isinstance(clock, Mapping) else None
        name = candidate if isinstance(candidate, str) else None
    try:
        return options, resolve_zone(name)
    except ValueError as exc:
        raise ToolError(f"invalid {OPTION_PATH}: {exc}") from None


class RepeatArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frequency: Literal["daily", "weekly", "monthly", "yearly"]
    interval: int = Field(
        default=1, ge=1, le=366, description="Every N days/weeks/months/years."
    )
    count: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Total number of occurrences, including the first.",
    )
    until: date | None = Field(
        default=None, description="Last date (YYYY-MM-DD) an occurrence may fall on."
    )

    @model_validator(mode="after")
    def _one_end(self) -> RepeatArgs:
        if self.count is not None and self.until is not None:
            raise ValueError("give count or until, not both")
        return self


class EventArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    start: str = Field(
        description=(
            "'YYYY-MM-DD' for an all-day event, or the local start time "
            "'YYYY-MM-DDTHH:MM' in the calendar's timezone (an explicit offset "
            "such as '2026-10-21T15:00+02:00' is also accepted)."
        )
    )
    end: str | None = Field(
        default=None,
        description=(
            "Same form as start. For an all-day event, the last day (inclusive); "
            "for a timed event, when it ends. Omit to use duration_minutes or "
            "the default length."
        ),
    )
    duration_minutes: int | None = Field(default=None, ge=1, le=14 * 24 * 60)
    location: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=4000)
    reminders: list[int] = Field(
        default_factory=list,
        max_length=3,
        description="Alerts, in minutes before the start (0 = at the start).",
    )
    repeat: RepeatArgs | None = None

    @field_validator("title")
    @classmethod
    def _title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be blank")
        return value.strip()

    @field_validator("reminders")
    @classmethod
    def _reminders(cls, values: list[int]) -> list[int]:
        for minutes in values:
            if not 0 <= minutes <= MAX_REMINDER_MINUTES:
                raise ValueError(
                    f"reminders must be 0-{MAX_REMINDER_MINUTES} minutes before"
                )
        return sorted(set(values), reverse=True)

    @model_validator(mode="after")
    def _one_length(self) -> EventArgs:
        if self.end is not None and self.duration_minutes is not None:
            raise ValueError("give end or duration_minutes, not both")
        return self


class AddArgs(BaseModel):
    events: list[EventArgs] = Field(min_length=1, max_length=MAX_EVENTS_PER_CALL)


class EventChanges(BaseModel):
    """An explicit patch: omitted fields retain their saved values."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    start: str | None = None
    end: str | None = None
    duration_minutes: int | None = Field(default=None, ge=1, le=14 * 24 * 60)
    location: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=4000)
    reminders: list[int] | None = Field(default=None, max_length=3)
    repeat: RepeatArgs | None = None

    @field_validator("title")
    @classmethod
    def _title(cls, value: str | None) -> str | None:
        return EventArgs._title(value) if value is not None else None

    @field_validator("reminders")
    @classmethod
    def _reminders(cls, value: list[int] | None) -> list[int] | None:
        return EventArgs._reminders(value) if value is not None else None

    @model_validator(mode="after")
    def _patch(self) -> EventChanges:
        if not self.model_fields_set:
            raise ValueError("give at least one field to change")
        for name in ("title", "start", "end", "duration_minutes", "reminders"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null; omit it to keep its value")
        if self.end is not None and self.duration_minutes is not None:
            raise ValueError("give end or duration_minutes, not both")
        return self


class UpdateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Event id shown by calendar_list.")
    changes: EventChanges = Field(
        description=(
            "Fields to change; omitted fields stay intact. Set location, "
            "description or repeat to null to clear them; reminders: [] clears alarms. "
            "Moving start alone preserves the event's length."
        )
    )
    scope: Literal["series", "occurrence"] | None = Field(
        default=None,
        description="Required for a repeating event: whole series or one occurrence.",
    )
    occurrence: str | None = Field(
        default=None,
        description=(
            "For scope=occurrence, its ORIGINAL start (YYYY-MM-DD for all day, "
            "otherwise YYYY-MM-DDTHH:MM, with an offset for an ambiguous time). "
            "Keep this original start when editing a previously moved occurrence."
        ),
    )

    @model_validator(mode="after")
    def _scope(self) -> UpdateArgs:
        self.id = self.id.strip()
        if not self.id:
            raise ValueError("id must not be blank")
        if (self.scope == "occurrence") != (self.occurrence is not None):
            raise ValueError("give occurrence exactly when scope is 'occurrence'")
        if self.scope == "occurrence" and "repeat" in self.changes.model_fields_set:
            raise ValueError("repeat can only be changed for the whole series")
        return self


class ListArgs(BaseModel):
    start: date | None = Field(
        default=None, description="First day to list (YYYY-MM-DD). Default: today."
    )
    end: date | None = Field(
        default=None,
        description=(
            "Last day to list, inclusive. Default: the profile's listing window "
            "(about three months), or ten years when query is given."
        ),
    )
    query: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Only events whose title, place or notes contain this text "
            "(case-insensitive), e.g. 'dentist'."
        ),
    )


class RemoveArgs(BaseModel):
    ids: list[str] = Field(
        min_length=1,
        max_length=MAX_EVENTS_PER_CALL,
        description=(
            "Event ids as shown by calendar_list or calendar_add. Removing a "
            "repeating event removes the whole series."
        ),
    )


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


@dataclass(frozen=True, slots=True)
class Event:
    title: str
    start: date | datetime  # an aware datetime for a timed event
    end: date | datetime  # exclusive
    location: str | None
    description: str | None
    reminders: tuple[int, ...]
    repeat: RepeatArgs | None

    @property
    def all_day(self) -> bool:
        return not isinstance(self.start, datetime)


def _parse_when(text: str, zone: Zone, label: str) -> date | datetime:
    value = text.strip()
    if _DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ToolError(f"{label} is not a valid date: {value!r}") from None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ToolError(
            f"{label} must be 'YYYY-MM-DD' (all day) or 'YYYY-MM-DDTHH:MM', "
            f"got {value!r}"
        ) from None
    parsed = parsed.replace(microsecond=0)
    if parsed.tzinfo is not None:
        return parsed.astimezone(zone.tzinfo)
    local = parsed.replace(tzinfo=zone.tzinfo)
    if local.astimezone(_UTC).astimezone(zone.tzinfo).replace(tzinfo=None) != parsed:
        raise ToolError(
            f"{label} {value!r} does not exist in {zone.name}: a daylight-saving "
            "change skips that time"
        )
    return local


def _after(start: datetime, minutes: int, zone: Zone) -> datetime:
    """``minutes`` of elapsed time after ``start``, across any offset change."""
    return (start.astimezone(_UTC) + timedelta(minutes=minutes)).astimezone(zone.tzinfo)


def _length(start: date | datetime, end: date | datetime) -> timedelta:
    if isinstance(start, datetime):
        assert isinstance(end, datetime)
        return end.astimezone(_UTC) - start.astimezone(_UTC)
    assert not isinstance(end, datetime)
    return end - start


def _advance(start: date | datetime, length: timedelta) -> date | datetime:
    if isinstance(start, datetime):
        return (start.astimezone(_UTC) + length).astimezone(start.tzinfo)
    return start + length


def _ambiguous(value: datetime) -> bool:
    return value.replace(fold=0).utcoffset() != value.replace(fold=1).utcoffset()


def resolve_event(args: EventArgs, zone: Zone, default_minutes: int) -> Event:
    start = _parse_when(args.start, zone, "start")
    timed = isinstance(start, datetime)
    end: date | datetime
    if args.end is not None:
        end = _parse_when(args.end, zone, "end")
        if isinstance(end, datetime) != timed:
            raise ToolError("start and end must both be dates or both be date-times")
        if not timed:
            end = end + timedelta(days=1)  # inclusive last day -> exclusive
        if _key(end, zone) <= _key(start, zone):
            raise ToolError(f"event {args.title!r} ends before it starts")
    elif args.duration_minutes is not None:
        if not isinstance(start, datetime):
            raise ToolError(
                "duration_minutes needs a timed start; for several days give end"
            )
        end = _after(start, args.duration_minutes, zone)
    elif isinstance(start, datetime):
        end = _after(start, default_minutes, zone)
    else:
        end = start + timedelta(days=1)
    if args.repeat is not None and args.repeat.until is not None:
        first = start.date() if isinstance(start, datetime) else start
        if args.repeat.until < first:
            raise ToolError(f"event {args.title!r} repeats until before it starts")
    return Event(
        args.title,
        start,
        end,
        (args.location or "").strip() or None,
        (args.description or "").strip() or None,
        tuple(args.reminders),
        args.repeat,
    )


def _key(value: date | datetime, zone: Zone) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(_UTC)
    return datetime.combine(value, datetime.min.time(), tzinfo=zone.tzinfo).astimezone(
        _UTC
    )
