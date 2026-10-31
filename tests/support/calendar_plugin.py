"""Fixtures and builders for calendar plugin tests."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

import lingcore.timezones as timezones
from lingcore.plugins.discovery import discover_plugins
from lingcore.skills import _load_tool_module
from lingcore.tools import ToolContext


def _load_calendar():
    plugin = discover_plugins(None)["calendar"]
    assert plugin.manifest.module is not None
    _load_tool_module(
        "calendar",
        plugin.root / plugin.manifest.module,
        plugin.manifest.provides,
        prefix="calendar",
    )
    name = next(
        n for n in sys.modules if n.startswith("lingcore_skill_tools.calendar.")
    )
    return sys.modules[name]


cal = _load_calendar()


BERLIN = timezones.resolve_zone("Europe/Berlin")


def ctx(tmp_path, **calendar):
    options = {"calendar": {"timezone": "Europe/Berlin", **calendar}}
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return ToolContext(workspace=workspace, profile_dir=tmp_path, options=options)


async def add(context, *events):
    return await cal.calendar_add(
        cal.AddArgs(events=[cal.EventArgs(**e) for e in events]), context
    )


async def listed(context, **args):
    return await cal.calendar_list(cal.ListArgs(**args), context)


async def remove(context, *ids):
    return await cal.calendar_remove(cal.RemoveArgs(ids=list(ids)), context)


async def update(context, id, *, scope=None, occurrence=None, **changes):
    return await cal.calendar_update(
        cal.UpdateArgs(id=id, changes=changes, scope=scope, occurrence=occurrence),
        context,
    )


def ids(result):
    return re.findall(r"\[id ([^\]]+)\]", result)


def content_lines(tmp_path):
    """Unfolded content lines of the profile's calendar."""
    return cal._unfold((tmp_path / "calendar.ics").read_text("utf-8"))


FOREIGN = "\r\n".join(
    [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Example//EN",
        "X-WR-CALNAME:Home",
        "BEGIN:VTIMEZONE",
        "TZID:America/New_York",
        "BEGIN:STANDARD",
        "DTSTART:19701101T020000",
        "TZOFFSETFROM:-0400",
        "TZOFFSETTO:-0500",
        "END:STANDARD",
        "END:VTIMEZONE",
        "BEGIN:VTODO",
        "UID:todo-1",
        "SUMMARY:Buy stamps",
        "END:VTODO",
        "BEGIN:VEVENT",
        "UID:month-end",
        "SUMMARY:Rent",
        "DTSTART;VALUE=DATE:20260131",
        "RRULE:FREQ=MONTHLY;COUNT=4",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "UID:byday",
        "SUMMARY:Gym",
        "DTSTART;TZID=Europe/Berlin:20260105T070000",
        "DURATION:PT1H",
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "UID:cancelled",
        "SUMMARY:Cancelled thing",
        "STATUS:CANCELLED",
        "DTSTART:20260210T100000Z",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "UID:utc",
        "SUMMARY:Webinar with a very long title that a client folded onto two li",
        " nes",
        "DTSTART:20260210T100000Z",
        "DTEND:20260210T110000Z",
        "BEGIN:VALARM",
        "TRIGGER:-PT5M",
        "DESCRIPTION:not the event summary",
        "END:VALARM",
        "LOCATION:Online",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ]
)


def _today():
    return datetime.now(timezone.utc).astimezone(BERLIN.tzinfo).date()
