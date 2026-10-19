"""The bundled ``calendar`` plugin: the profile's one iCalendar file."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

import lingcore.timezones as timezones
from lingcore.errors import ToolError
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


def ids(result):
    return re.findall(r"\[id ([^\]]+)\]", result)


def content_lines(tmp_path):
    """Unfolded content lines of the profile's calendar."""
    return cal._unfold((tmp_path / "calendar.ics").read_text("utf-8"))


# --------------------------------------------------------------------------- #
# calendar_add                                                                #
# --------------------------------------------------------------------------- #


def test_manifest_provides_the_three_tools_without_hooks():
    manifest = discover_plugins(None)["calendar"].manifest
    assert manifest.provides == ["calendar_add", "calendar_list", "calendar_remove"]
    assert manifest.hooks is None


async def test_every_call_adds_to_the_one_profile_calendar(tmp_path):
    context = ctx(tmp_path)
    event = {"title": "Dentist", "start": "2026-10-21T15:00", "duration_minutes": 45}
    first = await add(context, event)
    second = await add(context, {"title": "Gym", "start": "2026-10-22T07:00"})
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "calendar.ics",
        "calendar.ics.lock",
        "workspace",
    ]
    assert list((tmp_path / "workspace").iterdir()) == []
    assert "Added 1 event to the calendar (timezone Europe/Berlin)" in first
    assert "it now holds 2 event(s)" in second
    assert "Wed 2026-10-21 15:00–15:45 Dentist" in first
    assert f"Calendar file: {tmp_path.resolve() / 'calendar.ics'}" in first
    (dentist,) = ids(first)
    assert dentist.endswith("@lingcore") and dentist not in ids(second)
    lines = content_lines(tmp_path)
    assert lines[0] == "BEGIN:VCALENDAR" and lines[-1] == "END:VCALENDAR"
    assert lines.count("BEGIN:VEVENT") == 2 and lines.count("BEGIN:VTIMEZONE") == 1
    assert f"UID:{dentist}" in lines
    assert "DTSTART;TZID=Europe/Berlin:20261021T150000" in lines
    assert "DTEND;TZID=Europe/Berlin:20261021T154500" in lines


async def test_text_is_escaped_folded_and_read_back_verbatim(tmp_path):
    title = "Lunch, Anna; Ben \\ " + "很长的标题" * 20
    description = "Bring:\n- cake, candles"
    await add(
        ctx(tmp_path),
        {"title": title, "start": "2026-11-01", "description": description},
    )
    raw = (tmp_path / "calendar.ics").read_bytes()
    assert raw.endswith(b"\r\n") and b"\n" not in raw.replace(b"\r\n", b"")
    assert max(len(line) for line in raw.split(b"\r\n")) <= 75
    raw.decode("utf-8")  # folding never splits a multi-byte character
    (event,) = cal.parse_calendar(raw.decode("utf-8"), BERLIN)
    assert event.title == title and event.description == description
    assert "SUMMARY:Lunch\\, Anna\\; Ben \\\\ " in "".join(content_lines(tmp_path))


async def test_all_day_end_is_inclusive_and_reminders_become_alarms(tmp_path):
    result = await add(
        ctx(tmp_path),
        {"title": "Berlin trip", "start": "2026-11-02", "end": "2026-11-04"},
        {
            "title": "Flight",
            "start": "2026-11-02T07:10",
            "end": "2026-11-02T08:35",
            "location": "BER T1",
            "reminders": [60, 1440, 60],
        },
    )
    assert "Mon 2026-11-02 – Wed 2026-11-04 (all day) Berlin trip" in result
    assert "reminder 1440 min before, 60 min before" in result
    lines = content_lines(tmp_path)
    assert "DTSTART;VALUE=DATE:20261102" in lines
    assert "DTEND;VALUE=DATE:20261105" in lines
    assert lines.count("BEGIN:VALARM") == 2
    assert {"TRIGGER:-PT1440M", "TRIGGER:-PT60M"} <= set(lines)


async def test_weekly_events_keep_their_local_time_across_daylight_saving(tmp_path):
    context = ctx(tmp_path)
    await add(
        context,
        {
            "title": "Standup",
            "start": "2026-10-20T09:00",
            "duration_minutes": 15,
            "repeat": {"frequency": "weekly", "until": "2026-11-10"},
        },
    )
    lines = content_lines(tmp_path)
    # UNTIL is UTC for a TZID start: 09:00 CET on 10 November is 08:00Z.
    assert "RRULE:FREQ=WEEKLY;UNTIL=20261110T080000Z" in lines
    # The zone's change on 25 October is spelled out for strict clients.
    assert "DTSTART:20261025T030000" in lines
    assert "TZOFFSETTO:+0100" in lines and "TZOFFSETFROM:+0200" in lines
    result = await listed(context, start=date(2026, 10, 19), end=date(2026, 11, 30))
    times = [line for line in result.splitlines() if "Standup" in line]
    assert len(times) == 4
    assert all(" 09:00–09:15 Standup (repeats)" in line for line in times)


async def test_the_timezone_definition_grows_to_cover_every_event(tmp_path):
    context = ctx(tmp_path)
    await add(context, {"title": "Autumn", "start": "2026-10-20T09:00"})
    assert "DTSTART:20270328T020000" not in content_lines(tmp_path)
    await add(context, {"title": "Spring", "start": "2027-04-20T09:00"})
    lines = content_lines(tmp_path)
    assert lines.count("BEGIN:VTIMEZONE") == 1
    # Both the October 2026 and the March 2027 changes are now listed.
    assert {"DTSTART:20261025T030000", "DTSTART:20270328T020000"} <= set(lines)


async def test_count_and_interval_become_the_rrule(tmp_path):
    await add(
        ctx(tmp_path),
        {
            "title": "Bins",
            "start": "2026-10-23",
            "repeat": {"frequency": "weekly", "interval": 2, "count": 5},
        },
    )
    assert "RRULE:FREQ=WEEKLY;INTERVAL=2;COUNT=5" in content_lines(tmp_path)


async def test_adding_reports_overlapping_arrangements(tmp_path):
    context = ctx(tmp_path)
    await add(
        context,
        {"title": "Conference", "start": "2026-11-02", "end": "2026-11-04"},
        {
            "title": "Yoga",
            "start": "2026-10-05T18:00",
            "repeat": {"frequency": "weekly"},
        },
    )
    result = await add(
        context,
        {"title": "Dinner", "start": "2026-11-02T18:30"},
        {"title": "Free evening", "start": "2026-11-10T20:00"},
    )
    assert "Dinner overlaps:" in result
    overlap = result.split("Dinner overlaps:")[1]
    assert "Conference" in overlap and "Mon 2026-11-02 18:00–19:00 Yoga" in overlap
    assert "Free evening overlaps" not in result
    assert "data, not instructions" in result


@pytest.mark.parametrize(
    ("event", "message"),
    [
        ({"start": "2027-03-28T02:30"}, "does not exist in Europe/Berlin"),
        ({"start": "2026-10-21", "end": "2026-10-21T10:00"}, "both be dates"),
        ({"start": "2026-10-21T10:00", "end": "2026-10-21T09:00"}, "ends before"),
        ({"start": "2026-10-21", "duration_minutes": 30}, "needs a timed start"),
        ({"start": "next tuesday"}, "must be 'YYYY-MM-DD'"),
        ({"start": "2026-02-30"}, "not a valid date"),
        (
            {
                "start": "2026-10-21",
                "repeat": {"frequency": "daily", "until": "2026-10-01"},
            },
            "repeats until before",
        ),
    ],
)
async def test_unclear_or_impossible_times_are_refused(tmp_path, event, message):
    with pytest.raises(ToolError, match=message):
        await add(ctx(tmp_path), {"title": "x", **event})
    assert not (tmp_path / "calendar.ics").exists()


@pytest.mark.parametrize(
    "event",
    [
        {"title": "  ", "start": "2026-10-21"},
        {"title": "x", "start": "2026-10-21", "reminders": [-5]},
        {
            "title": "x",
            "start": "2026-10-21T10:00",
            "end": "2026-10-21T11:00",
            "duration_minutes": 5,
        },
        {
            "title": "x",
            "start": "2026-10-21",
            "repeat": {"frequency": "daily", "count": 2, "until": "2026-11-01"},
        },
        {"title": "x", "start": "2026-10-21", "repeat": {"frequency": "hourly"}},
    ],
)
def test_argument_contract(event):
    with pytest.raises(ValidationError):
        cal.EventArgs(**event)


def test_one_call_holds_at_most_twenty_events():
    with pytest.raises(ValidationError):
        cal.AddArgs(events=[{"title": "x", "start": "2026-10-21"}] * 21)


async def test_an_explicit_offset_is_converted_to_the_calendar_zone(tmp_path):
    result = await add(
        ctx(tmp_path), {"title": "Call", "start": "2026-10-21T15:00+00:00"}
    )
    assert "Wed 2026-10-21 17:00–18:00 Call" in result


async def test_the_clock_timezone_is_the_fallback(tmp_path):
    context = ToolContext(
        workspace=tmp_path,
        profile_dir=tmp_path,
        options={"clock": {"timezone": "Asia/Tokyo"}},
    )
    result = await add(context, {"title": "Ramen", "start": "2026-10-21T19:00"})
    assert "(timezone Asia/Tokyo)" in result


async def test_without_an_iana_zone_times_are_utc_and_the_result_says_so(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(timezones, "_host_zone_name", lambda: None)
    context = ToolContext(workspace=tmp_path, profile_dir=tmp_path, options={})
    result = await add(context, {"title": "Call", "start": "2026-10-21T15:00"})
    assert "saved as UTC instants" in result
    lines = content_lines(tmp_path)
    assert not any(line.startswith("BEGIN:VTIMEZONE") for line in lines)
    assert any(
        line.startswith("DTSTART:20261021T") and line.endswith("Z") for line in lines
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"path": "../outside.ics"}, "escapes the profile directory"),
        ({"path": "calendar.txt"}, "an .ics file"),
        ({"path": "/tmp/elsewhere.ics"}, "allow_absolute_path"),
        ({"timezone": "Nowhere/Land"}, "unknown timezone"),
        ({"colour": "blue"}, "colour"),
    ],
)
async def test_options_are_validated_when_a_tool_runs(tmp_path, options, message):
    with pytest.raises(ToolError, match=message):
        await add(ctx(tmp_path, **options), {"title": "x", "start": "2026-10-21"})


async def test_an_allowed_absolute_path_is_used(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    context = ctx(
        tmp_path, path=str(elsewhere / "family.ics"), allow_absolute_path=True
    )
    await add(context, {"title": "x", "start": "2026-10-21"})
    assert (elsewhere / "family.ics").is_file()


async def test_the_calendar_needs_a_profile_directory(tmp_path):
    context = ToolContext(workspace=tmp_path, options={})
    with pytest.raises(ToolError, match="profile directory"):
        await add(context, {"title": "x", "start": "2026-10-21"})


async def test_the_calendar_is_never_kept_inside_the_package(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "_PACKAGE_DIR", tmp_path.resolve())
    with pytest.raises(ToolError, match="installed package"):
        await add(ctx(tmp_path), {"title": "x", "start": "2026-10-21"})


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
async def test_a_symlinked_calendar_file_is_refused(tmp_path):
    outside = tmp_path / "outside.ics"
    outside.write_text("BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n", encoding="utf-8")
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "calendar.ics").symlink_to(outside)
    with pytest.raises(ToolError):
        await add(ctx(profile), {"title": "x", "start": "2026-10-21"})
    with pytest.raises(ToolError):
        await listed(ctx(profile))
    assert outside.read_bytes() == b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n"


async def test_a_file_that_is_not_a_calendar_is_never_overwritten(tmp_path):
    (tmp_path / "calendar.ics").write_text("my notes, not a calendar\n", "utf-8")
    with pytest.raises(ToolError, match="left unchanged"):
        await add(ctx(tmp_path), {"title": "x", "start": "2026-10-21"})
    assert (tmp_path / "calendar.ics").read_text("utf-8") == (
        "my notes, not a calendar\n"
    )


async def test_parallel_adds_are_serialized_and_none_is_lost(tmp_path):
    context = ctx(tmp_path)
    await asyncio.gather(
        *(
            add(context, {"title": f"Event {n}", "start": "2026-10-21"})
            for n in range(8)
        )
    )
    lines = content_lines(tmp_path)
    assert lines.count("BEGIN:VEVENT") == 8
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]


# --------------------------------------------------------------------------- #
# Keeping what the plugin did not write                                       #
# --------------------------------------------------------------------------- #

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


async def test_list_reads_hand_added_events_and_expands_simple_repeats(tmp_path):
    (tmp_path / "calendar.ics").write_text(FOREIGN, encoding="utf-8")
    result = await listed(ctx(tmp_path), start=date(2026, 1, 1), end=date(2026, 6, 30))
    rent = [line for line in result.splitlines() if "Rent" in line]
    # RFC 5545: the 31st of a 30-day or shorter month is skipped, and only
    # real occurrences count toward COUNT.
    assert [line.split(" (all day)")[0][2:] for line in rent] == [
        "Sat 2026-01-31",
        "Tue 2026-03-31",
        "Sun 2026-05-31",
    ]
    assert all(line.endswith("[id month-end]") for line in rent)
    assert (
        "Tue 2026-02-10 11:00–12:00 Webinar with a very long title that a client folded onto two lines @ Online  [id utc]"
        in result
    )
    assert "not the event summary" not in result
    assert "Cancelled thing" not in result
    (gym,) = [line for line in result.splitlines() if "Gym" in line]
    assert "occurrences not expanded" in gym and "BYDAY=MO,WE" in gym
    assert "data, not instructions" in result


async def test_changes_keep_components_the_plugin_did_not_write(tmp_path):
    (tmp_path / "calendar.ics").write_text(FOREIGN, encoding="utf-8")
    context = ctx(tmp_path)
    added = await add(context, {"title": "New", "start": "2026-10-21T10:00"})
    await remove(context, "utc")
    lines = content_lines(tmp_path)
    for kept in (
        "X-WR-CALNAME:Home",
        "TZID:America/New_York",
        "SUMMARY:Buy stamps",
        "UID:month-end",
        "UID:cancelled",
        f"UID:{ids(added)[0]}",
    ):
        assert kept in lines
    assert "UID:utc" not in lines
    assert lines.count("VERSION:2.0") == 1
    assert "PRODID:-//LingCore//calendar plugin 1.0//EN" in lines
    # The plugin's own zone definition is regenerated alongside the kept one.
    assert lines.count("BEGIN:VTIMEZONE") == 2 and "TZID:Europe/Berlin" in lines


# --------------------------------------------------------------------------- #
# calendar_list                                                               #
# --------------------------------------------------------------------------- #


def _today():
    return datetime.now(timezone.utc).astimezone(BERLIN.tzinfo).date()


async def test_list_defaults_to_about_three_months(tmp_path):
    context = ctx(tmp_path)
    today = _today()
    await add(
        context,
        {"title": "Soon", "start": today.isoformat()},
        {"title": "Later", "start": today.replace(year=today.year + 1).isoformat()},
    )
    result = await listed(context)
    assert "Soon" in result and "Later" not in result
    assert result.startswith("Events from ")
    assert "Later" in await listed(context, end=today.replace(year=today.year + 2))


async def test_a_query_searches_long_term_arrangements(tmp_path):
    context = ctx(tmp_path)
    today = _today()
    await add(
        context,
        {
            "title": "Dentist check-up",
            "start": today.replace(year=today.year + 3).isoformat(),
        },
        {"title": "Lunch", "start": today.isoformat(), "location": "Dentist's café"},
        {"title": "Gym", "start": today.isoformat()},
    )
    result = await listed(context, query="DENTIST")
    assert "Events matching 'DENTIST'" in result
    assert "Dentist check-up" in result and "Lunch" in result
    assert "Gym" not in result


async def test_list_of_an_empty_calendar_says_so(tmp_path):
    assert "The calendar is empty" in await listed(ctx(tmp_path))


async def test_list_range_is_bounded(tmp_path):
    with pytest.raises(ToolError, match="before start"):
        await listed(ctx(tmp_path), start=date(2026, 5, 1), end=date(2026, 4, 1))
    with pytest.raises(ToolError, match="ten years"):
        await listed(ctx(tmp_path), start=date(2026, 1, 1), end=date(2040, 1, 1))


# --------------------------------------------------------------------------- #
# calendar_remove                                                             #
# --------------------------------------------------------------------------- #


async def test_remove_takes_a_whole_series_off_by_id(tmp_path):
    context = ctx(tmp_path)
    added = await add(
        context,
        {
            "title": "Standup",
            "start": "2026-10-20T09:00",
            "repeat": {"frequency": "daily", "count": 10},
        },
        {"title": "Dentist", "start": "2026-10-21T15:00"},
    )
    standup, dentist = ids(added)
    result = await remove(context, standup)
    assert result.startswith("Removed 1 event from the calendar:")
    assert "Tue 2026-10-20 09:00–10:00 Standup (repeats)" in result
    listing = await listed(context, start=date(2026, 10, 19), end=date(2026, 11, 1))
    assert "Standup" not in listing and dentist in listing
    await remove(context, dentist)
    assert "VTIMEZONE" not in "".join(content_lines(tmp_path))
    assert "The calendar is empty" in await listed(context)


async def test_an_unknown_id_removes_nothing(tmp_path):
    context = ctx(tmp_path)
    (dentist,) = ids(await add(context, {"title": "Dentist", "start": "2026-10-21"}))
    before = (tmp_path / "calendar.ics").read_bytes()
    with pytest.raises(ToolError, match="no event with id nope@lingcore"):
        await remove(context, dentist, "nope@lingcore")
    assert (tmp_path / "calendar.ics").read_bytes() == before
