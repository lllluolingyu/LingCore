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


async def test_an_explicit_second_occurrence_keeps_its_instant(tmp_path):
    context = ctx(tmp_path)
    result = await add(
        context,
        {"title": "Late call", "start": "2026-10-25T02:30+01:00"},
    )
    # A TZID-local 02:30 would mean the first occurrence (00:30Z).
    assert "DTSTART:20261025T013000Z" in content_lines(tmp_path)
    assert "DTEND:20261025T023000Z" in content_lines(tmp_path)
    (event,) = cal.parse_calendar((tmp_path / "calendar.ics").read_text(), BERLIN)
    assert event.start.astimezone(timezone.utc) == datetime(
        2026, 10, 25, 1, 30, tzinfo=timezone.utc
    )
    assert "02:30 (UTC+01:00)–03:30 (UTC+01:00) Late call" in result
    assert "Late call" in await listed(
        context, start=date(2026, 10, 25), end=date(2026, 10, 25)
    )


@pytest.mark.parametrize(
    ("length", "end_stamp", "display_end"),
    [
        ({"end": "2026-10-25T02:15+01:00"}, "011500Z", "02:15"),
        ({"duration_minutes": 45}, "011500Z", "02:15"),
        ({"end": "2026-10-25T02:30+01:00"}, "013000Z", "02:30"),
        ({}, "013000Z", "02:30"),
    ],
)
async def test_events_crossing_the_fall_back_keep_elapsed_duration(
    tmp_path, length, end_stamp, display_end
):
    context = ctx(tmp_path)
    result = await add(
        context,
        {"title": "Crossing", "start": "2026-10-25T02:30+02:00", **length},
    )
    lines = content_lines(tmp_path)
    assert "DTSTART;TZID=Europe/Berlin:20261025T023000" in lines
    assert f"DTEND:20261025T{end_stamp}" in lines
    span = f"02:30 (UTC+02:00)–{display_end} (UTC+01:00) Crossing"
    assert span in result
    assert span in await listed(
        context, start=date(2026, 10, 25), end=date(2026, 10, 25)
    )


async def test_a_later_wall_clock_end_cannot_precede_the_start_in_utc(tmp_path):
    with pytest.raises(ToolError, match="ends before"):
        await add(
            ctx(tmp_path),
            {
                "title": "Backwards",
                "start": "2026-10-25T02:15+01:00",
                "end": "2026-10-25T02:30+02:00",
            },
        )
    assert not (tmp_path / "calendar.ics").exists()


async def test_overlap_checks_distinguish_the_two_fall_back_hours(tmp_path):
    context = ctx(tmp_path)
    await add(
        context,
        {
            "title": "First hour",
            "start": "2026-10-25T02:10+02:00",
            "duration_minutes": 20,
        },
    )
    later = await add(
        context,
        {
            "title": "Second hour",
            "start": "2026-10-25T02:10+01:00",
            "duration_minutes": 20,
        },
    )
    assert "overlaps:" not in later
    clash = await add(
        context,
        {
            "title": "Actual clash",
            "start": "2026-10-25T02:15+01:00",
            "duration_minutes": 5,
        },
    )
    assert "Actual clash overlaps:" in clash and "Second hour" in clash
    assert "First hour" not in clash
    listing = await listed(context, start=date(2026, 10, 25), end=date(2026, 10, 25))
    assert listing.index("First hour") < listing.index("Second hour")


@pytest.mark.parametrize(
    ("frequency", "start", "days"),
    [
        ("weekly", "2027-03-21T02:30", ["2027-03-21", "2027-04-04", "2027-04-11"]),
        ("daily", "2027-03-27T02:30", ["2027-03-27", "2027-03-29", "2027-03-30"]),
    ],
)
async def test_nonexistent_repeat_times_do_not_consume_count(
    tmp_path, frequency, start, days
):
    context = ctx(tmp_path)
    await add(
        context,
        {
            "title": "Early call",
            "start": start,
            "duration_minutes": 30,
            "repeat": {"frequency": frequency, "count": 3},
        },
    )
    result = await listed(context, start=date(2027, 3, 20), end=date(2027, 4, 15))
    calls = [line for line in result.splitlines() if "Early call" in line]
    assert [line.split()[2] for line in calls] == days
    assert all("02:30–03:00" in line for line in calls)


async def test_repeated_event_end_uses_elapsed_time_across_spring_forward(tmp_path):
    context = ctx(tmp_path)
    await add(
        context,
        {
            "title": "Two hours",
            "start": "2027-03-21T01:30",
            "duration_minutes": 120,
            "repeat": {"frequency": "weekly", "count": 2},
        },
    )
    result = await listed(context, start=date(2027, 3, 21), end=date(2027, 3, 28))
    assert "2027-03-21 01:30–03:30 Two hours" in result
    assert "2027-03-28 01:30 (UTC+01:00)–04:30 (UTC+02:00) Two hours" in result


@pytest.mark.parametrize("ending", [{"count": 30}, {"until": "2027-05-01"}])
async def test_repeating_second_occurrence_keeps_future_local_time(tmp_path, ending):
    context = ctx(tmp_path)
    await add(
        context,
        {
            "title": "Late series",
            "start": "2026-10-25T02:30+01:00",
            "duration_minutes": 30,
            "repeat": {"frequency": "weekly", **ending},
        },
    )
    lines = content_lines(tmp_path)
    assert "DTSTART;TZID=Europe/Berlin:20261025T023000" in lines
    assert "EXDATE:20261025T003000Z" in lines
    assert "RDATE:20261025T013000Z" in lines
    first = await listed(context, start=date(2026, 10, 25), end=date(2026, 10, 25))
    assert first.count("Late series") == 1
    assert "02:30 (UTC+01:00)–03:00 (UTC+01:00) Late series" in first
    summer = await listed(context, start=date(2027, 4, 4), end=date(2027, 4, 4))
    assert "02:30–03:00 Late series" in summer
    assert "Coverage incomplete" not in first + summer


async def test_replacing_the_first_fold_occurrence_preserves_count(tmp_path):
    context = ctx(tmp_path)
    await add(
        context,
        {
            "title": "Only once",
            "start": "2026-10-25T02:30+01:00",
            "repeat": {"frequency": "weekly", "count": 1},
        },
    )
    result = await listed(context, start=date(2026, 10, 25), end=date(2026, 11, 30))
    assert result.count("Only once") == 1 and "2026-11-01" not in result


async def test_second_occurrence_in_a_half_hour_transition(tmp_path):
    context = ctx(tmp_path, timezone="Australia/Lord_Howe")
    result = await add(
        context,
        {"title": "Half-hour shift", "start": "2026-04-05T01:45+10:30"},
    )
    assert "DTSTART:20260404T151500Z" in content_lines(tmp_path)
    assert "01:45 (UTC+10:30)–02:45 (UTC+10:30)" in result


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


@pytest.mark.parametrize(
    "rule",
    [
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE",
        "RRULE:FREQ=WEEKLY;INTERVAL=bad",
        "RDATE;VALUE=PERIOD:20260202T070000Z/PT1H",
    ],
)
async def test_unexpanded_series_remain_visible_after_the_original_date(tmp_path, rule):
    (tmp_path / "calendar.ics").write_text(
        FOREIGN.replace("RRULE:FREQ=WEEKLY;BYDAY=MO,WE", rule), encoding="utf-8"
    )
    context = ctx(tmp_path)
    result = await listed(
        context, start=date(2026, 2, 1), end=date(2026, 2, 28), query="Gym"
    )
    assert "0 confirmed occurrence(s); 1 unexpanded series" in result
    assert "Coverage incomplete" in result
    assert "do not assume unlisted times are free" in result
    assert "original start shown" in result
    assert result.count("[id byday]") == 1
    assert "2026-01-05 07:00–08:00 Gym" in result
    assert "occurrences not expanded" in result
    unrelated = await listed(
        context, start=date(2026, 2, 1), end=date(2026, 2, 28), query="Webinar"
    )
    assert "Coverage incomplete" not in unrelated and "Gym" not in unrelated


async def test_scheduling_warns_about_unknown_series_without_claiming_a_clash(tmp_path):
    (tmp_path / "calendar.ics").write_text(FOREIGN, encoding="utf-8")
    result = await add(
        ctx(tmp_path), {"title": "Breakfast", "start": "2026-02-02T07:30"}
    )
    assert "Coverage incomplete" in result and "Gym" in result
    assert "Breakfast overlaps:" not in result


async def test_a_series_with_an_override_stays_visible_in_later_windows(tmp_path):
    text = FOREIGN.replace("RRULE:FREQ=WEEKLY;BYDAY=MO,WE", "RRULE:FREQ=WEEKLY")
    text = text.replace(
        "END:VCALENDAR",
        "BEGIN:VEVENT\r\nUID:byday\r\nSUMMARY:Moved gym\r\n"
        "RECURRENCE-ID;TZID=Europe/Berlin:20260112T070000\r\n"
        "DTSTART;TZID=Europe/Berlin:20260112T090000\r\n"
        "DTEND;TZID=Europe/Berlin:20260112T100000\r\n"
        "END:VEVENT\r\nEND:VCALENDAR",
    )
    (tmp_path / "calendar.ics").write_text(text, encoding="utf-8")
    result = await listed(ctx(tmp_path), start=date(2026, 2, 1), end=date(2026, 2, 28))
    assert "Coverage incomplete" in result and "Gym" in result
    assert "Moved gym" not in result


async def test_simple_recurrence_dates_are_unioned_and_exclusions_win(tmp_path):
    text = FOREIGN.replace(
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE",
        "RRULE:FREQ=WEEKLY;COUNT=3\r\n"
        "RDATE;TZID=Europe/Berlin:20260112T070000,20260202T070000\r\n"
        "RDATE:20260209T060000Z\r\n"
        "EXDATE;TZID=Europe/Berlin:20260112T070000\r\n"
        "EXDATE:20260119T060000Z",
    )
    (tmp_path / "calendar.ics").write_text(text, encoding="utf-8")
    result = await listed(
        ctx(tmp_path), start=date(2026, 1, 1), end=date(2026, 2, 28), query="Gym"
    )
    gym = [line for line in result.splitlines() if "[id byday]" in line]
    assert [line.split()[2] for line in gym] == [
        "2026-01-05",
        "2026-02-02",
        "2026-02-09",
    ]
    assert "Coverage incomplete" not in result


@pytest.mark.parametrize(
    ("duration", "normal_end", "transition_end"),
    [
        ("P1D", "2027-03-22 01:30", "2027-03-29 01:30"),
        ("PT24H", "2027-03-22 01:30", "2027-03-29 02:30"),
        ("PT2H", "03:30", "04:30"),
    ],
)
async def test_imported_durations_preserve_calendar_days_and_elapsed_hours(
    tmp_path, duration, normal_end, transition_end
):
    (tmp_path / "calendar.ics").write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:duration\n"
        "SUMMARY:Imported duration\nDTSTART;TZID=Europe/Berlin:20270321T013000\n"
        f"DURATION:{duration}\nRRULE:FREQ=WEEKLY;COUNT=2\n"
        "END:VEVENT\nEND:VCALENDAR\n",
        encoding="utf-8",
    )
    result = await listed(ctx(tmp_path), start=date(2027, 3, 21), end=date(2027, 3, 29))
    lines = [line for line in result.splitlines() if "Imported duration" in line]
    assert len(lines) == 2
    assert normal_end in lines[0] and transition_end in lines[1]
    assert "UTC+02:00" in lines[1]


async def test_all_day_extra_dates_keep_their_date_semantics(tmp_path):
    (tmp_path / "calendar.ics").write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:dates\n"
        "SUMMARY:All day\nDTSTART;VALUE=DATE:20270327\nDURATION:P1D\n"
        "RDATE;VALUE=DATE:20270328,20270404\nEXDATE;VALUE=DATE:20270327\n"
        "END:VEVENT\nEND:VCALENDAR\n",
        encoding="utf-8",
    )
    result = await listed(ctx(tmp_path), start=date(2027, 3, 27), end=date(2027, 4, 4))
    lines = [line for line in result.splitlines() if "[id dates]" in line]
    assert [line.split()[2] for line in lines] == ["2027-03-28", "2027-04-04"]
    assert all("(all day) All day" in line for line in lines)
    assert "Coverage incomplete" not in result


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
