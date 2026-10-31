"""Calendar plugin tests: recurrence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import lingcore.timezones as timezones
from lingcore.errors import ToolError
from lingcore.tools import ToolContext
from tests.support.calendar_plugin import add, cal, content_lines, ctx


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
