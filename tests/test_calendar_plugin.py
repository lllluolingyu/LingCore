"""Calendar plugin tests: core behavior."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from lingcore.errors import ToolError
from lingcore.plugins.discovery import discover_plugins
from tests.support.calendar_plugin import (
    BERLIN,
    add,
    cal,
    content_lines,
    ctx,
    ids,
    listed,
)


def test_manifest_provides_the_four_tools_without_hooks():
    manifest = discover_plugins(None)["calendar"].manifest
    assert manifest.provides == [
        "calendar_add",
        "calendar_update",
        "calendar_list",
        "calendar_remove",
    ]
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
