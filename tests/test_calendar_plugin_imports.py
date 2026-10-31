"""Calendar plugin tests: imports."""

from __future__ import annotations

import asyncio
import os
from datetime import date

import pytest

from lingcore.errors import ToolError
from tests.support.calendar_plugin import (
    FOREIGN,
    _today,
    add,
    cal,
    content_lines,
    ctx,
    ids,
    listed,
    remove,
    update,
)


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
    text = FOREIGN
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


async def test_update_moves_an_event_without_losing_id_or_details(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Dentist",
                "start": "2026-10-21T15:00",
                "duration_minutes": 45,
                "location": "Clinic",
                "description": "Bring insurance",
                "reminders": [60],
            },
        )
    )
    path = tmp_path / "calendar.ics"
    path.write_text(
        path.read_text().replace(
            "SUMMARY:Dentist",
            "SEQUENCE:4\nCREATED:20261001T000000Z\n"
            "ATTENDEE:mailto:dentist@example.com\nX-CUSTOM:keep\nSUMMARY:Dentist",
        )
    )
    result = await update(context, uid, start="2026-10-22T10:00")
    lines = content_lines(tmp_path)
    assert ids(result) == [uid] and lines.count("BEGIN:VEVENT") == 1
    assert "Thu 2026-10-22 10:00–10:45 Dentist @ Clinic" in result
    for kept in (
        "DESCRIPTION:Bring insurance",
        "TRIGGER:-PT60M",
        "X-CUSTOM:keep",
        "ATTENDEE:mailto:dentist@example.com",
        "CREATED:20261001T000000Z",
        "SEQUENCE:5",
    ):
        assert kept in lines
    assert "DTSTART;TZID=Europe/Berlin:20261022T100000" in lines
    assert any(line.startswith("LAST-MODIFIED:") for line in lines)
    assert "Dentist" not in await listed(
        context, start=date(2026, 10, 21), end=date(2026, 10, 21)
    )
