"""Calendar plugin tests: updates."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest
from pydantic import ValidationError

from lingcore.errors import ToolError
from tests.support.calendar_plugin import (
    FOREIGN,
    add,
    cal,
    content_lines,
    ctx,
    ids,
    listed,
    remove,
    update,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"title": None},
        {"start": None},
        {"reminders": None},
        {},
        {"title": "  "},
        {"reminders": [-1]},
        {"extra": "no"},
        {"end": "2026-10-21T16:00", "duration_minutes": 30},
    ],
)
def test_update_rejects_invalid_patches(changes):
    with pytest.raises(ValidationError):
        cal.UpdateArgs(id="example", changes=changes)


async def test_update_can_clear_optional_details_and_replace_reminders(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Call",
                "start": "2026-10-21T15:00",
                "location": "Office",
                "description": "Old notes",
                "reminders": [60, 10],
            },
        )
    )
    await update(context, uid, location=None, description=None, reminders=[])
    lines = content_lines(tmp_path)
    assert not any(
        line.startswith(("LOCATION:", "DESCRIPTION:", "BEGIN:VALARM")) for line in lines
    )
    await update(context, uid, title="New call", reminders=[15])
    lines = content_lines(tmp_path)
    assert "SUMMARY:New call" in lines and "TRIGGER:-PT15M" in lines
    assert lines.count("BEGIN:VALARM") == 1


async def test_invalid_update_keeps_the_original_file_byte_for_byte(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(await add(context, {"title": "Call", "start": "2026-10-21T15:00"}))
    before = (tmp_path / "calendar.ics").read_bytes()
    with pytest.raises(ToolError, match="ends before"):
        await update(context, uid, start="2026-10-22T10:00", end="2026-10-22T09:00")
    assert (tmp_path / "calendar.ics").read_bytes() == before
    with pytest.raises(ToolError, match="no unique event"):
        await update(context, "missing", title="No")
    assert (tmp_path / "calendar.ics").read_bytes() == before


async def test_failed_update_publication_keeps_the_original(tmp_path, monkeypatch):
    context = ctx(tmp_path)
    (uid,) = ids(await add(context, {"title": "Call", "start": "2026-10-21T15:00"}))
    before = (tmp_path / "calendar.ics").read_bytes()

    def fail_replace(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(cal.ConfinedDirectory, "replace", fail_replace)
    with pytest.raises(ToolError, match="simulated replace failure"):
        await update(context, uid, start="2026-10-22T10:00")
    assert (tmp_path / "calendar.ics").read_bytes() == before
    assert not list(tmp_path.glob("*.part"))


async def test_concurrent_patches_merge_against_fresh_saved_state(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(await add(context, {"title": "Call", "start": "2026-10-21T15:00"}))
    await asyncio.gather(
        update(context, uid, location="New office"),
        update(ctx(tmp_path), uid, description="Bring notes"),
    )
    lines = content_lines(tmp_path)
    assert "LOCATION:New office" in lines and "DESCRIPTION:Bring notes" in lines
    assert lines.count("BEGIN:VEVENT") == 1


async def test_update_reports_conflicts_but_does_not_conflict_with_itself(tmp_path):
    context = ctx(tmp_path)
    first, _ = ids(
        await add(
            context,
            {"title": "Dentist", "start": "2026-10-21T15:00"},
            {"title": "Lunch", "start": "2026-10-21T12:00"},
        )
    )
    result = await update(context, first, title="Check-up")
    assert "overlaps:" not in result
    result = await update(context, first, start="2026-10-21T12:30")
    assert "Check-up overlaps:" in result and "Lunch" in result


async def test_all_day_update_preserves_the_number_of_days(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context, {"title": "Trip", "start": "2026-11-01", "end": "2026-11-03"}
        )
    )
    result = await update(context, uid, start="2026-11-05")
    assert "2026-11-05 – Sat 2026-11-07 (all day) Trip" in result
    assert "DTEND;VALUE=DATE:20261108" in content_lines(tmp_path)


async def test_update_preserves_explicit_fall_back_instants(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {"title": "Call", "start": "2026-10-24T09:00", "duration_minutes": 45},
        )
    )
    result = await update(context, uid, start="2026-10-25T02:30+01:00")
    assert "DTSTART:20261025T013000Z" in content_lines(tmp_path)
    assert "DTEND:20261025T021500Z" in content_lines(tmp_path)
    assert "02:30 (UTC+01:00)–03:15 (UTC+01:00)" in result


async def test_a_series_update_requires_explicit_scope_and_keeps_its_rule(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Standup",
                "start": "2026-10-20T09:00",
                "duration_minutes": 15,
                "repeat": {"frequency": "weekly", "count": 3},
            },
        )
    )
    before = (tmp_path / "calendar.ics").read_bytes()
    with pytest.raises(ToolError, match="choose scope"):
        await update(context, uid, start="2026-10-20T10:00")
    assert (tmp_path / "calendar.ics").read_bytes() == before
    result = await update(context, uid, scope="series", start="2026-10-20T10:00")
    assert "Updated the series" in result and ids(result) == [uid]
    assert "RRULE:FREQ=WEEKLY;COUNT=3" in content_lines(tmp_path)
    listing = await listed(context, start=date(2026, 10, 20), end=date(2026, 11, 3))
    assert listing.count("10:00–10:15 Standup") == 3
    assert "09:00" not in listing


async def test_update_can_add_replace_and_clear_a_repeat_rule(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(await add(context, {"title": "Bins", "start": "2026-10-21"}))
    await update(context, uid, repeat={"frequency": "weekly", "count": 3})
    assert "RRULE:FREQ=WEEKLY;COUNT=3" in content_lines(tmp_path)
    await update(
        context, uid, scope="series", repeat={"frequency": "monthly", "count": 2}
    )
    assert "RRULE:FREQ=MONTHLY;COUNT=2" in content_lines(tmp_path)
    await update(context, uid, scope="series", repeat=None)
    assert not any(line.startswith("RRULE:") for line in content_lines(tmp_path))


async def test_adding_a_repeat_to_a_second_fold_instant_keeps_future_local_time(
    tmp_path,
):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(context, {"title": "Late", "start": "2026-10-25T02:30+01:00"})
    )
    await update(context, uid, repeat={"frequency": "weekly", "count": 30})
    lines = content_lines(tmp_path)
    assert "DTSTART;TZID=Europe/Berlin:20261025T023000" in lines
    assert "RDATE:20261025T013000Z" in lines
    assert "EXDATE:20261025T003000Z" in lines
    first = await listed(context, start=date(2026, 10, 25), end=date(2026, 10, 25))
    summer = await listed(context, start=date(2027, 4, 4), end=date(2027, 4, 4))
    assert "02:30 (UTC+01:00)–03:30 (UTC+01:00) Late" in first
    assert "02:30–03:30 Late" in summer


async def test_all_day_occurrence_update_leaves_the_other_dates_intact(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Bins",
                "start": "2026-10-21",
                "repeat": {"frequency": "weekly", "count": 3},
            },
        )
    )
    await update(
        context, uid, scope="occurrence", occurrence="2026-10-28", start="2026-10-29"
    )
    assert "RECURRENCE-ID;VALUE=DATE:20261028" in content_lines(tmp_path)
    result = await listed(context, start=date(2026, 10, 21), end=date(2026, 11, 4))
    assert (
        "2026-10-21 (all day) Bins" in result and "2026-11-04 (all day) Bins" in result
    )
    assert (
        "2026-10-29 (all day) Bins" in result
        and "Wed 2026-10-28 (all day)" not in result
    )


@pytest.mark.parametrize("range_override", [False, True])
async def test_cancelled_and_range_overrides_keep_correct_coverage(
    tmp_path, range_override
):
    range_param = ";RANGE=THISANDFUTURE" if range_override else ""
    (tmp_path / "calendar.ics").write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:series\nSUMMARY:Standup\n"
        "DTSTART;TZID=Europe/Berlin:20261020T090000\nDURATION:PT15M\n"
        "RRULE:FREQ=WEEKLY;COUNT=3\nEND:VEVENT\nBEGIN:VEVENT\nUID:series\n"
        f"RECURRENCE-ID;TZID=Europe/Berlin{range_param}:20261027T090000\n"
        "SUMMARY:Cancelled\nSTATUS:CANCELLED\n"
        "DTSTART;TZID=Europe/Berlin:20261027T090000\nEND:VEVENT\nEND:VCALENDAR\n",
        encoding="utf-8",
    )
    result = await listed(
        ctx(tmp_path), start=date(2026, 10, 20), end=date(2026, 11, 3)
    )
    assert "Cancelled" not in result
    if range_override:
        assert "Coverage incomplete" in result
    else:
        assert "2026-10-27" not in result and "Coverage incomplete" not in result
        assert result.count("Standup") == 2


async def test_one_occurrence_moves_without_changing_the_rest_of_the_series(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Standup",
                "start": "2026-10-20T09:00",
                "duration_minutes": 15,
                "location": "Office",
                "reminders": [10],
                "repeat": {"frequency": "weekly", "count": 3},
            },
        )
    )
    result = await update(
        context,
        uid,
        scope="occurrence",
        occurrence="2026-10-27T09:00",
        start="2026-10-28T11:00",
    )
    assert "Updated the occurrence" in result and ids(result) == [uid]
    assert "original occurrence 2026-10-27T09:00:00+01:00" in result
    lines = content_lines(tmp_path)
    assert lines.count("BEGIN:VEVENT") == 2 and lines.count(f"UID:{uid}") == 2
    assert "RECURRENCE-ID;TZID=Europe/Berlin:20261027T090000" in lines
    assert lines.count("TRIGGER:-PT10M") == 2
    result = await listed(context, start=date(2026, 10, 20), end=date(2026, 11, 3))
    assert "2026-10-20 09:00–09:15 Standup" in result
    assert "2026-11-03 09:00–09:15 Standup" in result
    assert "2026-10-28 11:00–11:15 Standup @ Office" in result
    assert "2026-10-27 09:00" not in result and "Coverage incomplete" not in result
    await update(
        context,
        uid,
        scope="occurrence",
        occurrence="2026-10-27T09:00",
        start="2026-10-28T12:00",
    )
    assert content_lines(tmp_path).count("BEGIN:VEVENT") == 2
    result = await listed(context, start=date(2026, 10, 28), end=date(2026, 10, 28))
    assert "12:00–12:15 Standup" in result and "11:00" not in result


async def test_an_invalid_occurrence_selector_keeps_the_series_unchanged(tmp_path):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Standup",
                "start": "2026-10-20T09:00",
                "repeat": {"frequency": "weekly", "count": 2},
            },
        )
    )
    before = (tmp_path / "calendar.ics").read_bytes()
    with pytest.raises(ToolError, match="no occurrence"):
        await update(
            context, uid, scope="occurrence", occurrence="2026-10-21T09:00", title="No"
        )
    assert (tmp_path / "calendar.ics").read_bytes() == before


@pytest.mark.parametrize(
    ("start", "changed"),
    [
        ("2026-10-20", {"start": "2026-10-21T09:00", "duration_minutes": 30}),
        ("2026-10-20T09:00", {"start": "2026-10-21", "end": "2026-10-21"}),
    ],
)
async def test_an_occurrence_keeps_the_recurrence_ids_date_type(
    tmp_path, start, changed
):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Series",
                "start": start,
                "repeat": {"frequency": "weekly", "count": 2},
            },
        )
    )
    before = (tmp_path / "calendar.ics").read_bytes()
    with pytest.raises(ToolError, match="all-day/timed form"):
        await update(context, uid, scope="occurrence", occurrence=start, **changed)
    assert (tmp_path / "calendar.ics").read_bytes() == before


async def test_whole_series_retiming_with_overrides_is_refused_without_data_loss(
    tmp_path,
):
    context = ctx(tmp_path)
    (uid,) = ids(
        await add(
            context,
            {
                "title": "Standup",
                "start": "2026-10-20T09:00",
                "repeat": {"frequency": "weekly", "count": 2},
            },
        )
    )
    await update(
        context,
        uid,
        scope="occurrence",
        occurrence="2026-10-27T09:00",
        location="Remote",
    )
    before = (tmp_path / "calendar.ics").read_bytes()
    with pytest.raises(ToolError, match="cannot safely retime"):
        await update(context, uid, scope="series", start="2026-10-20T10:00")
    assert (tmp_path / "calendar.ics").read_bytes() == before
    await update(context, uid, scope="series", title="Renamed")
    assert content_lines(tmp_path).count("SUMMARY:Renamed") == 2


async def test_update_keeps_imported_calendar_properties_and_timezone(tmp_path):
    (tmp_path / "calendar.ics").write_text(FOREIGN, encoding="utf-8")
    context = ctx(tmp_path)
    await update(context, "utc", title="Renamed webinar", duration_minutes=30)
    lines = content_lines(tmp_path)
    for kept in (
        "X-WR-CALNAME:Home",
        "TZID:America/New_York",
        "SUMMARY:Buy stamps",
        "UID:month-end",
        "DESCRIPTION:not the event summary",
    ):
        assert kept in lines
    assert "DTSTART:20260210T100000Z" in lines and "DTEND:20260210T103000Z" in lines
    await update(context, "byday", scope="series", location="New gym")
    assert "RRULE:FREQ=WEEKLY;BYDAY=MO,WE" in content_lines(tmp_path)


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
