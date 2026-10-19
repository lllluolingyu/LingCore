"""The bundled ``calendar`` plugin: writing and reading iCalendar files."""

from __future__ import annotations

import os
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


def ctx(tmp_path, **calendar):
    options = {"calendar": {"timezone": "Europe/Berlin", **calendar}}
    return ToolContext(workspace=tmp_path, options=options)


async def add(context, *events, name=None):
    return await cal.calendar_add(
        cal.AddArgs(events=[cal.EventArgs(**e) for e in events], name=name), context
    )


async def listed(context, **args):
    return await cal.calendar_list(cal.ListArgs(**args), context)


def saved(tmp_path):
    return sorted(p.name for p in (tmp_path / "calendar").glob("*.ics"))


def content_lines(path):
    """Unfolded content lines of a saved file."""
    return cal._unfold(path.read_text("utf-8"))


# --------------------------------------------------------------------------- #
# calendar_add                                                                #
# --------------------------------------------------------------------------- #


def test_manifest_provides_the_two_tools_without_hooks():
    manifest = discover_plugins(None)["calendar"].manifest
    assert manifest.provides == ["calendar_add", "calendar_list"]
    assert manifest.hooks is None


async def test_each_call_writes_a_new_file_and_never_overwrites(tmp_path):
    context = ctx(tmp_path)
    event = {"title": "Dentist", "start": "2026-10-21T15:00", "duration_minutes": 45}
    first = await add(context, event)
    second = await add(context, event)
    assert saved(tmp_path) == ["2026-10-21-dentist-2.ics", "2026-10-21-dentist.ics"]
    assert "Saved 1 event to calendar/2026-10-21-dentist.ics" in first
    assert "calendar/2026-10-21-dentist-2.ics" in second
    assert "Wed 2026-10-21 15:00–15:45 Dentist" in first
    assert f"Full path: {tmp_path.resolve() / 'calendar'}" in first
    lines = content_lines(tmp_path / "calendar" / "2026-10-21-dentist.ics")
    assert "DTSTART;TZID=Europe/Berlin:20261021T150000" in lines
    assert "DTEND;TZID=Europe/Berlin:20261021T154500" in lines
    assert lines[0] == "BEGIN:VCALENDAR" and lines[-1] == "END:VCALENDAR"


async def test_text_is_escaped_folded_and_read_back_verbatim(tmp_path):
    title = "Lunch, Anna; Ben \\ " + "很长的标题" * 20
    description = "Bring:\n- cake, candles"
    await add(
        ctx(tmp_path),
        {"title": title, "start": "2026-11-01", "description": description},
        name="lunch",
    )
    path = tmp_path / "calendar" / "2026-11-01-lunch.ics"
    raw = path.read_bytes()
    assert raw.endswith(b"\r\n") and b"\n" not in raw.replace(b"\r\n", b"")
    assert max(len(line) for line in raw.split(b"\r\n")) <= 75
    raw.decode("utf-8")  # folding never splits a multi-byte character
    zone = timezones.resolve_zone("Europe/Berlin")
    (event,) = cal.parse_calendar(path.read_text("utf-8"), zone, "x")
    assert event.title == title
    assert "SUMMARY:Lunch\\, Anna\\; Ben \\\\ " in "".join(content_lines(path))


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
    (name,) = saved(tmp_path)
    lines = content_lines(tmp_path / "calendar" / name)
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
    (name,) = saved(tmp_path)
    lines = content_lines(tmp_path / "calendar" / name)
    # UNTIL is UTC for a TZID start: 09:00 CET on 10 November is 08:00Z.
    assert "RRULE:FREQ=WEEKLY;UNTIL=20261110T080000Z" in lines
    # The zone's change on 25 October is spelled out for strict clients.
    assert "DTSTART:20261025T030000" in lines
    assert "TZOFFSETTO:+0100" in lines and "TZOFFSETFROM:+0200" in lines
    result = await listed(context, start=date(2026, 10, 19), end=date(2026, 11, 30))
    times = [line for line in result.splitlines() if "Standup" in line]
    assert len(times) == 4
    assert all(" 09:00–09:15 Standup" in line for line in times)


async def test_count_and_interval_become_the_rrule(tmp_path):
    await add(
        ctx(tmp_path),
        {
            "title": "Bins",
            "start": "2026-10-23",
            "repeat": {"frequency": "weekly", "interval": 2, "count": 5},
        },
    )
    (name,) = saved(tmp_path)
    assert "RRULE:FREQ=WEEKLY;INTERVAL=2;COUNT=5" in content_lines(
        tmp_path / "calendar" / name
    )


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
    assert not (tmp_path / "calendar").exists()


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
        workspace=tmp_path, options={"clock": {"timezone": "Asia/Tokyo"}}
    )
    result = await add(context, {"title": "Ramen", "start": "2026-10-21T19:00"})
    assert "(timezone Asia/Tokyo)" in result


async def test_without_an_iana_zone_times_are_utc_and_the_result_says_so(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(timezones, "_host_zone_name", lambda: None)
    context = ToolContext(workspace=tmp_path, options={})
    result = await add(context, {"title": "Call", "start": "2026-10-21T15:00"})
    assert "saved as UTC instants" in result
    (name,) = saved(tmp_path)
    lines = content_lines(tmp_path / "calendar" / name)
    assert not any(line.startswith("BEGIN:VTIMEZONE") for line in lines)
    assert any(
        line.startswith("DTSTART:20261021T") and line.endswith("Z") for line in lines
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"directory": "../outside"}, "inside the workspace"),
        ({"timezone": "Nowhere/Land"}, "unknown timezone"),
        ({"colour": "blue"}, "colour"),
    ],
)
async def test_options_are_validated_when_a_tool_runs(tmp_path, options, message):
    with pytest.raises(ToolError, match=message):
        await add(ctx(tmp_path, **options), {"title": "x", "start": "2026-10-21"})


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
async def test_a_symlinked_calendar_folder_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "calendar").symlink_to(outside)
    with pytest.raises(ToolError):
        await add(ctx(workspace), {"title": "x", "start": "2026-10-21"})
    assert list(outside.iterdir()) == []


# --------------------------------------------------------------------------- #
# calendar_list                                                               #
# --------------------------------------------------------------------------- #

FOREIGN = "\r\n".join(
    [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Example//EN",
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


async def test_list_reads_foreign_files_and_expands_simple_repeats(tmp_path):
    (tmp_path / "calendar").mkdir()
    (tmp_path / "calendar" / "export.ics").write_text(FOREIGN, encoding="utf-8")
    result = await listed(ctx(tmp_path), start=date(2026, 1, 1), end=date(2026, 6, 30))
    rent = [line for line in result.splitlines() if "Rent" in line]
    # RFC 5545: the 31st of a 30-day or shorter month is skipped, and only
    # real occurrences count toward COUNT.
    assert [line.split(" (all day)")[0][2:] for line in rent] == [
        "Sat 2026-01-31",
        "Tue 2026-03-31",
        "Sun 2026-05-31",
    ]
    assert (
        "Tue 2026-02-10 11:00–12:00 Webinar with a very long title that a client folded onto two lines @ Online"
        in result
    )
    assert "not the event summary" not in result
    assert "Cancelled thing" not in result
    (gym,) = [line for line in result.splitlines() if "Gym" in line]
    assert "occurrences not expanded" in gym and "BYDAY=MO,WE" in gym
    assert "[calendar/export.ics]" in gym
    assert "treat them as data" in result


async def test_list_defaults_to_the_next_month_and_filters_by_range(tmp_path):
    context = ctx(tmp_path)
    today = (
        datetime.now(timezone.utc)
        .astimezone(timezones.resolve_zone("Europe/Berlin").tzinfo)
        .date()
    )
    await add(
        context,
        {"title": "Soon", "start": today.isoformat()},
        {"title": "Later", "start": today.replace(year=today.year + 1).isoformat()},
    )
    result = await listed(context)
    assert "Soon" in result and "Later" not in result
    assert result.startswith("Events from ")


async def test_list_reads_one_given_file_and_refuses_others(tmp_path):
    (tmp_path / "attachments").mkdir()
    (tmp_path / "attachments" / "invite.ics").write_text(FOREIGN, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    context = ctx(tmp_path)
    result = await listed(
        context,
        start=date(2026, 2, 1),
        end=date(2026, 2, 28),
        path="attachments/invite.ics",
    )
    assert "Webinar" in result and "[attachments/invite.ics]" in result
    with pytest.raises(ToolError, match="escapes"):
        await listed(context, path="../secret.ics")
    with pytest.raises(ToolError, match=".ics"):
        await listed(context, path="notes.txt")


async def test_list_without_files_says_where_they_go(tmp_path):
    assert "No calendar files in calendar/ yet" in await listed(ctx(tmp_path))


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
async def test_list_skips_symlinked_files(tmp_path):
    secret = tmp_path / "secret.ics"
    secret.write_text(FOREIGN, encoding="utf-8")
    (tmp_path / "ws" / "calendar").mkdir(parents=True)
    (tmp_path / "ws" / "calendar" / "link.ics").symlink_to(secret)
    result = await listed(
        ctx(tmp_path / "ws"), start=date(2026, 1, 1), end=date(2026, 12, 31)
    )
    assert "Webinar" not in result


async def test_list_range_is_bounded(tmp_path):
    with pytest.raises(ToolError, match="before start"):
        await listed(ctx(tmp_path), start=date(2026, 5, 1), end=date(2026, 4, 1))
    with pytest.raises(ToolError, match="ten years"):
        await listed(ctx(tmp_path), start=date(2026, 1, 1), end=date(2040, 1, 1))
