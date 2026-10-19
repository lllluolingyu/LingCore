"""The bundled ``clock`` plugin and the shared timezone resolution."""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import pytest

import lingcore.timezones as timezones
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.events import Final
from lingcore.plugins import PluginContext, UserMessageEvent
from lingcore.plugins.discovery import discover_plugins
from lingcore.profiles import initialize_profile
from lingcore.skills import _load_tool_module
from tests.fakes import FakeLLMClient, ScriptedTurn


def _load_clock():
    plugin = discover_plugins(None)["clock"]
    assert plugin.manifest.module is not None
    _load_tool_module("clock", plugin.root / plugin.manifest.module, [], prefix="clock")
    name = next(n for n in sys.modules if n.startswith("lingcore_skill_tools.clock."))
    return sys.modules[name]


clock = _load_clock()


def hooks(options, now):
    ctx = PluginContext("clock", None, None, None, None, options, lambda n, d=None: d)
    return clock.ClockHooks(ctx, now=lambda: now)


def test_manifest_is_hooks_only_and_never_blocks():
    manifest = discover_plugins(None)["clock"].manifest
    assert manifest.hooks == "ClockHooks" and manifest.provides == []
    assert manifest.on_hook_error == "ignore"


async def test_each_message_gets_the_local_date_time_and_zone():
    now = datetime(2026, 10, 19, 14, 32, 7, tzinfo=timezone.utc)
    decision = await hooks({"timezone": "Asia/Shanghai"}, now).user_message(
        UserMessageEvent("hi", "hi")
    )
    assert decision.action == "add_context"
    assert decision.context == (
        "[clock] Now: Monday 2026-10-19 22:32 Asia/Shanghai (UTC+08:00)"
    )


@pytest.mark.parametrize(
    ("utc", "expected"),
    [
        (
            datetime(2026, 10, 24, 12, 0),
            "Saturday 2026-10-24 14:00 Europe/Berlin (UTC+02:00)",
        ),
        (
            datetime(2026, 10, 26, 12, 0),
            "Monday 2026-10-26 13:00 Europe/Berlin (UTC+01:00)",
        ),
    ],
)
def test_daylight_saving_changes_the_offset(utc, expected):
    zone = timezones.resolve_zone("Europe/Berlin")
    assert clock.clock_line(utc.replace(tzinfo=timezone.utc), zone).endswith(expected)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"timezone": "Mars/Olympus_Mons"}, "unknown timezone"),
        ({"tz": "Asia/Tokyo"}, "tz: Extra inputs"),
    ],
)
def test_bad_options_fail_when_the_agent_is_built(options, message):
    with pytest.raises(ConfigError, match=message):
        hooks(options, datetime.now(timezone.utc))


def test_blank_timezone_uses_the_host_zone(monkeypatch):
    monkeypatch.setenv("TZ", "America/Chicago")
    zone = timezones.resolve_zone("  ")
    assert zone.iana and zone.name == "America/Chicago"


def test_a_host_without_an_iana_zone_falls_back_to_its_offset(monkeypatch):
    monkeypatch.setattr(timezones, "_host_zone_name", lambda: None)
    zone = timezones.resolve_zone(None)
    assert not zone.iana and zone.name.startswith("UTC")
    line = clock.clock_line(datetime.now(timezone.utc), zone)
    assert line.endswith(zone.name) and "(" not in line


def test_offsets_format_with_seconds_only_when_needed():
    from datetime import timedelta

    assert timezones.format_offset(timedelta(hours=-3, minutes=-30)) == "UTC-03:30"
    assert timezones.format_offset(timedelta(seconds=3208)) == "UTC+00:53:28"
    assert timezones.format_offset(None) == "UTC+00:00"


async def test_daily_sends_the_clock_with_the_message_not_the_system_prompt(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LINGCORE_TIMEZONE", "Asia/Tokyo")
    monkeypatch.delenv("LINGCORE_SEARCH_PROVIDER", raising=False)
    profile = AgentProfile.load(
        initialize_profile("daily", destination=tmp_path / "daily")
    )
    assert profile.plugins == ["search", "clock"]
    fake = FakeLLMClient([ScriptedTurn(text="one"), ScriptedTurn(text="two")])
    async with Agent.from_profile(profile, llm=fake) as agent:
        for text in ("hello", "again"):
            events = [event async for event in agent.run(text)]
            assert isinstance(events[-1], Final)
    first, second = fake.calls
    system = first[0]
    assert system.role == "system" and "Asia/Tokyo" not in system.content
    user = first[-1]
    assert user.content.startswith("hello\n[clock] Now: ")
    assert user.content.endswith(" Asia/Tokyo (UTC+09:00)")
    # The stored input stays what the user typed; history keeps each time.
    assert user.input_text == "hello"
    assert second[0].content == system.content
    assert [m.content for m in second[: len(first)]] == [m.content for m in first]
