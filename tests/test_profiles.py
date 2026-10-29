"""Writable profiles initialized from immutable package templates."""

from __future__ import annotations

import pytest

from lingcore.__main__ import main
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.profiles import (
    PROFILE_TEMPLATE_FILES,
    default_profile_path,
    initialize_profile,
)


def test_initialize_profile_copies_only_declared_template_files(tmp_path):
    destination = tmp_path / "local-agent"
    initialized = initialize_profile("coding", destination=destination)

    assert initialized == destination
    assert {path.name for path in destination.iterdir()} == set(
        PROFILE_TEMPLATE_FILES["coding"]
    )
    assert not (destination / "sessions.db").exists()
    assert not (destination / "workspace").exists()
    assert AgentProfile.load(destination).name == "coding"


def test_every_template_file_is_packaged_into_the_wheel():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    included = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    for template, files in PROFILE_TEMPLATE_FILES.items():
        for relative in files:
            source = f"profiles/{template}/{relative}"
            assert included.get(source) == (
                f"lingcore/profile_templates/{template}/{relative}"
            ), source
            assert (root / source).is_file(), source


async def test_daily_ships_its_slash_commands(tmp_path, monkeypatch):
    from lingcore.agent import Agent
    from tests.fakes import FakeLLMClient

    monkeypatch.delenv("LINGCORE_TIMEZONE", raising=False)
    destination = initialize_profile("daily", destination=tmp_path / "daily")
    assert sorted(p.name for p in (destination / "commands").iterdir()) == [
        "brief.md",
        "compare.md",
        "event.md",
        "plan.md",
        "summarize.md",
        "today.md",
    ]
    profile = AgentProfile.load(destination)
    async with Agent.from_profile(profile, llm=FakeLLMClient([])) as agent:
        names = {command.qualified_name for command in agent.commands.commands}
        assert names == {
            "/brief",
            "/compare",
            "/event",
            "/plan",
            "/summarize",
            "/today",
        }
        expanded = agent.commands.resolve("/event dentist friday 3pm", reserved=())
        agenda = agent.commands.resolve("/today trip preparation", reserved=())
    assert expanded is not None
    assert "Add to my calendar: dentist friday 3pm" in expanded.text
    assert expanded.display_text == "/event dentist friday 3pm"
    assert agenda is not None
    assert "Optional focus: trip preparation" in agenda.text
    assert agenda.display_text == "/today trip preparation"


async def test_today_reads_the_requested_day_and_memory_through_the_agent(
    tmp_path, monkeypatch
):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from lingcore.agent import Agent
    from lingcore.events import Final
    from lingcore.io.cli import CLIFrontend
    from lingcore.message import ToolCall
    from tests.fakes import FakeLLMClient, ScriptedTurn

    monkeypatch.setenv("LINGCORE_TIMEZONE", "Europe/Berlin")
    destination = initialize_profile("daily", destination=tmp_path / "daily")
    profile = AgentProfile.load(destination)
    today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    tomorrow = today + timedelta(days=1)
    llm = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="calendar",
                        name="calendar_list",
                        arguments={
                            "start": today.isoformat(),
                            "end": today.isoformat(),
                        },
                    ),
                    ToolCall(
                        id="preferences", name="memory", arguments={"action": "read"}
                    ),
                ]
            ),
            ScriptedTurn(text="Here is today's agenda."),
        ]
    )
    async with Agent.from_profile(profile, llm=llm) as agent:
        add = agent.tools.get("calendar_add")
        await add.run(
            add.validate_args(
                {
                    "events": [
                        {"title": "Today appointment", "start": today.isoformat()},
                        {
                            "title": "Tomorrow appointment",
                            "start": tomorrow.isoformat(),
                        },
                    ]
                }
            ),
            agent.tool_ctx,
        )
        memory_path = destination / "memory.md"
        memory_path.write_text(
            "## preferences\nLeave time for lunch.\n", encoding="utf-8"
        )
        calendar_before = (destination / "calendar.ics").read_bytes()
        memory_before = memory_path.read_bytes()
        frontend = CLIFrontend()
        frontend.set_commands(agent.commands)

        async def typed_command():
            return "/today"

        monkeypatch.setattr(frontend, "_read_message", typed_command)
        incoming = await frontend.read_input()
        assert incoming is not None and incoming.display_text == "/today"
        events = [event async for event in agent.run(incoming)]
        assert isinstance(events[-1], Final)
        assert "Show me today's agenda" in llm.calls[0][-1].content
        assert "[clock] Now:" in llm.calls[0][-1].content
        results = "\n".join(m.content for m in llm.calls[-1] if m.role == "tool")
        assert "Today appointment" in results and "Tomorrow appointment" not in results
        assert "Leave time for lunch" in results
        assert (destination / "calendar.ics").read_bytes() == calendar_before
        assert memory_path.read_bytes() == memory_before


def test_initialize_profile_never_overwrites_existing_directory(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("mine", encoding="utf-8")

    with pytest.raises(ConfigError, match="already exists"):
        initialize_profile("coding", destination=destination)
    assert sentinel.read_text(encoding="utf-8") == "mine"


def test_default_profile_uses_writable_state_outside_a_checkout(tmp_path, monkeypatch):
    import lingcore.profiles as profiles_module

    monkeypatch.setattr(profiles_module, "_REPO_PROFILE_ROOT", tmp_path / "missing")
    path = default_profile_path({"LINGCORE_STATE_HOME": str(tmp_path / "state")})
    assert path == tmp_path / "state" / "profiles" / "coding"


def test_profile_cli_lists_and_initializes(tmp_path, capsys):
    assert main(["profile", "list"]) == 0
    listed = capsys.readouterr().out
    assert set(PROFILE_TEMPLATE_FILES) <= set(listed.split())

    destination = tmp_path / "frontend"
    assert (
        main(
            [
                "profile",
                "init",
                "frontend",
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    assert (destination / "config.yaml").is_file()
    assert "Initialized 'frontend'" in capsys.readouterr().out


def test_main_help_advertises_profile_management(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    assert "{doctor,telegram,profile,plugin,trace,eval}" in capsys.readouterr().out
