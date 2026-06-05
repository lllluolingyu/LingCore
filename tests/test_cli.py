"""Tests for the frontend boundary and CLI adapter (M5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.events import Error, Final, TextDelta, ToolCallStarted, ToolResultEvent
from lingcore.io.base import run_session
from lingcore.io.cli import CLIFrontend
from lingcore.message import ToolCall, ToolResult
from tests.fakes import FakeLLMClient, ScriptedTurn

PROFILE = """
name: smoke
workspace: ${SMOKE_WS:-.}
llm:
  model: test-model
  base_url: http://localhost:11434/v1
persona:
  system_prompt: "You are a smoke-test agent."
tools:
  - read_file
  - run_shell
loop:
  max_iters: 5
"""


class ScriptedFrontend:
    """A Frontend that replays canned inputs and records rendered events."""

    def __init__(self, inputs: list[str], confirm_answer: bool = True):
        self._inputs = list(inputs)
        self.confirm_answer = confirm_answer
        self.events: list = []
        self.confirmed: list[str] = []

    async def read_input(self) -> str | None:
        if self._inputs:
            return self._inputs.pop(0)
        return None

    def render(self, event) -> None:
        self.events.append(event)

    async def confirm(self, command: str) -> bool:
        self.confirmed.append(command)
        return self.confirm_answer


def _profile(tmp_path: Path) -> AgentProfile:
    p = tmp_path / "p.yaml"
    p.write_text(PROFILE, encoding="utf-8")
    return AgentProfile.load(p)


async def test_run_session_drives_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_WS", str(tmp_path))
    (tmp_path / "a.txt").write_text("contents", encoding="utf-8")
    prof = _profile(tmp_path)

    llm = FakeLLMClient([ScriptedTurn(text="Hello from the agent.")])
    agent = Agent.from_profile(prof, llm=llm, base_dir=tmp_path)

    frontend = ScriptedFrontend(inputs=["hi"])
    await run_session(agent, frontend)

    assert any(isinstance(e, TextDelta) for e in frontend.events)
    assert isinstance(frontend.events[-1], Final)


async def test_run_session_ends_on_none(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_WS", str(tmp_path))
    prof = _profile(tmp_path)
    llm = FakeLLMClient([ScriptedTurn(text="unused")])
    agent = Agent.from_profile(prof, llm=llm, base_dir=tmp_path)

    frontend = ScriptedFrontend(inputs=[])  # immediately ends
    await run_session(agent, frontend)
    assert frontend.events == []


async def test_session_shell_confirm_flows_to_frontend(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_WS", str(tmp_path))
    prof = _profile(tmp_path)

    call = ToolCall(id="c1", name="run_shell", arguments={"command": "echo hi"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="done"),
    ])
    frontend = ScriptedFrontend(inputs=["run echo"], confirm_answer=True)
    agent = Agent.from_profile(prof, confirm=frontend.confirm, llm=llm, base_dir=tmp_path)

    await run_session(agent, frontend)

    # The shell command's confirmation was routed to the frontend...
    assert frontend.confirmed == ["echo hi"]
    # ...and the tool actually ran (echoed output came back ok).
    results = [e for e in frontend.events if isinstance(e, ToolResultEvent)]
    assert results[0].result.ok is True
    assert "hi" in results[0].result.content


async def test_session_shell_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_WS", str(tmp_path))
    prof = _profile(tmp_path)

    call = ToolCall(id="c1", name="run_shell", arguments={"command": "rm -rf /"})
    llm = FakeLLMClient([
        ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
        ScriptedTurn(text="ok, skipped"),
    ])
    frontend = ScriptedFrontend(inputs=["do something scary"], confirm_answer=False)
    agent = Agent.from_profile(prof, confirm=frontend.confirm, llm=llm, base_dir=tmp_path)

    await run_session(agent, frontend)

    result = [e for e in frontend.events if isinstance(e, ToolResultEvent)][0].result
    assert result.ok is False
    assert "declined" in result.content


# --- CLI adapter rendering (no real terminal) -----------------------------


def test_cli_renders_all_event_types_without_error():
    cli = CLIFrontend(agent_name="t")
    cli.console.quiet = True  # swallow output
    cli.render(TextDelta("hello "))
    cli.render(TextDelta("world"))
    cli.render(ToolCallStarted(ToolCall(id="c", name="read_file", arguments={"path": "a"})))
    cli.render(ToolResultEvent(ToolResult(call_id="c", name="read_file", content="data")))
    cli.render(Final("hello world"))
    cli.render(Error("something broke"))
    # If we got here, all event branches rendered without raising.


async def test_cli_confirm_allow_once(monkeypatch):
    cli = CLIFrontend()
    cli.console.quiet = True
    for token in ("a", "y", "yes", ""):  # all mean allow once
        monkeypatch.setattr(cli.console, "input", lambda *a, t=token, **k: t)
        assert await cli.confirm("echo hi") is True
    # "allow once" must NOT persist anything to the allowlist.
    assert cli._tool_options.get("run_shell", {}).get("allow_patterns", []) == []


async def test_cli_confirm_deny(monkeypatch):
    cli = CLIFrontend()
    cli.console.quiet = True
    for token in ("d", "n", "no", "x"):
        monkeypatch.setattr(cli.console, "input", lambda *a, t=token, **k: t)
        assert await cli.confirm("echo hi") is False


async def test_cli_confirm_allow_always_persists(monkeypatch):
    opts: dict = {}
    cli = CLIFrontend(tool_options=opts)
    cli.console.quiet = True
    monkeypatch.setattr(cli.console, "input", lambda *a, **k: "A")
    assert await cli.confirm("pytest -q") is True
    # The command's first token is appended to the shared options dict, so the
    # agent's ToolContext (which holds the same dict) auto-approves it next time.
    assert opts["run_shell"]["allow_patterns"] == ["pytest"]
    # A second matching command would not even reach confirm; but a re-prompt of
    # the same prefix should not duplicate the entry.
    await cli.confirm("pytest tests/test_x.py")
    assert opts["run_shell"]["allow_patterns"] == ["pytest"]
