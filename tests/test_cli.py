"""Tests for the frontend boundary and CLI adapter (M5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.events import (
    Error,
    Final,
    StreamRetry,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.io.base import run_session
from lingcore.io.cli import CLIFrontend, _parse_attachments
from lingcore.message import ToolCall, ToolResult, UserInput
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


async def test_run_session_closes_turn_when_render_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_WS", str(tmp_path))
    prof = _profile(tmp_path)
    llm = FakeLLMClient(
        [
            ScriptedTurn(text="discarded"),
            ScriptedTurn(text="recovered"),
        ]
    )
    agent = Agent.from_profile(prof, llm=llm, base_dir=tmp_path)

    class FailingFrontend(ScriptedFrontend):
        def __init__(self) -> None:
            super().__init__(["first", "second"])
            self._fail_render = True

        def render(self, event) -> None:
            if self._fail_render:
                self._fail_render = False
                raise RuntimeError("renderer failed")
            super().render(event)

    frontend = FailingFrontend()
    with pytest.raises(RuntimeError, match="renderer failed"):
        await run_session(agent, frontend)

    # Cleanup is complete before the exception escapes; immediate reuse cannot
    # observe the abandoned turn lease. The accepted first user row is kept.
    assert agent._turn_checkpoint is None
    assert [message.content for message in agent.memory.messages] == ["first"]

    await run_session(agent, frontend)
    assert isinstance(frontend.events[-1], Final)
    assert frontend.events[-1].content == "recovered"


async def test_session_shell_confirm_flows_to_frontend(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_WS", str(tmp_path))
    prof = _profile(tmp_path)

    call = ToolCall(id="c1", name="run_shell", arguments={"command": "echo hi"})
    llm = FakeLLMClient(
        [
            ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
            ScriptedTurn(text="done"),
        ]
    )
    frontend = ScriptedFrontend(inputs=["run echo"], confirm_answer=True)
    agent = Agent.from_profile(
        prof, confirm=frontend.confirm, llm=llm, base_dir=tmp_path
    )

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
    llm = FakeLLMClient(
        [
            ScriptedTurn(tool_calls=[call], finish_reason="tool_calls"),
            ScriptedTurn(text="ok, skipped"),
        ]
    )
    frontend = ScriptedFrontend(inputs=["do something scary"], confirm_answer=False)
    agent = Agent.from_profile(
        prof, confirm=frontend.confirm, llm=llm, base_dir=tmp_path
    )

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
    cli.render(
        ToolCallStarted(ToolCall(id="c", name="read_file", arguments={"path": "a"}))
    )
    cli.render(
        ToolResultEvent(ToolResult(call_id="c", name="read_file", content="data"))
    )
    cli.render(
        StreamRetry(
            attempt=1,
            max_attempts=3,
            reason="stream interrupted: [boom]",
            discarded_chars=11,
        )
    )
    cli.render(Final("hello world"))
    cli.render(Error("something broke"))
    # If we got here, all event branches rendered without raising.


def test_cli_escapes_model_controlled_markup():
    # Model/tool text must be escaped, not interpreted as Rich markup — otherwise
    # a model could spoof terminal styling, and an unbalanced tag would raise.
    from rich.console import Console

    cli = CLIFrontend(agent_name="t")
    cli.console = Console(record=True, width=240)
    cli.render(TextDelta("[bold red]spoof[/] and [unbalanced"))
    cli.render(
        ToolResultEvent(ToolResult(call_id="c", name="tool", content="[link=x]y"))
    )
    out = cli.console.export_text()
    assert "[bold red]spoof[/]" in out  # rendered literally, not as styling
    assert "[unbalanced" in out
    assert "[link=x]y" in out


async def test_cli_confirm_allow_once(monkeypatch):
    cli = CLIFrontend()
    cli.console.quiet = True
    for token in ("a", "y", "yes", ""):  # all mean allow once
        monkeypatch.setattr(cli.console, "input", lambda *a, t=token, **k: t)
        assert await cli.confirm("echo hi") is True
    # "allow once" must NOT persist anything to the allowlist.
    assert cli._tool_options.get("run_shell", {}).get("allow_patterns", []) == []


def test_cli_resume_replay_escapes_stored_tool_names():
    # Session replay renders *persisted* model output: a tool call whose name
    # carries Rich markup must render literally, exactly like live events.
    from datetime import datetime, timezone

    from rich.console import Console

    from lingcore.message import Message
    from lingcore.sessions import SessionMeta

    cli = CLIFrontend(agent_name="t")
    cli.console = Console(record=True, width=240)
    now = datetime.now(timezone.utc)
    meta = SessionMeta(id="abcd1234", title="t", created_at=now, updated_at=now)
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c", name="[bold red]spoof[/]", arguments={})],
        ),
        Message(role="tool", name="[link=x]y", content="out", tool_call_id="c"),
    ]
    cli.show_resume(meta, messages)
    out = cli.console.export_text()
    assert "[bold red]spoof[/]" in out  # literal, not styled
    assert "[link=x]y" in out


def test_cli_resume_labels_compaction_summary_as_derived_state():
    from datetime import datetime, timezone

    from rich.console import Console

    from lingcore.message import Message
    from lingcore.sessions import SessionMeta

    cli = CLIFrontend(agent_name="t")
    cli.console = Console(record=True, width=240)
    now = datetime.now(timezone.utc)
    meta = SessionMeta(id="abcd1234", title="t", created_at=now, updated_at=now)

    cli.show_resume(
        meta,
        [Message(role="user", name="summary", content="earlier facts")],
    )

    out = cli.console.export_text()
    assert "earlier summary › earlier facts" in out
    assert "you › earlier facts" not in out


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
    # The exact command token prefix is appended to the shared options dict, so
    # approval does not expand to every command sharing the first executable.
    assert opts["run_shell"]["allow_patterns"] == ["pytest -q"]
    # A second matching command would not even reach confirm; but a re-prompt of
    # the same prefix should not duplicate the entry.
    await cli.confirm("pytest -q")
    assert opts["run_shell"]["allow_patterns"] == ["pytest -q"]


async def test_cli_confirm_allow_always_skips_shell_control(monkeypatch):
    opts: dict = {}
    cli = CLIFrontend(tool_options=opts)
    cli.console.quiet = True
    monkeypatch.setattr(cli.console, "input", lambda *a, **k: "A")
    for command in ("ls; echo unsafe", "printf approved & printf chained"):
        assert await cli.confirm(command) is True
    assert opts["run_shell"]["allow_patterns"] == []


async def test_user_input_frontend_drives_agent(tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nrest")
    prof = _profile(tmp_path)
    llm = FakeLLMClient([ScriptedTurn(text="I see it")])
    agent = Agent.from_profile(prof, llm=llm, base_dir=tmp_path)

    from lingcore.media import attachment_from_path

    att = attachment_from_path(tmp_path / "pic.png")
    incoming = UserInput(text="describe", attachments=[att])

    class AttachFrontend:
        def __init__(self):
            self.sent = False

        async def read_input(self):
            if self.sent:
                return None
            self.sent = True
            return incoming

        def render(self, event):
            pass

        async def confirm(self, command):
            return True

    frontend = AttachFrontend()
    from lingcore.io.base import run_session

    await run_session(agent, frontend)
    assert agent.memory.messages[0].attachments


# --- @path attachment parsing ---------------------------------------------


def test_parse_attachments_any_file_type(tmp_path):
    (tmp_path / "notes.md").write_text("# hi", encoding="utf-8")
    ui, warnings = _parse_attachments("see @notes.md please", base=tmp_path)
    assert warnings == []
    assert len(ui.attachments) == 1
    assert ui.attachments[0].kind == "text"
    assert ui.text == "see notes.md please"  # the @ marker is stripped


def test_parse_attachments_missing_path_stays_literal_with_warning(tmp_path):
    ui, warnings = _parse_attachments("look at @missing.png", base=tmp_path)
    assert ui.attachments == []
    assert "@missing.png" in ui.text  # the typed token is preserved verbatim
    assert any("no such file" in w for w in warnings)


def test_parse_attachments_bare_mention_is_silent(tmp_path):
    ui, warnings = _parse_attachments("ping @alice about it", base=tmp_path)
    assert ui.attachments == []
    assert "@alice" in ui.text
    assert warnings == []  # a non-path @mention must not nag


def test_parse_attachments_quoted_path_with_spaces(tmp_path):
    (tmp_path / "my file.txt").write_text("data", encoding="utf-8")
    ui, warnings = _parse_attachments('read @"my file.txt"', base=tmp_path)
    assert len(ui.attachments) == 1
    assert ui.attachments[0].name == "my file.txt"
    assert ui.text == "read my file.txt"


# --- Stop, slash commands, diff previews, usage ----------------------------


async def test_run_session_stop_cancels_only_the_turn(tmp_path, monkeypatch):
    import asyncio
    from contextlib import contextmanager

    from lingcore.events import TurnCancelled

    monkeypatch.setenv("SMOKE_WS", str(tmp_path))
    prof = _profile(tmp_path)
    llm = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", name="run_shell", arguments={"command": "ls"})
                ],
                finish_reason="tool_calls",
            ),
            ScriptedTurn(text="second answer"),
        ]
    )

    class StoppingFrontend(ScriptedFrontend):
        stop = None

        @contextmanager
        def interrupt_scope(self, stop):
            self.stop = stop
            yield

        async def confirm(self, command: str) -> bool:
            # The user presses Stop while the confirmation is pending.
            assert self.stop is not None and self.stop() is True
            await asyncio.sleep(10)
            return True

    frontend = StoppingFrontend(["first", "second"])
    agent = Agent.from_profile(
        prof, llm=llm, base_dir=tmp_path, confirm=frontend.confirm
    )

    await run_session(agent, frontend)

    kinds = [type(e).__name__ for e in frontend.events]
    assert "TurnCancelled" in kinds
    assert isinstance(frontend.events[kinds.index("TurnCancelled")], TurnCancelled)
    assert isinstance(frontend.events[-1], Final)
    assert frontend.events[-1].content == "second answer"
    # The stopped turn kept its user message but none of its tool state.
    roles = [(m.role, m.content) for m in agent.memory.messages]
    assert roles[0] == ("user", "first")
    assert all(m.role != "tool" for m in agent.memory.messages)


def test_cli_slash_commands(monkeypatch):
    from datetime import datetime, timezone

    from rich.console import Console

    from lingcore.sessions import SessionMeta

    now = datetime.now(timezone.utc)
    meta = SessionMeta(
        id="abcd1234ef", title="old chat", created_at=now, updated_at=now
    )

    class Store:
        def list(self):
            return [meta]

        def resolve_prefix(self, prefix):
            assert prefix == "abcd"
            return meta

    cli = CLIFrontend(agent_name="t", store=Store())
    cli.console = Console(record=True, width=200)
    assert cli._command("/help") is True
    assert cli._command("/usage") is True
    assert cli._command("/sessions") is True
    assert cli._command("/etc/hosts is broken") is False  # not a command
    out = cli.console.export_text()
    assert "/resume <id>" in out
    assert "no usage reported yet" in out
    assert "abcd1234" in out and "old chat" in out

    assert cli._command("/new") is None
    switch = cli.take_session_switch()
    assert switch is not None and switch.session_id is None
    assert cli.take_session_switch() is None
    assert cli._command("/resume abcd") is None
    switch = cli.take_session_switch()
    assert switch is not None and switch.session_id == "abcd1234ef"


def test_cli_renders_edit_diff_and_usage_footer():
    from rich.console import Console

    from lingcore.events import UsageReported
    from lingcore.usage import TokenUsage

    cli = CLIFrontend(agent_name="t")
    cli.console = Console(record=True, width=200)
    cli.render(
        ToolCallStarted(
            ToolCall(
                id="c",
                name="edit_file",
                arguments={"path": "src/a.py", "old": "x = 1\n", "new": "x = 2\n"},
            )
        )
    )
    cli.render(
        ToolCallStarted(
            ToolCall(
                id="d",
                name="patch_file",
                arguments={
                    "path": "b.py",
                    "diff": "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-old\n+[new]\n",
                },
            )
        )
    )
    usage = TokenUsage(
        model="m", input_tokens=12_300, cached_input_tokens=10_000, output_tokens=800
    )
    cli.render(UsageReported(usage))
    cli.render(UsageReported(usage))
    cli.render(Final("done"))
    out = cli.console.export_text()
    assert "edit_file src/a.py" in out
    assert "-x = 1" in out and "+x = 2" in out
    assert "+[new]" in out  # diff text is escaped, never parsed as markup
    assert "+++ b/b.py" not in out
    assert "2 requests · in 24.6k (20.0k cached) · out 1.6k" in out
    cli.render(Final("again"))  # the footer resets per turn
    assert "↳" not in cli.console.export_text()  # export_text() cleared the buffer
    assert cli._session_usage.requests == 2


async def test_cli_confirm_shows_session_pattern(monkeypatch):
    from rich.console import Console

    cli = CLIFrontend()
    cli.console = Console(record=True, width=200)
    prompts: list[str] = []

    def fake_input(prompt, *a, **k):
        prompts.append(prompt)
        return "d"

    monkeypatch.setattr(cli.console, "input", fake_input)
    await cli.confirm("uv run pytest -q")
    await cli.confirm("ls | wc -l")
    assert "uv run pytest -q" in prompts[0] and "this session" in prompts[0]
    assert "not available" in prompts[1]


async def test_cli_stale_prompt_is_consumed_before_next_input(monkeypatch):
    import asyncio
    import threading

    cli = CLIFrontend()
    cli.console.quiet = True
    release = threading.Event()
    answers = iter(["stale answer", "real message"])

    def fake_input(prompt, *a, **k):
        value = next(answers)
        if value == "stale answer":
            release.wait(5)
        return value

    monkeypatch.setattr(cli.console, "input", fake_input)
    pending = asyncio.create_task(cli.confirm("rm -rf build"))
    await asyncio.sleep(0.05)
    pending.cancel()  # Stop while the confirmation prompt is open
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cli._stale_input is not None
    release.set()
    # The abandoned prompt's line is discarded; the next line is the message.
    assert await cli.read_input() == "real message"
