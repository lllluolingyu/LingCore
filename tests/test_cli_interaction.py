"""Cli tests: interaction."""

from __future__ import annotations

import pytest

from lingcore.agent import Agent
from lingcore.events import (
    Final,
    ToolCallStarted,
)
from lingcore.io.base import run_session
from lingcore.io.cli import (
    CLIFrontend,
    _parse_attachments,
)
from lingcore.message import ToolCall, UserInput
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.cli import ScriptedFrontend, _announce_shell, _profile


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
    assert cli._command("/hepl") is True  # a typo is reported, not sent
    out = cli.console.export_text()
    assert "/resume <id>" in out
    assert "no usage reported yet" in out
    assert "abcd1234" in out and "old chat" in out
    assert "unknown command /hepl" in out

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
    _announce_shell(cli, "uv run pytest -q", "ls | wc -l")
    await cli.confirm("uv run pytest -q")
    await cli.confirm("ls | wc -l")
    assert "uv run pytest -q" in prompts[0] and "this session" in prompts[0]
    assert "not available" in prompts[1]


async def test_cli_confirm_non_shell_request_is_a_plain_approval(monkeypatch):
    # A confirmation that is not one of this turn's run_shell commands (a
    # non-public fetch, a skill activation) gets a plain allow/deny prompt —
    # no shell framing and no "always allow" that would edit the shell list.
    from rich.console import Console

    opts: dict = {}
    cli = CLIFrontend(tool_options=opts)
    cli.console = Console(record=True, width=200)
    prompts: list[str] = []

    def fake_input(prompt, *a, **k):
        prompts.append(prompt)
        return "A"

    monkeypatch.setattr(cli.console, "input", fake_input)
    request = "Allow fetch_url to reach a non-public address? http://x/"
    # "A" is only an allow-once here: it never writes a shell pattern.
    assert await cli.confirm(request) is True
    out = cli.console.export_text()
    assert "approve?" in out and request in out
    assert "run shell command" not in out and "$ Allow" not in out
    assert "always allow" not in prompts[0]
    assert "run_shell" not in opts
    monkeypatch.setattr(cli.console, "input", lambda *a, **k: "n")
    assert await cli.confirm(request) is False


async def test_cli_shell_commands_are_scoped_to_the_turn(monkeypatch):
    cli = CLIFrontend()
    cli.console.quiet = True
    _announce_shell(cli, "make")
    assert "make" in cli._shell_commands
    cli._start_turn()
    assert cli._shell_commands == set()


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
