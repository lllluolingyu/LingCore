"""Cli tests: editor."""

from __future__ import annotations

import pytest

from lingcore.events import (
    Final,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.io.cli import (
    CLIFrontend,
    _completions,
    _stable_prefix,
)
from lingcore.message import ToolCall, ToolResult
from tests.support.cli import _announce_shell, _prompt_with_keys, _recording_cli


def test_stable_prefix_commits_only_complete_markdown_blocks():
    assert _stable_prefix("one paragraph still streaming") == 0
    assert _stable_prefix("done.\n\nnext") == len("done.\n\n")
    # A blank line inside an open fence is not a block boundary...
    fenced = "```py\nx = 1\n\ny = 2\n"
    assert _stable_prefix(fenced) == 0
    # ...but the closing fence is.
    closed = fenced + "```\n"
    assert _stable_prefix(closed + "tail") == len(closed)
    # A shorter or different marker does not close a fence.
    assert _stable_prefix("````\ncode\n```\n\n") == 0
    assert _stable_prefix("~~~\ncode\n```\n\n") == 0


def test_cli_streams_reply_as_rendered_markdown_once():
    cli = _recording_cli()
    reply = "# Title\n\nSome **bold** and `code`.\n\n- a\n- b\n"
    for i in range(0, len(reply), 3):  # token-sized chunks
        cli.render(TextDelta(reply[i : i + 3]))
    cli.render(Final(reply))
    out = cli.console.export_text()
    assert "**" not in out and "`" not in out and "# Title" not in out
    assert out.count("Title") == 1 and out.count("bold") == 1
    assert "●" in out  # reply marker


def test_cli_markdown_links_show_their_target():
    cli = _recording_cli()
    cli.render(TextDelta("see [the docs](https://example.com/real)"))
    cli.render(Final(""))
    assert "https://example.com/real" in cli.console.export_text()


def test_cli_cancelled_turn_discards_uncommitted_text():
    from lingcore.events import TurnCancelled

    cli = _recording_cli()
    cli.render(TextDelta("kept.\n\nvoid partial"))
    cli.render(TurnCancelled())
    out = cli.console.export_text()
    assert "kept." in out
    assert "void partial" not in out
    assert "stopped by user" in out


def test_cli_summarizes_tool_calls_and_results():
    cli = _recording_cli()
    cli.render(
        ToolCallStarted(
            ToolCall(
                id="c",
                name="read_file",
                arguments={"path": "src/a.py", "offset": 5, "limit": 10},
            )
        )
    )
    cli.render(
        ToolResultEvent(
            ToolResult(call_id="c", name="read_file", content="5\tx\n6\ty\n7\tz")
        )
    )
    cli.render(
        ToolResultEvent(
            ToolResult(
                call_id="d",
                name="run_shell",
                ok=False,
                content="exit code 1\nFAILED test_a\n1 failed",
            )
        )
    )
    out = cli.console.export_text()
    assert "read_file src/a.py  offset=5, limit=10" in out
    assert "{'path'" not in out  # no Python dict reprs
    assert "read_file: read 3 lines" in out
    assert "\tx" not in out  # file contents are not dumped
    # Failures show their detail, not a one-line digest.
    assert "run_shell: exit code 1" in out
    assert "FAILED test_a" in out and "1 failed" in out


async def test_cli_input_continuation_and_leading_space(monkeypatch):
    cli = CLIFrontend()
    cli.console.quiet = True
    lines = iter(["first \\", "second", " /etc/hosts as text", "", "/hepl", "ok"])
    monkeypatch.setattr(cli.console, "input", lambda *a, **k: next(lines))
    assert await cli.read_input() == "first \nsecond"
    assert await cli.read_input() == " /etc/hosts as text"
    # Blank lines and unknown commands are consumed without ending the read.
    assert await cli.read_input() == "ok"


def test_completions_cover_commands_and_paths(tmp_path, monkeypatch):
    assert _completions("/re", at_line_start=True) == ["/resume"]
    assert _completions("/re", at_line_start=False) == []
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _completions("@no", at_line_start=False) == ["@notes.txt"]
    assert _completions("@do", at_line_start=False) == ["@docs/"]
    assert _completions("plain", at_line_start=True) == []


@pytest.mark.parametrize(
    "keys",
    [
        "a\x1b[13;2ub\r",  # Shift+Enter, CSI-u terminals (kitty, Ghostty, WezTerm…)
        "a\x1b[27;2;13~b\r",  # Shift+Enter, xterm modifyOtherKeys
        "a\x1b\rb\r",  # Alt/Option+Enter
        "a\nb\r",  # Ctrl-J
        "a\\\rb\r",  # backslash, then Enter
        "\x1b[200~a\nb\x1b[201~\r",  # bracketed paste keeps its newline
    ],
)
async def test_prompt_newline_keys_do_not_submit(keys):
    assert await _prompt_with_keys(keys) == "a\nb"


async def test_prompt_enter_submits_and_ctrl_c_clears_then_quits():
    assert await _prompt_with_keys("hello\r") == "hello"
    assert await _prompt_with_keys("draft\x03kept\r") == "kept"
    with pytest.raises(KeyboardInterrupt):
        await _prompt_with_keys("\x03")
    with pytest.raises(EOFError):
        await _prompt_with_keys("\x04")


@pytest.mark.parametrize(
    ("typed", "chosen", "keys", "expected"),
    [
        ("/he", "/help", "\r", "/help"),  # argument-free command: accept + send
        ("/re", "/resume", "\r1234\r", "/resume 1234"),  # waits for its argument
        ("see @no", "@notes.txt", "\r now\r", "see @notes.txt now"),  # keeps editing
    ],
)
async def test_prompt_enter_accepts_highlighted_completion(
    typed, chosen, keys, expected
):
    from prompt_toolkit.application import get_app
    from prompt_toolkit.buffer import CompletionState
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.document import Document

    def highlight() -> None:
        # As if the user arrowed onto ``chosen`` in the menu: prompt_toolkit has
        # already inserted the candidate into the buffer.
        word = typed.rsplit(" ", 1)[-1]
        buffer = get_app().current_buffer
        buffer.text = typed[: -len(word)] + chosen
        buffer.cursor_position = len(buffer.text)
        buffer.complete_state = CompletionState(
            Document(typed, len(typed)),
            [Completion(chosen, start_position=-len(word))],
            complete_index=0,
        )

    assert await _prompt_with_keys(keys, pre_run=highlight) == expected


def test_input_completer_offers_commands_with_descriptions():
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from lingcore.io.cli import _InputCompleter

    def complete(text: str) -> list[tuple[str, str]]:
        found = _InputCompleter().get_completions(
            Document(text, len(text)), CompleteEvent()
        )
        return [(c.text, c.display_meta_text) for c in found]

    assert complete("/se") == [("/sessions", "list stored sessions for this profile")]
    assert complete("hi /se") == []  # commands only at the start of the line
    assert complete("plain") == []


def test_status_bar_shows_session_and_usage_and_fits_width():
    from lingcore.events import UsageReported
    from lingcore.usage import TokenUsage

    def bar(cli: CLIFrontend, width: int) -> str:
        return "".join(text for _, text in cli._status_fragments(width))

    cli = CLIFrontend(agent_name="daily", model="m1", store=None)
    assert "not saved" in bar(cli, 100)
    cli.set_session("abcd1234ef")
    cli.render(
        UsageReported(
            TokenUsage(
                model="m1",
                input_tokens=10_000,
                cached_input_tokens=8_000,
                output_tokens=500,
            )
        )
    )
    wide = bar(cli, 120)
    assert (
        "daily · m1 · session abcd1234 · ctx 10.5k · ↑10.0k ↓500 (80% cached)" in wide
    )
    assert wide.endswith("/help ") and len(wide) == 120
    for width in (90, 60, 30, 8):
        narrow = bar(cli, width)
        assert "⏎ send" not in narrow
        assert len(narrow) <= max(width, len(" daily"))
    cli.attach({})  # a new session starts with an empty context gauge
    assert "ctx" not in bar(cli, 120)


def test_shell_result_digest_leads_with_exit_code():
    cli = _recording_cli()
    for code, output in (("0", "hi"), ("2", "boom\nmore")):
        cli.render(
            ToolResultEvent(
                ToolResult(
                    call_id="c",
                    name="run_shell",
                    content=f"$ cmd\n(runner: host)\n(exit code: {code})\n{output}",
                )
            )
        )
    out = cli.console.export_text()
    assert "run_shell: exit 0 · hi" in out
    assert "run_shell: exit 2 · boom  … +1 line" in out
    assert "(runner:" not in out


def test_input_recall_is_scoped_to_one_session(monkeypatch):
    cli = CLIFrontend(agent_name="t")
    monkeypatch.setattr(cli, "_tty", True)
    main, _ = cli._sessions()
    main.history.append_string("typed in the first session")
    assert main is cli._sessions()[0]  # stable within a session
    cli.attach({})  # /new or /resume
    fresh, _ = cli._sessions()
    assert fresh is not main
    assert list(fresh.history.get_strings()) == []


async def test_cli_confirm_background_process_never_offers_always_allow(monkeypatch):
    from rich.console import Console

    from lingcore.events import ToolCallStarted
    from lingcore.message import ToolCall

    opts: dict = {}
    cli = CLIFrontend(tool_options=opts)
    cli.console = Console(record=True, width=200)
    prompts: list[str] = []

    def fake_input(prompt, *a, **k):
        prompts.append(prompt)
        return "A"

    monkeypatch.setattr(cli.console, "input", fake_input)
    command = "npm run dev"
    # Also a run_shell command this turn: the process framing still wins, so
    # "A" can never write run_shell's allowlist.
    _announce_shell(cli, command)
    cli.render(
        ToolCallStarted(
            ToolCall(id="p", name="process_start", arguments={"command": command})
        )
    )
    assert await cli.confirm(command) is True
    out = cli.console.export_text()
    assert "start background process?" in out and f"$ {command}" in out
    assert "always allow" not in prompts[0]
    assert opts.get("run_shell", {}).get("allow_patterns", []) == []
    cli._start_turn()
    assert cli._process_commands == set()
