"""A Rich-based terminal frontend.

Renders streamed replies as markdown (completed blocks are committed to the
scrollback while the block in progress previews live), shows a spinner while
the model or a tool is working, summarizes tool calls/results in one line
each, and routes shell-command confirmation prompts to the terminal. Blocking
On a real terminal, input is a prompt_toolkit editor (multi-line with
Shift/Alt+Enter, history, slash-command/``@path`` completion) with a status
bar; elsewhere (pipes, tests) blocking console input runs in a worker thread
so the asyncio event loop is never stalled.
"""

from __future__ import annotations

import asyncio
import difflib
import glob
import json
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, ClassVar

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Heading, Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.segment import Segment
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from lingcore.events import (
    AgentEvent,
    Compacted,
    Error,
    Final,
    PluginNotice,
    SkillActivated,
    StreamRetry,
    TextDelta,
    TodoUpdated,
    ToolCallStarted,
    ToolResultEvent,
    TurnCancelled,
    UsageReported,
)
from lingcore.media import attachment_from_path
from lingcore.media_types import (
    MAX_ATTACHMENTS,
    TOTAL_ATTACHMENT_MAX_BYTES,
    decoded_payload_size,
)
from lingcore.message import Attachment, ToolCall, ToolResult, UserInput
from lingcore.plugins.commands import CommandCatalog
from lingcore.todos import TodoItem
from lingcore.tools.builtin.shell import allowlist_pattern_for
from lingcore.usage import TokenUsage

if TYPE_CHECKING:
    from lingcore.message import Message
    from lingcore.sessions import SessionMeta, SessionStore

_EXIT_COMMANDS = {"/exit", "/quit", "/q"}
_COMMANDS = ("/new", "/sessions", "/resume", "/usage", "/help", "/exit", "/quit")
_HELP_COMMANDS = (
    ("/new", "start a fresh session"),
    ("/sessions", "list stored sessions for this profile"),
    ("/resume <id>", "switch to a stored session by id prefix"),
    ("/usage", "token usage for this session"),
    ("/help", "show this help"),
    ("/exit", "quit (also /quit, /q, Ctrl-D)"),
)
_HELP_KEYS = (
    ("Enter", "send the message"),
    ("Shift/Alt-Enter", "new line (also Ctrl-J, or \\ then Enter)"),
    ("Tab", "complete /commands and @paths"),
    ("↑ / ↓", "move between lines, then browse input history"),
    ("Ctrl-C", "clear the input; stop a running turn (twice quits)"),
    ("Ctrl-D", "quit"),
    ("@path", "attach a file to the message"),
)
_ARGUMENT_COMMANDS = {"/resume"}
_COMMAND_META = {key.split()[0]: text for key, text in _HELP_COMMANDS} | {
    "/quit": "quit"
}
# Diff previews for file edits stay short; the full change is in the workspace.
_DIFF_PREVIEW_LINES = 40
_ATTACH_RE = re.compile(r'(?<!\S)@("[^"]+"|\S+)')
# Argument keys that best identify what a tool call acts on, in priority order.
_HEADLINE_KEYS = ("command", "path", "url", "query", "pattern", "name", "action")
_ERROR_PREVIEW_LINES = 4
# Markdown styles that read on both dark and light terminals (Rich's defaults
# paint inline code on a black background).
_THEME = Theme(
    {
        "markdown.code": "bold cyan",
        "markdown.code_block": "cyan",
        "markdown.h1": "bold underline",
        "markdown.h2": "bold",
        "markdown.h3": "bold",
        "markdown.h4": "bold italic",
        "markdown.block_quote": "italic dim",
        "markdown.hr": "dim",
    }
)
_PROMPT = "\n[bold cyan]❯[/] "
_CONTINUATION_PROMPT = "[dim]·[/] "
_PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "rule": "#5f5f5f",
        "continuation": "#5f5f5f",
        "bottom-toolbar": "noreverse",
        "status": "#8a8a8a",
        "status.agent": "ansicyan bold",
        "status.dim": "#5f5f5f",
        "status.key": "ansicyan",
    }
)
# Shift+Enter has no single encoding. Terminals using the CSI-u or xterm
# modifyOtherKeys schemes send these, which prompt_toolkit would otherwise read
# as plain Enter (submit); route them to Alt+Enter, which inserts a newline.
_SHIFT_ENTER_SEQUENCES = ("\x1b[13;2u", "\x1b[27;2;13~")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _looks_like_path(raw: str) -> bool:
    """Heuristic for whether an unmatched ``@token`` plausibly meant a file.

    Used only to decide whether to warn: a bare ``@mention`` shouldn't nag,
    but a mistyped ``@notes.txt`` should not vanish silently.
    """
    return "/" in raw or raw.startswith("~") or bool(Path(raw).suffix)


def _parse_attachments(
    line: str, base: Path | None = None
) -> tuple[UserInput, list[str]]:
    """Parse ``@path`` CLI attachments from one line.

    A token attaches iff it resolves to an existing file (any type); otherwise
    the ``@token`` stays in the text verbatim. Returns the parsed input plus
    warning lines for tokens that looked like a path but couldn't attach (no
    such file, too large, over the per-message limit) — the typed line is never
    lost, and caps can never make ``UserInput`` construction raise.
    """
    base = base or Path.cwd()
    attachments: list[Attachment] = []
    warnings: list[str] = []
    out: list[str] = []
    pos = 0
    total = 0
    for match in _ATTACH_RE.finditer(line):
        token = match.group(1)
        raw_path = (
            token[1:-1] if token.startswith('"') and token.endswith('"') else token
        )
        path = Path(os.path.expanduser(raw_path))
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            if _looks_like_path(raw_path):
                warnings.append(f"@{raw_path}: no such file; sent as text")
            continue
        if len(attachments) >= MAX_ATTACHMENTS:
            warnings.append(
                f"@{raw_path}: attachment limit ({MAX_ATTACHMENTS}) reached; "
                "sent as text"
            )
            continue
        try:
            attachment = attachment_from_path(path)
        except Exception as e:
            warnings.append(
                f"@{raw_path}: {e}; sent as text — copy it into the workspace "
                "and ask the agent to read it with tools"
            )
            continue
        size = decoded_payload_size(attachment.data)
        if total + size > TOTAL_ATTACHMENT_MAX_BYTES:
            warnings.append(
                f"@{raw_path}: would exceed the total attachment size limit; "
                "sent as text"
            )
            continue
        attachments.append(attachment)
        total += size
        out.append(line[pos : match.start()])
        out.append(raw_path)
        pos = match.end()
    out.append(line[pos:])
    return UserInput(text="".join(out), attachments=attachments), warnings


def rel_time(dt: datetime) -> str:
    """Compact relative time for session listings ("just now", "5m ago")."""
    seconds = int((datetime.now(timezone.utc) - dt).total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _short(text: str, limit: int = 200) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit] + " …"


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def _tilde(path: object) -> str:
    """Shorten a path under the home directory to ``~/…`` for display."""
    text = str(path)
    home = os.path.expanduser("~")
    if home != "~" and (text == home or text.startswith(home + os.sep)):
        return "~" + text[len(home) :]
    return text


@dataclass(slots=True)
class _UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: TokenUsage) -> None:
        self.requests += 1
        self.input_tokens += usage.input_tokens
        self.cached_input_tokens += usage.cached_input_tokens
        self.output_tokens += usage.output_tokens

    def describe(self) -> str:
        noun = "request" if self.requests == 1 else "requests"
        cached = (
            f" ({_tokens(self.cached_input_tokens)} cached)"
            if self.cached_input_tokens
            else ""
        )
        return (
            f"{self.requests} {noun} · in {_tokens(self.input_tokens)}{cached} · "
            f"out {_tokens(self.output_tokens)}"
        )


@dataclass(frozen=True, slots=True)
class SessionSwitch:
    """A ``/new`` or ``/resume`` request for the composition root to act on."""

    session_id: str | None  # None = start a fresh session


def _diff_lines(call: ToolCall) -> list[str] | None:
    """Unified-diff preview lines for a file-editing call, else ``None``."""
    args: dict[str, Any] = call.arguments
    if call.name == "edit_file":
        old, new = args.get("old"), args.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        lines = list(
            difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2)
        )
        return [line for line in lines if not line.startswith(("---", "+++"))]
    if call.name == "patch_file":
        diff = args.get("diff")
        if not isinstance(diff, str):
            return None
        return [
            line
            for line in diff.splitlines()
            if not line.startswith(("---", "+++", "diff --git", "index "))
        ]
    return None


def _style_diff_line(line: str) -> str:
    text = escape(line)
    if line.startswith("@@"):
        return f"[cyan]{text}[/]"
    if line.startswith("+"):
        return f"[green]{text}[/]"
    if line.startswith("-"):
        return f"[red]{text}[/]"
    return f"[dim]{text}[/]"


def _attachment_summary(message: "Message") -> str:
    if not message.attachments:
        return ""
    labels = [f"{a.kind}: {a.name or a.media_type}" for a in message.attachments]
    return " [" + "; ".join(labels) + "]"


def _brief(value: object, limit: int = 40) -> str:
    """Compact one-line rendering of a tool argument value."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    return _short(text, limit)


def _call_text(call: ToolCall) -> Text:
    """One-line summary of a tool call: marker, name, headline argument, rest.

    Built as a ``Text`` (never markup): tool names and arguments are model
    output, so they must render literally.
    """
    args: dict[str, Any] = call.arguments
    text = Text(no_wrap=call.name != "run_shell", overflow="ellipsis")
    text.append("▸ ", style="cyan")
    text.append(call.name, style="bold")
    shown: set[str] = set()
    head = next(
        (k for k in _HEADLINE_KEYS if isinstance(args.get(k), str) and args[k]), None
    )
    if head is not None:
        shown.add(head)
        value = _short(args[head], 300)
        text.append(" ")
        text.append(f"$ {value}" if call.name == "run_shell" else value)
    if call.name in {"edit_file", "patch_file"}:
        shown.update({"old", "new", "diff"})  # shown as the diff preview
    if call.name == "write_file" and isinstance(args.get("content"), str):
        shown.add("content")
        lines = len(args["content"].splitlines())
        text.append(f"  ({_plural(lines, 'line')})", style="dim")
    rest = [f"{k}={_brief(v)}" for k, v in args.items() if k not in shown]
    if rest:
        text.append("  " + ", ".join(rest), style="dim")
    return text


def _result_text(result: ToolResult) -> Text:
    """One-line digest of a successful tool result (the model sees it all)."""
    lines = [line for line in result.content.strip().splitlines() if line.strip()]
    text = Text(no_wrap=True, overflow="ellipsis", style="dim")
    if not lines:
        text.append("(no output)")
    elif result.name == "read_file" and "\t" in lines[0]:
        numbered = sum(1 for line in lines if "\t" in line)
        text.append(f"read {_plural(numbered, 'line')}")
        if lines[-1].startswith("…"):
            text.append(" (more available)")
    elif result.name == "list_dir" and lines[0] != "(empty)":
        entries = sum(1 for line in lines if not line.startswith("… ("))
        text.append(f"{entries} {'entry' if entries == 1 else 'entries'}")
        if entries < len(lines):
            text.append(f" {lines[-1]}")
    elif result.name == "run_shell" and lines[0].startswith("$ "):
        # "$ cmd" / "(runner: …)" / "(exit code: N)" header, then the output;
        # the command itself is already on the call line.
        output = lines[1:]
        code = None
        while output and output[0].startswith(("(runner: ", "(exit code: ")):
            meta = output.pop(0)
            if meta.startswith("(exit code: "):
                code = meta.removeprefix("(exit code: ").rstrip(")")
        if code is not None:
            text.append(f"exit {code}", style="dim" if code == "0" else "red")
        if output:
            text.append(" · " if code is not None else "")
            text.append(output[0])
            if len(output) > 1:
                text.append(f"  … +{_plural(len(output) - 1, 'line')}")
    else:
        text.append(lines[0])
        if len(lines) > 1:
            text.append(f"  … +{_plural(len(lines) - 1, 'line')}")
    if result.attachments:
        text.append(f"  [+{_plural(len(result.attachments), 'attachment')}]")
    return text


def _stable_prefix(text: str) -> int:
    """Length of the prefix of streamed ``text`` that holds only complete
    markdown blocks — up to the last blank line or closing code fence outside
    an open fence. Everything after it may still change as tokens arrive.
    """
    boundary = 0
    fence: str | None = None
    pos = 0
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            break  # the trailing line is still being streamed
        pos += len(line)
        match = _FENCE_RE.match(line)
        if fence is not None:
            marker = match.group(1) if match else ""
            if (
                marker
                and marker[0] == fence[0]
                and len(marker) >= len(fence)
                and line.strip() == marker
            ):
                fence = None
                boundary = pos
        elif match:
            fence = match.group(1)
        elif not line.strip():
            boundary = pos
    return boundary


class _LeftHeading(Heading):
    """Rich centers ``#`` headings; a chat transcript reads better flush left."""

    LEVEL_ALIGN: ClassVar[dict[str, Any]] = dict.fromkeys(
        ("h1", "h2", "h3", "h4", "h5", "h6"), "left"
    )


class _CodeBlock(CodeBlock):
    """Indented instead of boxed: Rich pads code blocks with blank rows meant
    to carry a background colour, which the terminal-palette theme omits."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Syntax(
            code, self.lexer_name, theme=self.theme, word_wrap=True, padding=(0, 2)
        )


class _Markdown(Markdown):
    elements: ClassVar[dict[str, Any]] = {
        **Markdown.elements,
        "heading_open": _LeftHeading,
        "fence": _CodeBlock,
        "code_block": _CodeBlock,
    }


def _markdown(text: str) -> Markdown:
    # hyperlinks=False prints "label (url)": an OSC-8 link would let model
    # text hide its real target behind arbitrary label text.
    return _Markdown(text, code_theme="ansi_dark", hyperlinks=False)


class _Tail:
    """Render only the last lines of a renderable that fit the live region,
    so a long block in progress scrolls instead of being cropped at the top."""

    def __init__(self, renderable: RenderableType) -> None:
        self.renderable = renderable

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        lines = console.render_lines(self.renderable, options, pad=False)
        keep = max(1, console.size.height - 3)
        for line in lines[-keep:]:
            yield from line
            yield Segment.line()


class _Working:
    """Spinner with the current phase and the turn's elapsed time."""

    def __init__(self) -> None:
        self._spinner = Spinner("dots", style="cyan")
        self.label = "thinking"
        self.started = time.monotonic()

    def __rich__(self) -> RenderableType:
        elapsed = time.monotonic() - self.started
        self._spinner.update(
            text=Text.assemble(
                (f"{self.label}…", "cyan"),
                (f"  {_duration(elapsed)} · ctrl-c to stop", "dim"),
            )
        )
        return self._spinner


def _completions(
    text: str, at_line_start: bool, commands: CommandCatalog | None = None
) -> list[str]:
    """Readline candidates: slash commands at line start, ``@`` file paths."""
    if text.startswith("/") and at_line_start:
        names = list(_COMMANDS) + (
            [c.qualified_name for c in commands.commands] if commands else []
        )
        return [command for command in names if command.startswith(text)]
    if text.startswith("@"):
        prefix = text[1:]
        expanded = os.path.expanduser(prefix)
        out = []
        for match in sorted(glob.glob(glob.escape(expanded) + "*"))[:200]:
            # Keep the user's own spelling (e.g. "~/") in front of the match.
            shown = prefix + match[len(expanded) :]
            out.append("@" + shown + ("/" if os.path.isdir(match) else ""))
        return out
    return []


class _InputCompleter(Completer):
    """Complete slash commands at line start and ``@`` attachment paths."""

    def __init__(self, commands: CommandCatalog | None = None) -> None:
        self.commands = commands
        self._metadata = (
            {c.qualified_name: c.description for c in commands.commands}
            if commands
            else {}
        )

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith(("/", "@")):
            return
        at_line_start = document.text_before_cursor == word
        for candidate in _completions(word, at_line_start, self.commands):
            yield Completion(
                candidate,
                start_position=-len(word),
                display_meta=_COMMAND_META.get(candidate, "")
                or (self._metadata.get(candidate, "")),
            )


def _input_key_bindings() -> KeyBindings:
    for sequence in _SHIFT_ENTER_SEQUENCES:
        ANSI_SEQUENCES[sequence] = (Keys.Escape, Keys.ControlM)
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        state = buffer.complete_state
        if state is not None and state.current_completion is not None:
            # The menu already inserted the highlighted candidate; Enter
            # accepts it. Only a complete argument-free command also submits —
            # a path or an argument-taking command keeps the line open.
            buffer.complete_state = None
            chosen = state.current_completion.text
            if chosen in _ARGUMENT_COMMANDS:
                buffer.insert_text(" ")
                return
            if not chosen.startswith("/") or buffer.text != chosen:
                return
        if buffer.document.text_before_cursor.endswith("\\"):
            buffer.delete_before_cursor(1)
            buffer.insert_text("\n")
            return
        buffer.validate_and_handle()

    @bindings.add("escape", "enter")  # Alt+Enter, and Shift+Enter (see above)
    @bindings.add("c-j")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def _clear_or_quit(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.text:
            buffer.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    return bindings


def _continuation(
    width: int, line_number: int, is_soft_wrap: int
) -> StyleAndTextTuples:
    return [("class:continuation", " " * width if is_soft_wrap else "· ")]


def session_table(sessions: Sequence["SessionMeta"]) -> Table:
    """Stored-session listing shared by ``/sessions`` and ``--list-sessions``."""
    table = Table(box=None, pad_edge=False, header_style="bold dim")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("title", overflow="ellipsis", no_wrap=True)
    table.add_column("msgs", justify="right", style="dim")
    table.add_column("updated", style="dim", no_wrap=True)
    table.add_column("created", style="dim", no_wrap=True)
    for meta in sessions:
        table.add_row(
            meta.id[:8],
            Text(meta.title or "(untitled)"),
            str(meta.message_count),
            rel_time(meta.updated_at),
            rel_time(meta.created_at),
        )
    return table


class CLIFrontend:
    """Implements the ``Frontend`` protocol over a Rich console."""

    def __init__(
        self,
        agent_name: str = "agent",
        tool_options: "dict | None" = None,
        store: "SessionStore | None" = None,
        model: str | None = None,
        commands: CommandCatalog | None = None,
    ) -> None:
        self.console = Console(theme=_THEME)
        self.agent_name = agent_name
        self._model = model
        self._commands = commands or CommandCatalog()
        # Status-bar state: which session this is and how full the context was
        # after the last model request (its input + output tokens).
        self._session_label = "not saved" if store is None else None
        self._context_tokens = 0
        # Streamed reply text not yet committed to the scrollback (the block in
        # progress), and whether this reply segment has committed anything.
        self._pending = ""
        self._segment_open = False
        self._segment_committed = False
        # What the transcript printed last ("text", "tool", or None at turn
        # start) — drives the blank-line rhythm between replies and tool runs.
        self._last_kind: str | None = None
        # The one transient Live region: spinner or reply-in-progress preview.
        self._live: Live | None = None
        self._working = _Working()
        self._turn_started: float | None = None
        # Serialize confirmation prompts: with parallel_tools, two tools can call
        # confirm() at once — without this their console.input threads would race
        # on stdin and interleave prompts.
        self._confirm_lock = asyncio.Lock()
        # This turn's run_shell commands. A confirmation whose text is one of
        # them is a shell prompt (with "always allow"); any other is a generic
        # approval (skill activation, a non-public fetch, ...).
        self._shell_commands: set[str] = set()
        # Shared with the agent's ToolContext so "allow always" writes land live.
        self._tool_options: dict = tool_options if tool_options is not None else {}
        # Session store for /sessions and /resume (None = persistence off).
        self._store = store
        # A console.input thread whose awaiting task was cancelled (Stop during
        # a confirmation prompt). The thread still owns stdin, so the next
        # prompt must consume its line first instead of racing it.
        self._stale_input: asyncio.Future[str] | None = None
        self._turn_usage = _UsageTotals()
        self._session_usage = _UsageTotals()
        self._session_switch: SessionSwitch | None = None
        # The running turn's stop request while interrupt_scope is active (the
        # prompt_toolkit confirm prompt sees Ctrl-C as a key, not a signal).
        self._request_stop: Callable[[], None] | None = None
        self._tty = sys.stdin.isatty() and sys.stdout.isatty()
        self._main_prompt: PromptSession[str] | None = None
        self._confirm_prompt: PromptSession[str] | None = None

    def set_commands(self, commands: CommandCatalog) -> None:
        """Rebind command help/completion for the active Agent."""
        self._commands = commands
        self._main_prompt = None

    def attach(
        self, tool_options: dict, commands: CommandCatalog | None = None
    ) -> None:
        """Rebind to a new agent's options after a session switch.

        Session allowlists, usage counters, and input recall (↑/↓ history)
        belong to the session they were granted or typed in, so they do not
        carry over.
        """
        self._tool_options = tool_options
        if commands is not None:
            self.set_commands(commands)
        self._turn_usage = _UsageTotals()
        self._session_usage = _UsageTotals()
        self._context_tokens = 0
        # Rebuilt lazily with an empty in-memory history on the next prompt.
        self._main_prompt = self._confirm_prompt = None

    def set_session(self, session_id: str | None) -> None:
        """Show the current session in the status bar (``None`` = not saved)."""
        self._session_label = (
            f"session {session_id[:8]}" if session_id is not None else "not saved"
        )

    def take_session_switch(self) -> SessionSwitch | None:
        """Return and clear a pending ``/new`` or ``/resume`` request."""
        switch, self._session_switch = self._session_switch, None
        return switch

    # --- input ----------------------------------------------------------

    @property
    def _rich_prompt(self) -> bool:
        """Use the prompt_toolkit editor (a real terminal on both ends)."""
        return self._tty and self.console.is_terminal

    def _sessions(self) -> tuple[PromptSession[str], PromptSession[str]]:
        if self._main_prompt is None or self._confirm_prompt is None:
            self._main_prompt = PromptSession(
                message=self._prompt_message,
                multiline=True,
                key_bindings=_input_key_bindings(),
                history=InMemoryHistory(),
                completer=_InputCompleter(self._commands),
                complete_while_typing=True,
                reserve_space_for_menu=4,
                bottom_toolbar=self._status_bar,
                prompt_continuation=_continuation,
                style=_PROMPT_STYLE,
            )
            # Separate history: confirmation answers are not messages.
            self._confirm_prompt = PromptSession(style=_PROMPT_STYLE)
        return self._main_prompt, self._confirm_prompt

    def _prompt_message(self) -> StyleAndTextTuples:
        width = get_app().output.get_size().columns
        return [
            ("", "\n"),
            ("class:rule", "─" * width + "\n"),
            ("class:prompt", "❯ "),
        ]

    def _status_bar(self) -> StyleAndTextTuples:
        return self._status_fragments(get_app().output.get_size().columns)

    def _status_fragments(self, width: int) -> StyleAndTextTuples:
        """The status bar: who/what/where on the left, key hints on the right.

        Items are dropped from the end (hints first) until the line fits.
        """
        items = [self._model, self._session_label]
        if self._context_tokens:
            items.append(f"ctx {_tokens(self._context_tokens)}")
        usage = self._session_usage
        if usage.requests:
            spent = f"↑{_tokens(usage.input_tokens)} ↓{_tokens(usage.output_tokens)}"
            if usage.cached_input_tokens and usage.input_tokens:
                share = usage.cached_input_tokens * 100 // usage.input_tokens
                spent += f" ({share}% cached)"
            items.append(spent)
        left: StyleAndTextTuples = [("class:status.agent", f" {self.agent_name}")]
        for item in items:
            if item:
                left += [("class:status.dim", " · "), ("class:status", item)]
        right: StyleAndTextTuples = [
            ("class:status.key", "⏎"),
            ("class:status.dim", " send  "),
            ("class:status.key", "⇧⏎"),
            ("class:status.dim", " newline  "),
            ("class:status.key", "/help"),
            ("class:status.dim", " "),
        ]
        gap = width - fragment_list_width(left) - fragment_list_width(right)
        if gap >= 2:
            return left + [("", " " * gap)] + right
        while len(left) > 1 and fragment_list_width(left) > width:
            del left[-2:]  # drop the last " · item" pair
        return left

    def _ansi(self, markup: str) -> ANSI:
        """Rich markup rendered to ANSI for a prompt_toolkit prompt."""
        with self.console.capture() as capture:
            self.console.print(markup, end="")
        return ANSI(capture.get())

    async def _read_message(self) -> str:
        if not self._rich_prompt:
            line = await self._input(_PROMPT)
            # A trailing backslash continues the message on the next line.
            while line.endswith("\\"):
                line = line[:-1] + "\n" + await self._input(_CONTINUATION_PROMPT)
            return line
        self._stop_live()
        main, _ = self._sessions()
        return await main.prompt_async(handle_sigint=False)

    async def _input(self, prompt: str) -> str:
        """``console.input`` in a worker thread, safe against cancellation."""
        stale, self._stale_input = self._stale_input, None
        if stale is not None:
            try:
                await stale  # the abandoned prompt's line is discarded
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                pass
        self._stop_live()
        future = asyncio.ensure_future(asyncio.to_thread(self.console.input, prompt))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if not future.done():
                self._stale_input = future
            raise

    @contextmanager
    def interrupt_scope(self, stop: Callable[[], bool]) -> Iterator[None]:
        """Make Ctrl-C stop the running turn instead of quitting.

        A second Ctrl-C while the stop is pending falls through to the previous
        handler (asyncio's, which ends the process). Only the main thread can
        own signal handlers; elsewhere the scope is a no-op.
        """
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        loop = asyncio.get_running_loop()
        previous = signal.getsignal(signal.SIGINT)
        pressed = False

        def request_stop() -> None:
            if stop():
                self.console.print("[yellow]■ stopping… (Ctrl-C again to quit)[/]")
                self._show_working("stopping")

        def handler(signum: int, frame: FrameType | None) -> None:
            nonlocal pressed
            if pressed and callable(previous):
                previous(signum, frame)
                return
            pressed = True
            # Signal handlers must not touch the loop directly.
            loop.call_soon_threadsafe(request_stop)

        signal.signal(signal.SIGINT, handler)
        self._request_stop = request_stop
        try:
            yield
        finally:
            self._request_stop = None
            signal.signal(signal.SIGINT, previous)

    def _command(self, line: str) -> bool | None:
        """Handle a slash command. ``True`` = handled, ``None`` = end session,
        ``False`` = not a command (send as a message)."""
        parts = line.split()
        if not parts:
            return False
        name, rest = parts[0], parts[1:]
        if name in _EXIT_COMMANDS:
            return None
        if name == "/help":
            self._print_help()
            return True
        if name == "/usage":
            if self._session_usage.requests:
                self.console.print(
                    f"[dim]session usage: {self._session_usage.describe()}[/]"
                )
            else:
                self.console.print("[dim]no usage reported yet[/]")
            return True
        if name == "/new":
            self._session_switch = SessionSwitch(None)
            return None
        if name == "/sessions":
            self._list_sessions()
            return True
        if name == "/resume":
            if self._store is None:
                self.console.print("[yellow]sessions are disabled for this profile[/]")
                return True
            if len(rest) != 1:
                self.console.print(
                    "[yellow]usage: /resume <id-prefix>[/] "
                    "[dim](see /sessions for ids)[/]"
                )
                return True
            from lingcore.errors import SessionError

            try:
                meta = self._store.resolve_prefix(rest[0])
            except SessionError as e:
                self.console.print(f"[yellow]{escape(str(e))}[/]")
                return True
            self._session_switch = SessionSwitch(meta.id)
            return None
        if self._commands.resolve(line, reserved=_COMMANDS + ("/q",)) is not None:
            return False
        if "/" not in name[1:]:
            # "/foo" with no further slash is a mistyped command, not a path.
            self.console.print(
                f"[yellow]unknown command {escape(name)}[/] [dim]— /help lists "
                "commands; start the line with a space to send it as text[/]"
            )
            return True
        return False

    def _print_help(self) -> None:
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="cyan", no_wrap=True)
        grid.add_column()
        grid.add_row(Text("commands", style="bold"), "")
        for key, description in _HELP_COMMANDS:
            grid.add_row(Text("  " + key), description)
        for command in self._commands.commands:
            label = command.qualified_name + (
                " " + command.argument_hint if command.argument_hint else ""
            )
            grid.add_row(Text("  " + label), command.description)
        grid.add_row("", "")
        grid.add_row(Text("keys", style="bold"), "")
        for key, description in _HELP_KEYS:
            grid.add_row(Text("  " + key), description)
        self.console.print(grid)

    def _list_sessions(self) -> None:
        if self._store is None:
            self.console.print("[dim]sessions are disabled for this profile[/]")
            return
        sessions = self._store.list()
        if not sessions:
            self.console.print("[dim]no stored sessions[/]")
            return
        self.console.print(session_table(sessions[:20]))
        if len(sessions) > 20:
            self.console.print(f"[dim]… {len(sessions) - 20} more[/]")
        self.console.print("[dim]switch with /resume <id>[/]")

    async def read_input(self) -> str | UserInput | None:
        while True:
            if self._stale_input is not None:
                self.console.print(
                    "[dim](the stopped prompt is still waiting — press Enter)[/]"
                )
            try:
                line = await self._read_message()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye[/]")
                return None
            if line.startswith("/"):
                handled = self._command(line.strip())
                if handled is None:
                    return None
                if handled:
                    continue
            if line.strip():
                break
        try:
            incoming, warnings = _parse_attachments(line)
        except Exception as e:
            # Parsing must never lose the user's line; fall back to plain text.
            self.console.print(f"[red]attachment error:[/] {escape(str(e))}")
            self._start_turn()
            return line
        for warning in warnings:
            self.console.print(f"[yellow]⚠ {escape(warning)}[/]")
        for attachment in incoming.attachments:
            self.console.print(
                f"[dim]📎 {escape(attachment.name or attachment.media_type)}"
                f" ({escape(attachment.media_type)})[/]"
            )
        expanded = self._commands.resolve(incoming.text, reserved=_COMMANDS + ("/q",))
        if expanded is not None:
            expanded.attachments = incoming.attachments
            expanded.display_text = line
            incoming = expanded
        self._start_turn()
        return incoming if incoming.attachments or expanded is not None else line

    # --- live region ----------------------------------------------------

    @property
    def _interactive(self) -> bool:
        console = self.console
        return (
            console.is_terminal and not console.is_dumb_terminal and not console.quiet
        )

    def _show_live(self, renderable: RenderableType) -> None:
        if not self._interactive:
            return
        if self._live is None:
            self._live = Live(
                renderable,
                console=self.console,
                transient=True,
                refresh_per_second=10,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
        else:
            self._live.update(renderable)

    def _stop_live(self) -> None:
        if self._live is not None:
            live, self._live = self._live, None
            live.stop()

    def _show_working(self, label: str) -> None:
        self._working.label = label
        self._show_live(self._working)

    def _start_turn(self) -> None:
        self._turn_started = time.monotonic()
        self._working.started = self._turn_started
        self._last_kind = None
        self._shell_commands.clear()
        self._show_working("thinking")

    # --- streamed reply -------------------------------------------------

    def _reply_block(self, text: str, *, first: bool, tail: bool = False) -> Table:
        """A reply block with the ``●`` marker gutter on a segment's first block."""
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1, no_wrap=True)
        grid.add_column(ratio=1)
        body: RenderableType = _markdown(text)
        marker = Text("●", style="bold green") if first else ""
        grid.add_row(marker, _Tail(body) if tail else body)
        return grid

    def _open_segment(self) -> None:
        if self._segment_open:
            return
        self._stop_live()
        self.console.print()
        self._segment_open = True
        self._segment_committed = False
        self._last_kind = "text"

    def _commit(self, text: str) -> None:
        if not text.strip():
            return
        if self._segment_committed:
            self.console.print()
        self.console.print(self._reply_block(text, first=not self._segment_committed))
        self._segment_committed = True

    def _feed(self, text: str) -> None:
        self._open_segment()
        self._pending += text
        cut = _stable_prefix(self._pending)
        if cut:
            done, self._pending = self._pending[:cut], self._pending[cut:]
            self._commit(done)
        if self._pending.strip():
            self._show_live(
                self._reply_block(
                    self._pending, first=not self._segment_committed, tail=True
                )
            )

    def _flush(self) -> None:
        """Commit any reply text in progress and close the segment."""
        self._stop_live()
        if self._segment_open:
            pending, self._pending = self._pending, ""
            self._commit(pending)
        self._segment_open = False

    def _discard(self) -> bool:
        """Drop reply text in progress; report whether any was committed."""
        self._stop_live()
        committed = self._segment_open and self._segment_committed
        self._pending = ""
        self._segment_open = False
        return committed

    def _note(self, markup: str) -> None:
        """A status line between transcript items."""
        self._flush()
        self.console.print(markup)

    # --- events ---------------------------------------------------------

    def render(self, event: AgentEvent) -> None:
        match event:
            case TextDelta(text):
                # Streamed model text is rendered as markdown (Rich builds Text
                # objects, never parses markup), so a model cannot inject
                # terminal styling/spoofing or crash the render on a "[".
                self._feed(text)
            case ToolCallStarted(call):
                if call.name == "run_shell":
                    command = call.arguments.get("command")
                    if isinstance(command, str):
                        self._shell_commands.add(command)
                self._flush()
                self._render_call(call)
                self._show_working(f"running {call.name}")
            case ToolResultEvent(result):
                self._flush()
                self._render_result(result)
                self._show_working("thinking")
            case PluginNotice(plugin, hook, action, message):
                self._flush()
                self.console.print(
                    Text(
                        f"plugin {plugin} · {hook} · {action}: {message}",
                        style="yellow",
                    )
                )
            case SkillActivated(name, active):
                verb = "activated" if active else "deactivated"
                self._note(f"[dim]◆ skill {verb}: {escape(name)}[/]")
                self._show_working("thinking")
            case Compacted(summarized_messages, before_tokens, after_tokens):
                self._note(
                    f"[dim]⊞ compacted {summarized_messages} earlier message(s) → "
                    f"summary (~{_tokens(before_tokens)} → ~{_tokens(after_tokens)} "
                    "tokens)[/]"
                )
                self._show_working("thinking")
            case StreamRetry(attempt, max_attempts, reason, discarded_chars):
                committed = self._discard()
                note = (
                    " — partial reply above discarded"
                    if discarded_chars and committed
                    else ""
                )
                self.console.print(
                    f"[yellow]⟲ {escape(reason)}; "
                    f"retrying ({attempt}/{max_attempts}){note}[/]"
                )
                self._show_working("retrying")
            case TurnCancelled(reason):
                self._discard()  # partial assistant text is void
                self.console.print(f"[yellow]■ {escape(reason)}[/]")
                self._end_turn()
            case TodoUpdated(todos):
                self._flush()
                self._render_todos(todos)
                self._show_working("thinking")
            case UsageReported(usage):
                self._turn_usage.add(usage)
                self._session_usage.add(usage)
                self._context_tokens = usage.input_tokens + usage.output_tokens
            case Final(_):
                self._flush()
                self._end_turn()
            case Error(message):
                self._flush()
                self.console.print(Text.assemble(("✗ error: ", "bold red"), message))
                self._end_turn()

    def _end_turn(self) -> None:
        """Print the turn's footer (time + usage) once, then reset the turn."""
        self._stop_live()
        parts = []
        if self._turn_started is not None:
            parts.append(_duration(time.monotonic() - self._turn_started))
        if self._turn_usage.requests:
            parts.append(self._turn_usage.describe())
        if parts:
            self.console.print(f"[dim]  ↳ {' · '.join(parts)}[/]")
        self._turn_usage = _UsageTotals()
        self._turn_started = None

    def _tool_gap(self) -> None:
        """Separate a tool run from the user's message or reply text before it."""
        if self._last_kind != "tool":
            self.console.print()
        self._last_kind = "tool"

    def _render_todos(self, todos: "tuple[TodoItem, ...]") -> None:
        self._tool_gap()
        if not todos:
            self.console.print("[dim]☐ todo list cleared[/]")
            return
        done = sum(1 for item in todos if item.status == "completed")
        self.console.print(f"[bold]☐ todos[/] [dim]{done}/{len(todos)} done[/]")
        for item in todos:
            if item.status == "completed":
                line = Text.assemble("  ✓ ", (item.content, "strike"), style="dim")
            elif item.status == "in_progress":
                line = Text.assemble("  ▸ ", item.content, style="bold yellow")
            else:
                line = Text.assemble("  ○ ", item.content)
            self.console.print(line)

    def _render_call(self, call: ToolCall) -> None:
        if call.name == "todo_write":
            return  # the resulting TodoUpdated checklist is the useful view
        self._tool_gap()
        self.console.print(_call_text(call))
        diff = _diff_lines(call)
        if diff is None or not isinstance(call.arguments.get("path"), str):
            return
        for line in diff[:_DIFF_PREVIEW_LINES]:
            self.console.print("    " + _style_diff_line(line), soft_wrap=True)
        if len(diff) > _DIFF_PREVIEW_LINES:
            self.console.print(
                f"[dim]    … {len(diff) - _DIFF_PREVIEW_LINES} more diff lines[/]"
            )

    def _render_result(self, result: ToolResult) -> None:
        if result.ok and result.name == "todo_write":
            return  # shown as the TodoUpdated checklist instead
        self._tool_gap()
        line = Text("  ⎿ ", style="dim")
        line.append(f"{result.name}: ", style="dim")
        if result.ok:
            line.append_text(_result_text(result))
            line.no_wrap, line.overflow = True, "ellipsis"
            self.console.print(line)
            return
        lines = result.content.strip().splitlines() or ["(no detail)"]
        line.append(lines[0], style="red")
        self.console.print(line)
        for extra in lines[1:_ERROR_PREVIEW_LINES]:
            self.console.print(Text("    " + extra, style="red"))
        if len(lines) > _ERROR_PREVIEW_LINES:
            self.console.print(
                f"[dim]    … +{_plural(len(lines) - _ERROR_PREVIEW_LINES, 'line')}[/]"
            )

    # --- composition-root UI ---------------------------------------------

    def show_banner(
        self, *, model: str, workspace: object, notice: str | None = None
    ) -> None:
        """Print the startup header. Composition-root UI, not ``Frontend``."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column()
        grid.add_row("agent", Text(self.agent_name, style="bold cyan"))
        grid.add_row("model", Text(model))
        grid.add_row("workspace", Text(_tilde(workspace)))
        self.console.print(
            Panel(
                grid,
                title="[bold]LingCore[/]",
                title_align="left",
                border_style="dim",
                expand=False,
                padding=(0, 1),
            )
        )
        if notice:
            self.console.print(Text(notice, style="dim"))
        self.console.print(
            "[dim]/help commands · @file attach · shift/alt-enter new line · "
            "ctrl-c stop turn · ctrl-d quit[/]"
        )

    def show_resume(
        self, meta: "SessionMeta", messages: "list[Message]", tail: int = 6
    ) -> None:
        """Print a resume banner plus a dim replay of the last few messages.

        Composition-root UI (called before the session loop starts), so it is
        not part of the ``Frontend`` protocol.
        """
        title = meta.title or "(untitled)"
        header = Text.assemble(
            ("resumed ", "dim"),
            (meta.id[:8], "cyan"),
            (" · ", "dim"),
            (title, "bold"),
            (
                f" · {meta.message_count} stored messages · last active "
                f"{rel_time(meta.updated_at)}",
                "dim",
            ),
        )
        self.console.rule(header, align="left", style="dim")
        shown = messages[-tail:]
        if len(messages) > len(shown):
            self.console.print(
                f"[dim]  … {len(messages) - len(shown)} earlier messages omitted …[/]"
            )
        for m in shown:
            if m.role == "user":
                summary = _attachment_summary(m)
                if m.name == "media":
                    self._replay(f"  ↥ media{summary}")
                elif m.name == "summary":
                    self._replay(f"  ≋ earlier summary › {_short(m.content)}")
                else:
                    text = m.input_text if m.input_text is not None else m.content
                    self._replay(f"  ❯ {_short(text)}{summary}", style="cyan")
            elif m.role == "assistant":
                if m.content:
                    self._replay(f"  ● {_short(m.content)}")
                for tc in m.tool_calls:
                    # Tool names are model-generated (an unknown-tool call is
                    # stored verbatim) — Text renders them literally.
                    self._replay(f"  {_call_text(tc).plain}")
            else:  # tool result
                result = ToolResult(call_id="", name=m.name or "", content=m.content)
                self._replay(f"  ⎿ {result.name}: {_result_text(result).plain}")
        self.console.rule(style="dim")

    def _replay(self, line: str, style: str = "") -> None:
        self.console.print(
            Text(line, style=f"dim {style}".strip(), no_wrap=True, overflow="ellipsis")
        )

    async def confirm(self, command: str) -> bool:
        # One prompt at a time: parallel tool calls must not race on stdin.
        async with self._confirm_lock:
            self._flush()
            if command in self._shell_commands:
                allowed = await self._confirm_shell(command)
                label = "run_shell" if allowed else "tools"
            else:
                allowed = await self._confirm_action(command)
                label = "tools"
            self._show_working(f"running {label}")
            return allowed

    async def _ask(self, prompt: str) -> str | None:
        """Read one confirmation answer; ``None`` when Ctrl-C/Ctrl-D denied it."""
        if not self._rich_prompt:
            return (await self._input(prompt)).strip()
        self._stop_live()
        _, confirm_prompt = self._sessions()
        try:
            answer = await confirm_prompt.prompt_async(
                self._ansi(prompt), handle_sigint=False
            )
        except (EOFError, KeyboardInterrupt):
            # Raw-mode Ctrl-C is a key, not SIGINT: deny, stop the turn.
            self.console.print("[red]  denied[/]")
            if self._request_stop is not None:
                self._request_stop()
            return None
        return answer.strip()

    async def _confirm_action(self, request: str) -> bool:
        self.console.print(
            Panel(
                Text(request),
                title="[yellow]approve?[/]",
                title_align="left",
                border_style="yellow",
                expand=False,
            )
        )
        answer = await self._ask(
            "  [yellow]y[/] allow once [dim](Enter)[/] · [yellow]n[/] deny\n"
            "[yellow]❯[/] "
        )
        if answer is None:
            return False
        allowed = answer.lower() in {"", "a", "y", "yes"}
        self.console.print("[dim]  allowed once[/]" if allowed else "[red]  denied[/]")
        return allowed

    async def _confirm_shell(self, command: str) -> bool:
        pattern = allowlist_pattern_for(command)
        self.console.print(
            Panel(
                Text(f"$ {command}", style="bold"),
                title="[yellow]run shell command?[/]",
                title_align="left",
                border_style="yellow",
                expand=False,
            )
        )
        session_choice = (
            f"[yellow]A[/] always allow [bold]{escape(pattern)}[/] this session"
            if pattern
            else "[dim]A always-allow not available (shell control syntax)[/]"
        )
        answer = await self._ask(
            f"  [yellow]y[/] allow once [dim](Enter)[/] · {session_choice} · "
            "[yellow]n[/] deny\n[yellow]❯[/] "
        )
        if answer is None:
            return False
        if answer == "A":
            # Persist approval for this exact token prefix for the rest of the session.
            run_shell_opts = self._tool_options.setdefault("run_shell", {})
            patterns: list[str] = run_shell_opts.setdefault("allow_patterns", [])
            if not pattern:
                self.console.print(
                    "[dim]  allowed once — command was not added to the "
                    "session allowlist[/]"
                )
            else:
                if pattern not in patterns:
                    patterns.append(pattern)
                self.console.print(
                    f"[dim]  allowed · {escape(repr(pattern))} added to "
                    "session allowlist[/]"
                )
            return True
        # Empty / Enter, "a", "y", "yes" all mean allow once.
        allowed = answer.lower() in {"", "a", "y", "yes"}
        self.console.print("[dim]  allowed once[/]" if allowed else "[red]  denied[/]")
        return allowed
