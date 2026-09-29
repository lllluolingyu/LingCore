"""A Rich-based terminal frontend.

Renders streamed text live, surfaces tool calls/results dimly, and routes
shell-command confirmation prompts to the terminal. Blocking console input
runs in a worker thread so the asyncio event loop is never stalled.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape

from lingcore.events import (
    AgentEvent,
    Compacted,
    Error,
    Final,
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
from lingcore.message import Attachment, ToolCall, UserInput
from lingcore.todos import TodoItem
from lingcore.tools.builtin.shell import allowlist_pattern_for
from lingcore.usage import TokenUsage

if TYPE_CHECKING:
    from lingcore.message import Message
    from lingcore.sessions import SessionMeta, SessionStore

_EXIT_COMMANDS = {"/exit", "/quit", "/q"}
_HELP = """[bold]commands[/]
  /new              start a fresh session
  /sessions         list stored sessions for this profile
  /resume <id>      switch to a stored session by id prefix
  /usage            token usage for this session
  /help             show this help
  /exit             quit (also /quit, /q, Ctrl-D)
[bold]keys[/]
  Ctrl-C            stop the running turn (press again to quit)
  @path             attach a file to the message"""
# Diff previews for file edits stay short; the full change is in the workspace.
_DIFF_PREVIEW_LINES = 40
_ATTACH_RE = re.compile(r'(?<!\S)@("[^"]+"|\S+)')


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


class CLIFrontend:
    """Implements the ``Frontend`` protocol over a Rich console."""

    def __init__(
        self,
        agent_name: str = "agent",
        tool_options: "dict | None" = None,
        store: "SessionStore | None" = None,
    ) -> None:
        self.console = Console()
        self.agent_name = agent_name
        self._needs_newline = False  # track whether streamed text left us mid-line
        # Serialize confirmation prompts: with parallel_tools, two tools can call
        # confirm() at once — without this their console.input threads would race
        # on stdin and interleave prompts.
        self._confirm_lock = asyncio.Lock()
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

    def attach(self, tool_options: dict) -> None:
        """Rebind to a new agent's options after a session switch.

        Session allowlists and usage counters belong to the session they were
        granted in, so they do not carry over.
        """
        self._tool_options = tool_options
        self._turn_usage = _UsageTotals()
        self._session_usage = _UsageTotals()

    def take_session_switch(self) -> SessionSwitch | None:
        """Return and clear a pending ``/new`` or ``/resume`` request."""
        switch, self._session_switch = self._session_switch, None
        return switch

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
                self._break_line()
                self.console.print("[yellow]■ stopping… (Ctrl-C again to quit)[/]")

        def handler(signum: int, frame: FrameType | None) -> None:
            nonlocal pressed
            if pressed and callable(previous):
                previous(signum, frame)
                return
            pressed = True
            # Signal handlers must not touch the loop directly.
            loop.call_soon_threadsafe(request_stop)

        signal.signal(signal.SIGINT, handler)
        try:
            yield
        finally:
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
            self.console.print(_HELP)
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
                self.console.print("[yellow]usage: /resume <id-prefix>[/]")
                return True
            from lingcore.errors import SessionError

            try:
                meta = self._store.resolve_prefix(rest[0])
            except SessionError as e:
                self.console.print(f"[yellow]{escape(str(e))}[/]")
                return True
            self._session_switch = SessionSwitch(meta.id)
            return None
        return False

    def _list_sessions(self) -> None:
        if self._store is None:
            self.console.print("[dim]sessions are disabled for this profile[/]")
            return
        sessions = self._store.list()
        if not sessions:
            self.console.print("[dim]no stored sessions[/]")
            return
        for meta in sessions[:20]:
            self.console.print(
                f"  [cyan]{meta.id[:8]}[/] {escape(meta.title or '(untitled)')} "
                f"[dim]· {meta.message_count} msgs · {rel_time(meta.updated_at)}[/]"
            )
        if len(sessions) > 20:
            self.console.print(f"[dim]  … {len(sessions) - 20} more[/]")

    async def read_input(self) -> str | UserInput | None:
        prompt = "\n[bold cyan]you ›[/] "
        while True:
            if self._stale_input is not None:
                self.console.print(
                    "[dim](the stopped prompt is still waiting — press Enter)[/]"
                )
            try:
                line = await self._input(prompt)
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye[/]")
                return None
            if line.lstrip().startswith("/"):
                handled = self._command(line.strip())
                if handled is None:
                    return None
                if handled:
                    continue
            break
        try:
            incoming, warnings = _parse_attachments(line)
        except Exception as e:
            # Parsing must never lose the user's line; fall back to plain text.
            self.console.print(f"[red]attachment error:[/] {escape(str(e))}")
            return line
        for warning in warnings:
            self.console.print(f"[yellow]{escape(warning)}[/]")
        for attachment in incoming.attachments:
            self.console.print(
                f"[dim]attached {escape(attachment.name or attachment.media_type)}"
                f" ({escape(attachment.media_type)})[/]"
            )
        return incoming if incoming.attachments else line

    def render(self, event: AgentEvent) -> None:
        match event:
            case TextDelta(text):
                if not self._needs_newline:
                    self.console.print(f"[bold green]{self.agent_name} ›[/] ", end="")
                    self._needs_newline = True
                # escape(): streamed model text must never be parsed as Rich
                # markup, or a model could inject terminal styling/spoofing (or
                # crash the render on an unbalanced "[").
                self.console.print(escape(text), end="", soft_wrap=True)
            case ToolCallStarted(call):
                self._break_line()
                self._render_call(call)
            case ToolResultEvent(result):
                if result.ok and result.name == "todo_write":
                    return  # shown as the TodoUpdated checklist instead
                status = "[dim]" if result.ok else "[red]"
                self.console.print(
                    f"{status}← {escape(result.name)}: "
                    f"{escape(_short(result.content))}[/]"
                )
            case SkillActivated(name, active):
                self._break_line()
                verb = "activated" if active else "deactivated"
                self.console.print(f"[dim]⚙ skill {verb}: {escape(name)}[/]")
            case Compacted(summarized_messages, before_tokens, after_tokens):
                self._break_line()
                self.console.print(
                    f"[dim]⊞ compacted {summarized_messages} earlier message(s) → "
                    f"summary (~{before_tokens} → ~{after_tokens} tokens)[/]"
                )
            case StreamRetry(attempt, max_attempts, reason, discarded_chars):
                self._break_line()
                note = " — partial reply above discarded" if discarded_chars else ""
                self.console.print(
                    f"[yellow]⟲ {escape(reason)}; "
                    f"retrying ({attempt}/{max_attempts}){note}[/]"
                )
            case TurnCancelled(reason):
                self._break_line()
                self.console.print(f"[yellow]■ {escape(reason)}[/]")
                self._end_turn()
            case TodoUpdated(todos):
                self._break_line()
                self._render_todos(todos)
            case UsageReported(usage):
                self._turn_usage.add(usage)
                self._session_usage.add(usage)
            case Final(_):
                self._break_line()
                self._end_turn()
            case Error(message):
                self._break_line()
                self.console.print(f"[bold red]error:[/] {escape(message)}")
                self._end_turn()

    def _end_turn(self) -> None:
        """Print the turn's usage footer once, then reset the turn counter."""
        if self._turn_usage.requests:
            self.console.print(f"[dim]  ↳ {self._turn_usage.describe()}[/]")
        self._turn_usage = _UsageTotals()

    def _render_todos(self, todos: "tuple[TodoItem, ...]") -> None:
        if not todos:
            self.console.print("[dim]☐ todo list cleared[/]")
            return
        done = sum(1 for item in todos if item.status == "completed")
        self.console.print(f"[dim]☐ todos {done}/{len(todos)}[/]")
        for item in todos:
            text = escape(item.content)
            if item.status == "completed":
                self.console.print(f"[dim]    ✓ [strike]{text}[/strike][/]")
            elif item.status == "in_progress":
                self.console.print(f"    [bold yellow]▸ {text}[/]")
            else:
                self.console.print(f"[dim]    ○ {text}[/]")

    def _render_call(self, call: ToolCall) -> None:
        if call.name == "todo_write":
            return  # the resulting TodoUpdated checklist is the useful view
        path = call.arguments.get("path")
        diff = _diff_lines(call)
        if diff is not None and isinstance(path, str):
            self.console.print(f"[dim]→ {escape(call.name)}[/] [bold]{escape(path)}[/]")
            for line in diff[:_DIFF_PREVIEW_LINES]:
                self.console.print("    " + _style_diff_line(line), soft_wrap=True)
            if len(diff) > _DIFF_PREVIEW_LINES:
                self.console.print(
                    f"[dim]    … {len(diff) - _DIFF_PREVIEW_LINES} more diff lines[/]"
                )
            return
        content = call.arguments.get("content")
        if (
            call.name == "write_file"
            and isinstance(path, str)
            and isinstance(content, str)
        ):
            lines = len(content.splitlines())
            self.console.print(
                f"[dim]→ write_file[/] [bold]{escape(path)}[/] "
                f"[dim]({lines} line{'s' if lines != 1 else ''})[/]"
            )
            return
        if call.name == "run_shell" and isinstance(call.arguments.get("command"), str):
            self.console.print(
                f"[dim]→ run_shell[/] [bold]$ "
                f"{escape(_short(call.arguments['command'], 300))}[/]"
            )
            return
        self.console.print(
            f"[dim]→ {escape(call.name)}({escape(_short(str(call.arguments)))})[/]"
        )

    def _break_line(self) -> None:
        if self._needs_newline:
            self.console.print()
            self._needs_newline = False

    def show_resume(
        self, meta: "SessionMeta", messages: "list[Message]", tail: int = 6
    ) -> None:
        """Print a resume banner plus a dim replay of the last few messages.

        Composition-root UI (called before the session loop starts), so it is
        not part of the ``Frontend`` protocol.
        """
        title = escape(meta.title or "(untitled)")
        self.console.print(
            f'[dim]resumed[/] [cyan]{meta.id[:8]}[/] [dim]· "{title}" · '
            f"{meta.message_count} stored messages · last active {rel_time(meta.updated_at)}[/]"
        )
        shown = messages[-tail:]
        if len(messages) > len(shown):
            self.console.print(
                f"[dim]  … {len(messages) - len(shown)} earlier messages omitted …[/]"
            )
        for m in shown:
            if m.role == "user":
                summary = _attachment_summary(m)
                if m.name == "media":
                    self.console.print(f"[dim]  ↥ media{escape(summary)}[/]")
                elif m.name == "summary":
                    self.console.print(
                        f"[dim]  ≋ earlier summary › {escape(_short(m.content))}[/]"
                    )
                else:
                    text = m.input_text if m.input_text is not None else m.content
                    self.console.print(
                        f"[dim]  you › {escape(_short(text))}{escape(summary)}[/]"
                    )
            elif m.role == "assistant":
                if m.content:
                    self.console.print(
                        f"[dim]  {self.agent_name} › {escape(_short(m.content))}[/]"
                    )
                for tc in m.tool_calls:
                    # Tool names are model-generated (an unknown-tool call is
                    # stored verbatim) — escape them like any other model text.
                    self.console.print(
                        f"[dim]  → {escape(tc.name)}({escape(_short(str(tc.arguments)))})[/]"
                    )
            else:  # tool result
                self.console.print(
                    f"[dim]  ← {escape(m.name or '')}: {escape(_short(m.content))}[/]"
                )

    async def confirm(self, command: str) -> bool:
        # One prompt at a time: parallel tool calls must not race on stdin.
        async with self._confirm_lock:
            self._break_line()
            pattern = allowlist_pattern_for(command)
            session_choice = (
                f"[yellow][A][/] always allow [bold]{escape(pattern)}[/] this session"
                if pattern
                else "[dim][A] (not available: command uses shell control syntax)[/]"
            )
            prompt = (
                f"[yellow]run shell command?[/] [bold]{escape(command)}[/]\n"
                f"[yellow]  [a][/] allow once (default)  {session_choice}  "
                "[yellow][d][/] deny\n"
                "[yellow]choice [a/A/d]:[/] "
            )
            answer = (await self._input(prompt)).strip()
            if answer == "A":
                # Persist approval for this exact token prefix for the rest of the session.
                run_shell_opts = self._tool_options.setdefault("run_shell", {})
                patterns: list[str] = run_shell_opts.setdefault("allow_patterns", [])
                if not pattern:
                    self.console.print(
                        "[dim]command was not added to session allowlist[/]"
                    )
                    return True
                if pattern not in patterns:
                    patterns.append(pattern)
                    self.console.print(
                        f"[dim]added {escape(pattern)!r} to session allowlist[/]"
                    )
                return True
            # Empty / Enter, "a", "y", "yes" all mean allow once.
            return answer.lower() in {"", "a", "y", "yes"}
