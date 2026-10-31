"""A Rich-based terminal frontend.

Renders streamed replies as markdown (completed blocks are committed to the
scrollback while the block in progress previews live), shows a spinner while
the model or a tool is working, summarizes tool calls/results in one line
each, and routes shell-command confirmation prompts to the terminal.
On a real terminal, input is a prompt_toolkit editor (multi-line with
Shift/Alt+Enter, history, slash-command/``@path`` completion) with a status
bar; elsewhere (pipes, tests) blocking console input runs in a worker thread
so the asyncio event loop is never stalled.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from rich.console import Console, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from lingcore.events import (
    AgentEvent,
)
from lingcore.io._cli import confirm as confirmations
from lingcore.io._cli import input as terminal_input
from lingcore.io._cli import render as rendering
from lingcore.io._cli.attachments import _parse_attachments as _parse_attachments
from lingcore.io._cli.definitions import (
    _THEME,
)
from lingcore.io._cli.definitions import COMMAND_TOOLS as COMMAND_TOOLS
from lingcore.io._cli.editor import _completions as _completions
from lingcore.io._cli.editor import _input_key_bindings as _input_key_bindings
from lingcore.io._cli.editor import _InputCompleter as _InputCompleter
from lingcore.io._cli.presentation import SessionSwitch as SessionSwitch
from lingcore.io._cli.presentation import _stable_prefix as _stable_prefix
from lingcore.io._cli.presentation import (
    _UsageTotals,
    _Working,
)
from lingcore.io._cli.presentation import rel_time as rel_time
from lingcore.io._cli.presentation import session_table as session_table
from lingcore.message import ToolCall, ToolResult, UserInput
from lingcore.plugins.commands import CommandCatalog
from lingcore.todos import TodoItem
from lingcore.usage import TurnStats

if TYPE_CHECKING:
    from lingcore.message import Message
    from lingcore.sessions import SessionMeta, SessionStore


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
        self._segment_text = ""
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
        # This turn's process_start commands: shell-framed too, but never
        # "always allow", which writes run_shell's allowlist.
        self._process_commands: set[str] = set()
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
        self._turn_stats: TurnStats | None = None
        self._session_switch: SessionSwitch | None = None
        # The running turn's stop request while interrupt_scope is active (the
        # prompt_toolkit confirm prompt sees Ctrl-C as a key, not a signal).
        self._request_stop: Callable[[], None] | None = None
        self._tty = sys.stdin.isatty() and sys.stdout.isatty()
        self._main_prompt: PromptSession[str] | None = None
        self._confirm_prompt: PromptSession[str] | None = None
        self._input_draft = Document()
        # Text the editor accepted in this prompt, and an accepted message a
        # monitor wakeup cancelled before prompt_async could return it.
        self._accepted: str | None = None
        self._pending_submit: str | None = None

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
        self._turn_stats = None
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
        return terminal_input._sessions(self)

    def _track_accept(self, session: PromptSession[str]) -> None:
        """Record each accepted message before prompt_async returns it."""
        return terminal_input._track_accept(self, session)

    def _prompt_message(self) -> StyleAndTextTuples:
        return terminal_input._prompt_message(self)

    def _status_bar(self) -> StyleAndTextTuples:
        return terminal_input._status_bar(self)

    def _status_fragments(self, width: int) -> StyleAndTextTuples:
        """The status bar: who/what/where on the left, key hints on the right.

        Items are dropped from the end (hints first) until the line fits.
        """
        return terminal_input._status_fragments(self, width)

    def _ansi(self, markup: str) -> ANSI:
        """Rich markup rendered to ANSI for a prompt_toolkit prompt."""
        return terminal_input._ansi(self, markup)

    async def _read_message(self) -> str:
        return await terminal_input._read_message(self)

    @property
    def supports_background(self) -> bool:
        return self._rich_prompt

    def background_received(self, incoming: UserInput) -> None:
        self.console.print(Text("Monitor: " + incoming.text, style="dim"))
        self._start_turn()

    def background_paused(self) -> None:
        self.console.print(
            "[dim]monitor events paused until your next message "
            "(ask to process_stop a watch to end it)[/]"
        )

    async def _input(self, prompt: str) -> str:
        """``console.input`` in a worker thread, safe against cancellation."""
        return await terminal_input._input(self, prompt)

    @contextmanager
    def interrupt_scope(self, stop: Callable[[], bool]) -> Iterator[None]:
        """Make Ctrl-C stop the running turn instead of quitting.

        A second Ctrl-C while the stop is pending falls through to the previous
        handler (asyncio's, which ends the process). Only the main thread can
        own signal handlers; elsewhere the scope is a no-op.
        """
        with terminal_input.interrupt_scope(self, stop):
            yield

    def _command(self, line: str) -> bool | None:
        """Handle a slash command. ``True`` = handled, ``None`` = end session,
        ``False`` = not a command (send as a message)."""
        return terminal_input._command(self, line)

    def _print_help(self) -> None:
        return terminal_input._print_help(self)

    def _list_sessions(self) -> None:
        return terminal_input._list_sessions(self)

    async def read_input(self) -> str | UserInput | None:
        return await terminal_input.read_input(self)

    # --- live region ----------------------------------------------------

    @property
    def _interactive(self) -> bool:
        console = self.console
        return (
            console.is_terminal and not console.is_dumb_terminal and not console.quiet
        )

    def _show_live(self, renderable: RenderableType) -> None:
        return rendering._show_live(self, renderable)

    def _stop_live(self) -> None:
        return rendering._stop_live(self)

    def _show_working(self, label: str) -> None:
        return rendering._show_working(self, label)

    def _start_turn(self) -> None:
        return rendering._start_turn(self)

    # --- streamed reply -------------------------------------------------

    def _reply_block(self, text: str, *, first: bool, tail: bool = False) -> Table:
        """A reply block with the ``●`` marker gutter on a segment's first block."""
        return rendering._reply_block(self, text, first=first, tail=tail)

    def _open_segment(self) -> None:
        return rendering._open_segment(self)

    def _commit(self, text: str) -> None:
        return rendering._commit(self, text)

    def _feed(self, text: str) -> None:
        return rendering._feed(self, text)

    def _flush(self) -> None:
        """Commit any reply text in progress and close the segment."""
        return rendering._flush(self)

    def _discard(self) -> bool:
        """Drop reply text in progress; report whether any was committed."""
        return rendering._discard(self)

    def _note(self, markup: str) -> None:
        """A status line between transcript items."""
        return rendering._note(self, markup)

    # --- events ---------------------------------------------------------

    def render(self, event: AgentEvent) -> None:
        return rendering.render(self, event)

    def _end_turn(self) -> None:
        """Print the turn's footer (time + usage) once, then reset the turn."""
        return rendering._end_turn(self)

    def _tool_gap(self) -> None:
        """Separate a tool run from the user's message or reply text before it."""
        return rendering._tool_gap(self)

    def _render_todos(self, todos: "tuple[TodoItem, ...]") -> None:
        return rendering._render_todos(self, todos)

    def _render_call(self, call: ToolCall) -> None:
        return rendering._render_call(self, call)

    def _render_result(self, result: ToolResult) -> None:
        return rendering._render_result(self, result)

    # --- composition-root UI ---------------------------------------------

    def show_banner(
        self, *, model: str, workspace: object, notice: str | None = None
    ) -> None:
        """Print the startup header. Composition-root UI, not ``Frontend``."""
        return rendering.show_banner(
            self, model=model, workspace=workspace, notice=notice
        )

    def show_resume(
        self, meta: "SessionMeta", messages: "list[Message]", tail: int = 6
    ) -> None:
        """Print a resume banner plus a dim replay of the last few messages.

        Composition-root UI (called before the session loop starts), so it is
        not part of the ``Frontend`` protocol.
        """
        return rendering.show_resume(self, meta, messages, tail)

    def _replay(self, line: str, style: str = "") -> None:
        return rendering._replay(self, line, style)

    async def confirm(self, command: str) -> bool:
        # One prompt at a time: parallel tool calls must not race on stdin.
        return await confirmations.confirm(self, command)

    async def _ask(self, prompt: str) -> str | None:
        """Read one confirmation answer; ``None`` when Ctrl-C/Ctrl-D denied it."""
        return await confirmations._ask(self, prompt)

    async def _confirm_action(self, request: str) -> bool:
        return await confirmations._confirm_action(self, request)

    async def _confirm_process(self, command: str) -> bool:
        return await confirmations._confirm_process(self, command)

    async def _confirm_shell(self, command: str) -> bool:
        return await confirmations._confirm_shell(self, command)
