"""Terminal frontend input behavior."""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.history import InMemoryHistory
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from lingcore.io._cli.attachments import _parse_attachments
from lingcore.io._cli.definitions import (
    _COMMANDS,
    _CONTINUATION_PROMPT,
    _EXIT_COMMANDS,
    _HELP_COMMANDS,
    _HELP_KEYS,
    _PROMPT,
    _PROMPT_STYLE,
)
from lingcore.io._cli.editor import _continuation, _input_key_bindings, _InputCompleter
from lingcore.io._cli.presentation import SessionSwitch, _tokens, session_table
from lingcore.message import UserInput

if TYPE_CHECKING:
    from lingcore.io.cli import CLIFrontend


def _sessions(self: CLIFrontend) -> tuple[PromptSession[str], PromptSession[str]]:
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
        self._track_accept(self._main_prompt)
        # Separate history: confirmation answers are not messages.
        self._confirm_prompt = PromptSession(style=_PROMPT_STYLE)
    return self._main_prompt, self._confirm_prompt


def _track_accept(self: CLIFrontend, session: PromptSession[str]) -> None:
    """Record each accepted message before prompt_async returns it."""
    accept = session.default_buffer.accept_handler

    def handler(buffer: Buffer) -> bool:
        self._accepted = buffer.text
        return accept(buffer) if accept is not None else False

    session.default_buffer.accept_handler = handler


def _prompt_message(self: CLIFrontend) -> StyleAndTextTuples:
    width = get_app().output.get_size().columns
    return [
        ("", "\n"),
        ("class:rule", "─" * width + "\n"),
        ("class:prompt", "❯ "),
    ]


def _status_bar(self: CLIFrontend) -> StyleAndTextTuples:
    return self._status_fragments(get_app().output.get_size().columns)


def _status_fragments(self: CLIFrontend, width: int) -> StyleAndTextTuples:
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


def _ansi(self: CLIFrontend, markup: str) -> ANSI:
    """Rich markup rendered to ANSI for a prompt_toolkit prompt."""
    with self.console.capture() as capture:
        self.console.print(markup, end="")
    return ANSI(capture.get())


async def _read_message(self: CLIFrontend) -> str:
    if not self._rich_prompt:
        line = await self._input(_PROMPT)
        # A trailing backslash continues the message on the next line.
        while line.endswith("\\"):
            line = line[:-1] + "\n" + await self._input(_CONTINUATION_PROMPT)
        return line
    self._stop_live()
    if self._pending_submit is not None:
        line, self._pending_submit = self._pending_submit, None
        self.console.print(Text("❯ " + line, style="dim"))
        return line
    main, _ = self._sessions()
    draft, self._input_draft = self._input_draft, Document()
    self._accepted = None
    try:
        return await main.prompt_async(default=draft, handle_sigint=False)
    except asyncio.CancelledError:
        if self._accepted is not None:
            # Enter landed just before the wakeup cancelled the editor:
            # send that message after the automatic turn, don't redraft it.
            self._pending_submit = self._accepted
        else:
            # A monitor wakes the Agent while the editor is open. Restore
            # the unsent text and cursor after the automatic turn finishes.
            self._input_draft = main.default_buffer.document
        raise
    finally:
        self._accepted = None


async def _input(self: CLIFrontend, prompt: str) -> str:
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
def interrupt_scope(self: CLIFrontend, stop: Callable[[], bool]) -> Iterator[None]:
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


def _command(self: CLIFrontend, line: str) -> bool | None:
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
                "[yellow]usage: /resume <id-prefix>[/] [dim](see /sessions for ids)[/]"
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


def _print_help(self: CLIFrontend) -> None:
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


def _list_sessions(self: CLIFrontend) -> None:
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


async def read_input(self: CLIFrontend) -> str | UserInput | None:
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
