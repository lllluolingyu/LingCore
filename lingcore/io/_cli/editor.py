"""Terminal editor helpers."""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys

from lingcore.io._cli.definitions import (
    _ARGUMENT_COMMANDS,
    _COMMAND_META,
    _COMMANDS,
    _SHIFT_ENTER_SEQUENCES,
)
from lingcore.plugins.commands import CommandCatalog


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
