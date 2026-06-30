"""A Rich-based terminal frontend.

Renders streamed text live, surfaces tool calls/results dimly, and routes
shell-command confirmation prompts to the terminal. Blocking console input
runs in a worker thread so the asyncio event loop is never stalled.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

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
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.media import attachment_from_path
from lingcore.media_types import (
    MAX_ATTACHMENTS,
    TOTAL_ATTACHMENT_MAX_BYTES,
    decoded_payload_size,
)
from lingcore.message import Attachment, UserInput
from lingcore.tools.builtin.shell import allowlist_pattern_for

if TYPE_CHECKING:
    from lingcore.message import Message
    from lingcore.sessions import SessionMeta

_EXIT_COMMANDS = {"/exit", "/quit", "/q"}
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
        out.append(line[pos:match.start()])
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
    ) -> None:
        self.console = Console()
        self.agent_name = agent_name
        self._needs_newline = False  # track whether streamed text left us mid-line
        # Shared with the agent's ToolContext so "allow always" writes land live.
        self._tool_options: dict = tool_options if tool_options is not None else {}

    async def read_input(self) -> str | UserInput | None:
        prompt = "\n[bold cyan]you ›[/] "
        try:
            line = await asyncio.to_thread(self.console.input, prompt)
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n[dim]bye[/]")
            return None
        if line.strip() in _EXIT_COMMANDS:
            return None
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
                self.console.print(text, end="", soft_wrap=True)
            case ToolCallStarted(call):
                self._break_line()
                self.console.print(f"[dim]→ {call.name}({_short(str(call.arguments))})[/]")
            case ToolResultEvent(result):
                status = "[dim]" if result.ok else "[red]"
                self.console.print(f"{status}← {result.name}: {_short(result.content)}[/]")
            case SkillActivated(name, active):
                self._break_line()
                verb = "activated" if active else "deactivated"
                self.console.print(f"[dim]⚙ skill {verb}: {name}[/]")
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
            case Final(_):
                self._break_line()
            case Error(message):
                self._break_line()
                self.console.print(f"[bold red]error:[/] {message}")

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
            f"[dim]resumed[/] [cyan]{meta.id[:8]}[/] [dim]· \"{title}\" · "
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
                else:
                    self.console.print(
                        f"[dim]  you › {escape(_short(m.content))}{escape(summary)}[/]"
                    )
            elif m.role == "assistant":
                if m.content:
                    self.console.print(
                        f"[dim]  {self.agent_name} › {escape(_short(m.content))}[/]"
                    )
                for tc in m.tool_calls:
                    self.console.print(
                        f"[dim]  → {tc.name}({escape(_short(str(tc.arguments)))})[/]"
                    )
            else:  # tool result
                self.console.print(f"[dim]  ← {m.name}: {escape(_short(m.content))}[/]")

    async def confirm(self, command: str) -> bool:
        self._break_line()
        prompt = (
            f"[yellow]run shell command?[/] [bold]{command}[/]\n"
            "[yellow]  [a][/] allow once (default)  [yellow][A][/] allow always  [yellow][d][/] deny\n"
            "[yellow]choice [a/A/d]:[/] "
        )
        answer = (await asyncio.to_thread(self.console.input, prompt)).strip()
        if answer == "A":
            # Persist approval for this exact token prefix for the rest of the session.
            run_shell_opts = self._tool_options.setdefault("run_shell", {})
            patterns: list[str] = run_shell_opts.setdefault("allow_patterns", [])
            pattern = allowlist_pattern_for(command)
            if not pattern:
                self.console.print("[dim]command was not added to session allowlist[/]")
                return True
            if pattern not in patterns:
                patterns.append(pattern)
                self.console.print(
                    f"[dim]added {pattern!r} to session allowlist[/]"
                )
            return True
        # Empty / Enter, "a", "y", "yes" all mean allow once.
        return answer.lower() in {"", "a", "y", "yes"}
