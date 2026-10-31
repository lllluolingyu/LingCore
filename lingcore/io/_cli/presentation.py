"""Terminal presentation helpers."""

from __future__ import annotations

import difflib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown
from rich.markup import escape
from rich.segment import Segment
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lingcore.io._cli.definitions import _FENCE_RE, _HEADLINE_KEYS, COMMAND_TOOLS
from lingcore.message import ToolCall, ToolResult
from lingcore.usage import TokenUsage, TurnStats

if TYPE_CHECKING:
    from lingcore.message import Message
    from lingcore.sessions import SessionMeta


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


def _rate(stats: TurnStats | None) -> str:
    """Output tokens per second of generation time, as LingCore measured it."""
    if stats is None or not stats.model_ms:
        return ""
    output = sum(usage.output_tokens for usage in stats.usage)
    if not output:
        return ""
    rate = output / (stats.model_ms / 1000)
    return f"{rate:.0f} tok/s" if rate >= 10 else f"{rate:.1f} tok/s"


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
    command_tool = call.name in COMMAND_TOOLS
    text = Text(no_wrap=not command_tool, overflow="ellipsis")
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
        text.append(f"$ {value}" if command_tool else value)
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
