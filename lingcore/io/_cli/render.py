"""Terminal frontend render behavior."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lingcore.errors import INCOMPLETE_FINISH_REASONS, incomplete_response_message
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
    TurnStatsReported,
    UsageReported,
)
from lingcore.io._cli.definitions import _DIFF_PREVIEW_LINES, _ERROR_PREVIEW_LINES
from lingcore.io._cli.presentation import (
    _attachment_summary,
    _call_text,
    _diff_lines,
    _duration,
    _markdown,
    _plural,
    _rate,
    _result_text,
    _short,
    _stable_prefix,
    _style_diff_line,
    _Tail,
    _tilde,
    _tokens,
    _UsageTotals,
    rel_time,
)
from lingcore.message import ToolCall, ToolResult
from lingcore.todos import TodoItem

if TYPE_CHECKING:
    from lingcore.io.cli import CLIFrontend
    from lingcore.message import Message
    from lingcore.sessions import SessionMeta


def _show_live(self: CLIFrontend, renderable: RenderableType) -> None:
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


def _stop_live(self: CLIFrontend) -> None:
    if self._live is not None:
        live, self._live = self._live, None
        live.stop()


def _show_working(self: CLIFrontend, label: str) -> None:
    self._working.label = label
    self._show_live(self._working)


def _start_turn(self: CLIFrontend) -> None:
    self._segment_text = ""
    self._segment_committed = False
    self._turn_started = time.monotonic()
    self._working.started = self._turn_started
    self._last_kind = None
    self._shell_commands.clear()
    self._process_commands.clear()
    self._show_working("thinking")


def _reply_block(
    self: CLIFrontend, text: str, *, first: bool, tail: bool = False
) -> Table:
    """A reply block with the ``●`` marker gutter on a segment's first block."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(ratio=1)
    body: RenderableType = _markdown(text)
    marker = Text("●", style="bold green") if first else ""
    grid.add_row(marker, _Tail(body) if tail else body)
    return grid


def _open_segment(self: CLIFrontend) -> None:
    if self._segment_open:
        return
    self._stop_live()
    self.console.print()
    self._segment_open = True
    self._segment_text = ""
    self._segment_committed = False
    self._last_kind = "text"


def _commit(self: CLIFrontend, text: str) -> None:
    if not text.strip():
        return
    if self._segment_committed:
        self.console.print()
    self.console.print(self._reply_block(text, first=not self._segment_committed))
    self._segment_committed = True


def _feed(self: CLIFrontend, text: str) -> None:
    self._open_segment()
    self._segment_text += text
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


def _flush(self: CLIFrontend) -> None:
    """Commit any reply text in progress and close the segment."""
    self._stop_live()
    if self._segment_open:
        pending, self._pending = self._pending, ""
        self._commit(pending)
    self._segment_open = False


def _discard(self: CLIFrontend) -> bool:
    """Drop reply text in progress; report whether any was committed."""
    self._stop_live()
    committed = self._segment_open and self._segment_committed
    self._pending = ""
    self._segment_open = False
    return committed


def _note(self: CLIFrontend, markup: str) -> None:
    """A status line between transcript items."""
    self._flush()
    self.console.print(markup)


def render(self: CLIFrontend, event: AgentEvent) -> None:
    match event:
        case TextDelta(text):
            # Streamed model text is rendered as markdown (Rich builds Text
            # objects, never parses markup), so a model cannot inject
            # terminal styling/spoofing or crash the render on a "[".
            self._feed(text)
        case ToolCallStarted(call):
            command = call.arguments.get("command")
            if isinstance(command, str):
                if call.name == "run_shell":
                    self._shell_commands.add(command)
                elif call.name in {"process_start", "process_monitor"}:
                    self._process_commands.add(command)
            self._flush()
            self._segment_text = ""
            self._segment_committed = False
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
        case TurnStatsReported(stats):
            self._turn_stats = stats
        case Final(_):
            self._flush()
            self._end_turn()
        case Error(message, partial_content=partial):
            if partial is not None and partial != self._segment_text:
                committed = self._segment_committed
                self._discard()
                if committed:
                    self.console.print(
                        "[yellow]Partial reply above replaced after output filtering.[/]"
                    )
                if partial:
                    self._feed(partial)
            self._flush()
            self.console.print(Text.assemble(("✗ error: ", "bold red"), message))
            self._end_turn()


def _end_turn(self: CLIFrontend) -> None:
    """Print the turn's footer (time + usage) once, then reset the turn."""
    self._stop_live()
    parts = []
    if self._turn_started is not None:
        parts.append(_duration(time.monotonic() - self._turn_started))
    if self._turn_usage.requests:
        parts.append(self._turn_usage.describe())
    rate = _rate(self._turn_stats)
    if rate:
        parts.append(rate)
    if parts:
        self.console.print(f"[dim]  ↳ {' · '.join(parts)}[/]")
    self._turn_usage = _UsageTotals()
    self._turn_started = None
    self._turn_stats = None


def _tool_gap(self: CLIFrontend) -> None:
    """Separate a tool run from the user's message or reply text before it."""
    if self._last_kind != "tool":
        self.console.print()
    self._last_kind = "tool"


def _render_todos(self: CLIFrontend, todos: "tuple[TodoItem, ...]") -> None:
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


def _render_call(self: CLIFrontend, call: ToolCall) -> None:
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


def _render_result(self: CLIFrontend, result: ToolResult) -> None:
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


def show_banner(
    self: CLIFrontend, *, model: str, workspace: object, notice: str | None = None
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
    self: CLIFrontend, meta: "SessionMeta", messages: "list[Message]", tail: int = 6
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
            elif m.name == "monitor":
                events = m.content.split("\n", 1)[-1]  # skip the fixed preamble
                self._replay(f"  ◇ monitor events › {_short(events)}")
            else:
                text = m.input_text if m.input_text is not None else m.content
                self._replay(f"  ❯ {_short(text)}{summary}", style="cyan")
        elif m.role == "assistant":
            if m.content:
                self._replay(f"  ● {_short(m.content)}")
            if m.finish_reason in INCOMPLETE_FINISH_REASONS:
                warning = m.finish_detail or incomplete_response_message(
                    m.finish_reason
                )
                self.console.print(Text(f"  ⚠ {warning}", style="yellow"))
            for tc in m.tool_calls:
                # Tool names are model-generated (an unknown-tool call is
                # stored verbatim) — Text renders them literally.
                self._replay(f"  {_call_text(tc).plain}")
        else:  # tool result
            result = ToolResult(call_id="", name=m.name or "", content=m.content)
            self._replay(f"  ⎿ {result.name}: {_result_text(result).plain}")
    self.console.rule(style="dim")


def _replay(self: CLIFrontend, line: str, style: str = "") -> None:
    self.console.print(
        Text(line, style=f"dim {style}".strip(), no_wrap=True, overflow="ellipsis")
    )
