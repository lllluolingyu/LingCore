"""A Rich-based terminal frontend.

Renders streamed text live, surfaces tool calls/results dimly, and routes
shell-command confirmation prompts to the terminal. Blocking console input
runs in a worker thread so the asyncio event loop is never stalled.
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    SkillActivated,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.tools.builtin.shell import allowlist_pattern_for

_EXIT_COMMANDS = {"/exit", "/quit", "/q"}


def _short(text: str, limit: int = 200) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit] + " …"


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

    async def read_input(self) -> str | None:
        prompt = "\n[bold cyan]you ›[/] "
        try:
            line = await asyncio.to_thread(self.console.input, prompt)
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n[dim]bye[/]")
            return None
        if line.strip() in _EXIT_COMMANDS:
            return None
        return line

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
            case Final(_):
                self._break_line()
            case Error(message):
                self._break_line()
                self.console.print(f"[bold red]error:[/] {message}")

    def _break_line(self) -> None:
        if self._needs_newline:
            self.console.print()
            self._needs_newline = False

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
