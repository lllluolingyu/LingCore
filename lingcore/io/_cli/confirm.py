"""Terminal frontend confirm behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from lingcore.tools.builtin.shell import allowlist_pattern_for

if TYPE_CHECKING:
    from lingcore.io.cli import CLIFrontend


async def confirm(self: CLIFrontend, command: str) -> bool:
    # One prompt at a time: parallel tool calls must not race on stdin.
    async with self._confirm_lock:
        self._flush()
        if command in self._process_commands:
            # Checked first: "always allow" would land in run_shell's list.
            allowed = await self._confirm_process(command)
            label = "process_start" if allowed else "tools"
        elif command in self._shell_commands:
            allowed = await self._confirm_shell(command)
            label = "run_shell" if allowed else "tools"
        else:
            allowed = await self._confirm_action(command)
            label = "tools"
        self._show_working(f"running {label}")
        return allowed


async def _ask(self: CLIFrontend, prompt: str) -> str | None:
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


async def _confirm_action(self: CLIFrontend, request: str) -> bool:
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
        "  [yellow]y[/] allow once [dim](Enter)[/] · [yellow]n[/] deny\n[yellow]❯[/] "
    )
    if answer is None:
        return False
    allowed = answer.lower() in {"", "a", "y", "yes"}
    self.console.print("[dim]  allowed once[/]" if allowed else "[red]  denied[/]")
    return allowed


async def _confirm_process(self: CLIFrontend, command: str) -> bool:
    self.console.print(
        Panel(
            Text(f"$ {command}", style="bold"),
            title="[yellow]start background process?[/]",
            subtitle="[dim]keeps running until stopped[/]",
            title_align="left",
            subtitle_align="left",
            border_style="yellow",
            expand=False,
        )
    )
    answer = await self._ask(
        "  [yellow]y[/] allow once [dim](Enter)[/] · [yellow]n[/] deny\n[yellow]❯[/] "
    )
    if answer is None:
        return False
    allowed = answer.lower() in {"", "a", "y", "yes"}
    self.console.print("[dim]  allowed once[/]" if allowed else "[red]  denied[/]")
    return allowed


async def _confirm_shell(self: CLIFrontend, command: str) -> bool:
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
