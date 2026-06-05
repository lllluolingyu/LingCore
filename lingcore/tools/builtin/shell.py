"""The run_shell tool — the coding agent's most powerful and riskiest tool.

run_shell executes an arbitrary command in the workspace directory. Unlike the
fs tools, the workspace boundary is NOT a security sandbox here: a shell
command can `cd ..`, open a network socket, or read anything the process user
can. The MVP mitigations are deliberately modest and layered:

  * cwd is set to the workspace (a convenience boundary, not a jail);
  * a wall-clock timeout kills the command (and its process group) on expiry;
  * stdout/stderr are captured and truncated to a sane size;
  * an optional confirmation gate (ctx.confirm) lets the frontend require a
    human yes/no before each command runs.

True isolation (containers, seccomp, user namespaces) is a deliberate
post-MVP concern; this module is the plug point for it.
"""

from __future__ import annotations

import asyncio
import os
import signal

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool

_MAX_OUTPUT_CHARS = 16_000
_DEFAULT_TIMEOUT = 60


class ShellArgs(BaseModel):
    command: str = Field(description="The shell command to execute in the workspace.")


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    head = text[: _MAX_OUTPUT_CHARS]
    return f"{head}\n... (truncated, {len(text) - _MAX_OUTPUT_CHARS} more chars)"


def _matches_allowlist(command: str, patterns: list[str]) -> bool:
    """Return True if command starts with any allowlisted prefix."""
    return any(command.strip().startswith(p) for p in patterns)


@tool(
    description=(
        "Run a shell command in the workspace directory and return its combined "
        "stdout/stderr and exit code. Use for builds, tests, git, and inspection. "
        "Commands run with a timeout and may require user confirmation."
    )
)
async def run_shell(args: ShellArgs, ctx: ToolContext) -> str:
    opts = ctx.options.get("run_shell", {}) if ctx.options else {}
    timeout = float(opts.get("timeout", _DEFAULT_TIMEOUT))
    require_confirmation = bool(opts.get("require_confirmation", True))
    allow_patterns: list[str] = opts.get("allow_patterns", [])

    needs_confirm = require_confirmation and not _matches_allowlist(
        args.command, allow_patterns
    )

    if needs_confirm:
        if ctx.confirm is None:
            raise ToolError(
                "run_shell requires confirmation but no confirmation handler is "
                "available on this frontend; command refused"
            )
        approved = await ctx.confirm(args.command)
        if not approved:
            raise ToolError(f"user declined to run command: {args.command!r}")

    # start_new_session=True puts the child in its own process group so a
    # timeout can kill the whole tree, not just the shell.
    try:
        proc = await asyncio.create_subprocess_shell(
            args.command,
            cwd=str(ctx.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as e:
        raise ToolError(f"failed to launch command: {e}") from None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_tree(proc)
        # Reap the killed process so we don't leak a zombie / warning.
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        raise ToolError(
            f"command timed out after {timeout:g}s and was killed: {args.command!r}"
        ) from None

    output = _truncate(stdout.decode("utf-8", errors="replace")) if stdout else ""
    code = proc.returncode
    header = f"$ {args.command}\n(exit code: {code})\n"
    return header + (output if output else "(no output)")


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the process group of a timed-out command."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # Fall back to killing just the child if the group is gone/inaccessible.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
