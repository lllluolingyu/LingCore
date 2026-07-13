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
import shlex

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._offload import DEFAULT_OFFLOAD_OVER_CHARS, offload_text

_MAX_OUTPUT_CHARS = 16_000
_DEFAULT_TIMEOUT = 60
# Hard ceiling on bytes retained in memory while reading a command's output.
# The reader keeps draining the pipe past this (so the child never blocks on a
# full pipe) but discards the overflow, so a runaway command can't exhaust
# memory before the timeout fires.
_MAX_CAPTURE_BYTES = 10 * 1024 * 1024
_SHELL_CONTROL_TOKENS = (
    ";", "&&", "&", "||", "|", "<", ">", "\n", "\r", "`", "$(", "${", "(", ")"
)


class ShellArgs(BaseModel):
    command: str = Field(description="The shell command to execute in the workspace.")


def _has_shell_control(command: str) -> bool:
    return any(token in command for token in _SHELL_CONTROL_TOKENS)


def _split_command(text: str) -> list[str] | None:
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return None


def allowlist_pattern_for(command: str) -> str:
    """Return the safest reusable allowlist pattern for a confirmed command."""
    stripped = command.strip()
    if _has_shell_control(stripped):
        return ""
    parts = _split_command(stripped)
    if not parts:
        return ""
    return " ".join(shlex.quote(p) for p in parts)


def _matches_allowlist(command: str, patterns: list[str]) -> bool:
    """Return True only for simple commands matching an allowlisted pattern.

    A *multi-token* pattern (e.g. ``git status``) matches that command plus any
    trailing arguments (prefix match), so the operator can allow a specific
    argument-bearing form. A *single-token* pattern (a bare program name like
    ``cat`` or ``ls``) matches ONLY the exact bare command with no arguments:
    a bare program name must never silently authorize arbitrary arguments
    (``cat`` in the allowlist must not green-light ``cat ~/.ssh/id_rsa``). To
    allow an argument-bearing invocation, the operator lists that specific form.
    """
    stripped = command.strip()
    if _has_shell_control(stripped):
        return False
    command_parts = _split_command(stripped)
    if not command_parts:
        return False
    for pattern in patterns:
        pattern_parts = _split_command(pattern.strip())
        if not pattern_parts:
            continue
        if len(pattern_parts) == 1:
            if command_parts == pattern_parts:  # exact: no trailing arguments
                return True
        elif command_parts[: len(pattern_parts)] == pattern_parts:
            return True
    return False


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
    max_capture = int(opts.get("max_capture_bytes", _MAX_CAPTURE_BYTES))

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
        stdout, truncated = await asyncio.wait_for(
            _read_capped(proc, max_capture), timeout=timeout
        )
    except asyncio.TimeoutError:
        await _kill_and_reap(proc)
        raise ToolError(
            f"command timed out after {timeout:g}s and was killed: {args.command!r}"
        ) from None
    except asyncio.CancelledError:
        # The turn was cancelled (e.g. the frontend disconnected mid-command).
        # Kill and reap the whole process group so no orphan keeps running,
        # then propagate the cancellation.
        await _kill_and_reap(proc)
        raise

    code = proc.returncode
    header = f"$ {args.command}\n(exit code: {code})\n"
    raw = stdout.decode("utf-8", errors="replace") if stdout else ""
    if truncated:
        raw += f"\n... (output exceeded {max_capture} bytes and was truncated)"
    if not raw:
        return header + "(no output)"
    # Heavy logs are staged to a workspace file (read the rest with read_file)
    # so they don't bloat the conversation; small output stays inline.
    body = offload_text(
        ctx,
        source="shell",
        text=raw,
        threshold=int(opts.get("offload_over_chars", DEFAULT_OFFLOAD_OVER_CHARS)),
        fallback_max_chars=int(opts.get("max_output_chars", _MAX_OUTPUT_CHARS)),
    )
    return header + body


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


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Kill the command's process group and reap it, best-effort.

    Reaping avoids a leaked zombie / event-loop warning. Used by both the
    timeout path and the cancellation path (a disconnected frontend), so a
    command whose turn is torn down never leaves an orphan behind.
    """
    _kill_tree(proc)
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


async def _read_capped(
    proc: asyncio.subprocess.Process, cap: int
) -> tuple[bytes, bool]:
    """Read the command's combined output, retaining at most ``cap`` bytes.

    Keeps draining the pipe past the cap (so the child never blocks on a full
    pipe) but stops storing the overflow, so a command that spews output can't
    exhaust memory before the wall-clock timeout fires. Returns
    ``(retained_bytes, truncated)`` and waits for the process to exit.
    """
    assert proc.stdout is not None
    buf = bytearray()
    truncated = False
    while True:
        chunk = await proc.stdout.read(65536)
        if not chunk:
            break
        remaining = cap - len(buf)
        if remaining > 0:
            buf.extend(chunk[:remaining])
            # Reaching the cap exactly is not truncation. Mark it only when
            # this chunk actually contains bytes that were not retained.
            if len(chunk) > remaining:
                truncated = True
        else:
            truncated = True
    await proc.wait()
    return bytes(buf), truncated
