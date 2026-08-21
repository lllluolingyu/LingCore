"""The run_shell tool — confirmation, supervision, and bounded output.

Profiles may select a fail-closed sandbox backend through
``tool_options.run_shell.sandbox``.  Omitting that block preserves the legacy
host runner for compatibility; the result header always says which runner was
used so the distinction cannot be mistaken.
"""

from __future__ import annotations

import asyncio
import shlex

from pydantic import BaseModel, Field, ValidationError

from lingcore.errors import ToolError
from lingcore.sandbox import launch_shell, parse_shell_options
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._offload import offload_text

_SHELL_CONTROL_TOKENS = (
    ";",
    "&&",
    "&",
    "||",
    "|",
    "<",
    ">",
    "\n",
    "\r",
    "`",
    "$(",
    "${",
    "(",
    ")",
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
    raw_options = ctx.options.get("run_shell", {}) if ctx.options else {}
    try:
        options = parse_shell_options(raw_options)
    except (ValidationError, ValueError) as exc:
        raise ToolError(f"invalid run_shell options: {exc}") from None

    needs_confirm = options.require_confirmation and not _matches_allowlist(
        args.command, options.allow_patterns
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

    execution = await launch_shell(
        args.command,
        workspace=ctx.workspace,
        options=options,
        getenv=ctx.getenv,
    )

    try:
        stdout, truncated = await asyncio.wait_for(
            _read_capped(execution.process, options.max_capture_bytes),
            timeout=options.timeout,
        )
    except asyncio.TimeoutError:
        await asyncio.shield(execution.abort())
        raise ToolError(
            f"command timed out after {options.timeout:g}s and was killed "
            f"by the {execution.runner} runner: {args.command!r}"
        ) from None
    except asyncio.CancelledError:
        # The turn was cancelled (e.g. the frontend disconnected mid-command).
        # Kill and reap the process tree/container, then propagate cancellation.
        await asyncio.shield(execution.abort())
        raise
    except BaseException:
        await asyncio.shield(execution.abort())
        raise

    code = await execution.finish()
    header = f"$ {args.command}\n(runner: {execution.runner})\n(exit code: {code})\n"
    raw = stdout.decode("utf-8", errors="replace") if stdout else ""
    if truncated:
        raw += (
            f"\n... (output exceeded {options.max_capture_bytes} bytes "
            "and was truncated)"
        )
    if not raw:
        return header + "(no output)"
    # Heavy logs are staged to a workspace file (read the rest with read_file)
    # so they don't bloat the conversation; small output stays inline.
    body = offload_text(
        ctx,
        source="shell",
        text=raw,
        threshold=options.offload_over_chars,
        fallback_max_chars=options.max_output_chars,
    )
    return header + body


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
