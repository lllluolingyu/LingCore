"""Tests for the run_shell tool (M6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lingcore.errors import ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.shell import ShellArgs, run_shell


def _ctx(workspace: Path, *, confirm=None, **run_shell_opts) -> ToolContext:
    return ToolContext(
        workspace=workspace,
        confirm=confirm,
        options={"run_shell": run_shell_opts} if run_shell_opts else {},
    )


async def _yes(prompt: str) -> bool:
    return True


async def _no(prompt: str) -> bool:
    return False


async def test_runs_and_captures_output(tmp_path):
    ctx = _ctx(tmp_path, confirm=_yes, require_confirmation=True)
    out = await run_shell(ShellArgs(command="echo hello"), ctx)
    assert "hello" in out
    assert "exit code: 0" in out


async def test_runs_in_workspace_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    ctx = _ctx(tmp_path, confirm=_yes, require_confirmation=True)
    out = await run_shell(ShellArgs(command="ls"), ctx)
    assert "marker.txt" in out


async def test_nonzero_exit_code_reported(tmp_path):
    ctx = _ctx(tmp_path, confirm=_yes, require_confirmation=True)
    out = await run_shell(ShellArgs(command="exit 3"), ctx)
    assert "exit code: 3" in out


async def test_stderr_is_captured(tmp_path):
    ctx = _ctx(tmp_path, confirm=_yes, require_confirmation=True)
    out = await run_shell(ShellArgs(command="echo oops >&2"), ctx)
    assert "oops" in out


async def test_confirmation_denied_refuses(tmp_path):
    ctx = _ctx(tmp_path, confirm=_no, require_confirmation=True)
    with pytest.raises(ToolError, match="declined"):
        await run_shell(ShellArgs(command="echo nope"), ctx)


async def test_confirmation_required_but_no_handler(tmp_path):
    ctx = _ctx(tmp_path, confirm=None, require_confirmation=True)
    with pytest.raises(ToolError, match="no confirmation handler"):
        await run_shell(ShellArgs(command="echo nope"), ctx)


async def test_no_confirmation_when_disabled(tmp_path):
    # require_confirmation=False -> runs without a confirm handler.
    ctx = _ctx(tmp_path, require_confirmation=False)
    out = await run_shell(ShellArgs(command="echo free"), ctx)
    assert "free" in out


async def test_allowlist_skips_confirmation(tmp_path):
    # An allowlisted prefix runs even with require_confirmation and no handler.
    ctx = _ctx(
        tmp_path,
        confirm=None,
        require_confirmation=True,
        allow_patterns=["echo "],
    )
    out = await run_shell(ShellArgs(command="echo allowed"), ctx)
    assert "allowed" in out


async def test_allowlist_miss_still_confirms(tmp_path):
    # A command not matching any pattern still hits the (here: denying) gate.
    ctx = _ctx(
        tmp_path,
        confirm=_no,
        require_confirmation=True,
        allow_patterns=["pytest"],
    )
    with pytest.raises(ToolError, match="declined"):
        await run_shell(ShellArgs(command="rm -rf /"), ctx)


async def test_allowlist_matches_on_prefix_only(tmp_path):
    # Patterns match the start of the (stripped) command, not a substring.
    ctx = _ctx(
        tmp_path,
        confirm=_no,
        require_confirmation=True,
        allow_patterns=["ls"],
    )
    # "echo ls" does not start with "ls" -> still gated -> denied.
    with pytest.raises(ToolError, match="declined"):
        await run_shell(ShellArgs(command="echo ls"), ctx)


@pytest.mark.parametrize(
    ("command", "pattern"),
    [
        ("ls; echo unsafe", "ls"),
        ("git status && echo unsafe", "git status"),
        ("pytestx", "pytest"),
    ],
)
async def test_allowlist_does_not_skip_unsafe_prefixes(tmp_path, command, pattern):
    ctx = _ctx(
        tmp_path,
        confirm=_no,
        require_confirmation=True,
        allow_patterns=[pattern],
    )
    with pytest.raises(ToolError, match="declined"):
        await run_shell(ShellArgs(command=command), ctx)


async def test_timeout_kills_command(tmp_path):
    ctx = _ctx(tmp_path, require_confirmation=False, timeout=0.5)
    with pytest.raises(ToolError, match="timed out"):
        await run_shell(ShellArgs(command="sleep 5"), ctx)


async def test_output_truncation(tmp_path):
    ctx = _ctx(tmp_path, require_confirmation=False)
    # Emit well over the 16k char cap.
    out = await run_shell(
        ShellArgs(command="for i in $(seq 1 5000); do echo 0123456789; done"),
        ctx,
    )
    assert "truncated" in out
