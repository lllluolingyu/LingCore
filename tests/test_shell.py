"""Tests for the run_shell tool (M6)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from lingcore.errors import ToolError
from lingcore.sandbox import ShellExecution
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
    # A multi-token allowlist pattern matches its command plus trailing args and
    # runs without a confirmation handler.
    ctx = _ctx(
        tmp_path,
        confirm=None,
        require_confirmation=True,
        allow_patterns=["echo allowed"],
    )
    out = await run_shell(ShellArgs(command="echo allowed extra"), ctx)
    assert "allowed extra" in out


async def test_allowlist_single_token_pattern_is_exact(tmp_path):
    # A bare single-token pattern (e.g. "echo") matches only the exact command
    # with no arguments — it must never green-light arbitrary arguments.
    ctx = _ctx(
        tmp_path,
        confirm=None,
        require_confirmation=True,
        allow_patterns=["echo"],
    )
    # Exact bare command: allowed.
    out = await run_shell(ShellArgs(command="echo"), ctx)
    assert "exit code: 0" in out
    # Same program with an argument: NOT covered by the bare pattern -> refused
    # (no handler), so a generic reader like `cat` can't read files silently.
    with pytest.raises(ToolError, match="no confirmation handler"):
        await run_shell(ShellArgs(command="echo leak"), ctx)


async def test_allowlist_bare_reader_does_not_leak(tmp_path):
    # A bare `cat` allowlist entry must not auto-approve reading a file outside
    # the workspace.
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    ctx = _ctx(
        tmp_path,
        confirm=_no,  # would deny if prompted
        require_confirmation=True,
        allow_patterns=["cat"],
    )
    with pytest.raises(ToolError, match="declined"):
        await run_shell(ShellArgs(command=f"cat {secret}"), ctx)


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
        ("printf approved & printf chained", "printf approved"),
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
async def test_timeout_kills_children_after_leader_exits(tmp_path):
    import shlex

    marker = tmp_path / "orphan-marker"
    # The leader exits immediately; the background subshell inherits stdout,
    # so the reader waits on the pipe and times out while the leader is
    # already reaped. Cleanup must still signal the original process group.
    command = f"(sleep 0.5; : > {shlex.quote(str(marker))}) &"
    ctx = _ctx(tmp_path, require_confirmation=False, timeout=0.1)
    with pytest.raises(ToolError, match="timed out"):
        await run_shell(ShellArgs(command=command), ctx)
    await asyncio.sleep(0.7)
    assert not marker.exists()


def _execution_with_failing_abort() -> tuple[ShellExecution, list[bool]]:
    aborted: list[bool] = []

    async def exit_code() -> int:
        raise AssertionError("failed reads must not inspect an exit code")

    async def cleanup(abort: bool) -> None:
        aborted.append(abort)
        raise ToolError("sandbox cleanup failed")

    return ShellExecution(object(), "test", exit_code, cleanup), aborted  # type: ignore[arg-type]


async def test_cleanup_error_does_not_replace_cancellation(tmp_path, monkeypatch):
    import lingcore.tools.builtin.shell as shell_module

    execution, aborted = _execution_with_failing_abort()
    reading = asyncio.Event()

    async def fake_launch(*args, **kwargs) -> ShellExecution:
        return execution

    async def fake_read(*args, **kwargs) -> tuple[bytes, bool]:
        reading.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(shell_module, "launch_shell", fake_launch)
    monkeypatch.setattr(shell_module, "_read_capped", fake_read)
    task = asyncio.create_task(
        run_shell(
            ShellArgs(command="long-running"),
            _ctx(tmp_path, require_confirmation=False),
        )
    )
    await reading.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert aborted == [True]


async def test_cleanup_error_does_not_replace_timeout(tmp_path, monkeypatch):
    import lingcore.tools.builtin.shell as shell_module

    execution, aborted = _execution_with_failing_abort()

    async def fake_launch(*args, **kwargs) -> ShellExecution:
        return execution

    async def fake_read(*args, **kwargs) -> tuple[bytes, bool]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(shell_module, "launch_shell", fake_launch)
    monkeypatch.setattr(shell_module, "_read_capped", fake_read)

    with pytest.raises(ToolError, match="command timed out"):
        await run_shell(
            ShellArgs(command="long-running"),
            _ctx(tmp_path, require_confirmation=False, timeout=0.01),
        )
    assert aborted == [True]


async def test_cleanup_error_does_not_replace_read_error(tmp_path, monkeypatch):
    import lingcore.tools.builtin.shell as shell_module

    execution, aborted = _execution_with_failing_abort()

    async def fake_launch(*args, **kwargs) -> ShellExecution:
        return execution

    async def fake_read(*args, **kwargs) -> tuple[bytes, bool]:
        raise RuntimeError("read failed")

    monkeypatch.setattr(shell_module, "launch_shell", fake_launch)
    monkeypatch.setattr(shell_module, "_read_capped", fake_read)

    with pytest.raises(RuntimeError, match="read failed"):
        await run_shell(
            ShellArgs(command="broken-read"),
            _ctx(tmp_path, require_confirmation=False),
        )
    assert aborted == [True]


async def test_output_offloaded_when_large(tmp_path):
    ctx = _ctx(tmp_path, require_confirmation=False)
    # Emit well over the 8k offload threshold.
    out = await run_shell(
        ShellArgs(command="for i in $(seq 1 5000); do echo 0123456789; done"),
        ctx,
    )
    assert "full output" in out and ".lingcore/tool-output/shell-" in out
    assert "(exit code: 0)" in out  # command/exit header stays inline


async def test_output_truncation_when_offload_disabled(tmp_path):
    ctx = _ctx(
        tmp_path,
        require_confirmation=False,
        offload_over_chars=0,
        max_output_chars=2000,
    )
    out = await run_shell(
        ShellArgs(command="for i in $(seq 1 5000); do echo 0123456789; done"),
        ctx,
    )
    assert "truncated" in out


@pytest.mark.parametrize(
    ("output", "expected_truncated"),
    [("12345", False), ("123456", True)],
)
async def test_capture_reports_only_discarded_bytes_as_truncated(
    tmp_path, output, expected_truncated
):
    ctx = _ctx(tmp_path, require_confirmation=False, max_capture_bytes=5)
    out = await run_shell(ShellArgs(command=f"printf {output}"), ctx)
    assert ("output exceeded 5 bytes" in out) is expected_truncated


async def test_per_call_timeout_is_clamped_to_max_timeout(tmp_path):
    from lingcore.sandbox import parse_shell_options

    # max_timeout below timeout is a configuration error.
    with pytest.raises(ValueError):
        parse_shell_options({"timeout": 30, "max_timeout": 0.5})
    ctx = _ctx(tmp_path, require_confirmation=False, timeout=0.2, max_timeout=0.4)
    with pytest.raises(ToolError, match=r"timed out after 0\.4s"):
        await run_shell(ShellArgs(command="sleep 5", timeout=600), ctx)


async def test_per_call_timeout_can_extend_up_to_ceiling(tmp_path):
    ctx = _ctx(tmp_path, require_confirmation=False, timeout=0.1, max_timeout=10)
    out = await run_shell(ShellArgs(command="sleep 0.3; echo done", timeout=5), ctx)
    assert "done" in out


async def test_without_max_timeout_a_call_can_only_shorten(tmp_path):
    ctx = _ctx(tmp_path, require_confirmation=False, timeout=0.3)
    with pytest.raises(ToolError, match=r"timed out after 0\.3s"):
        await run_shell(ShellArgs(command="sleep 5", timeout=60), ctx)
