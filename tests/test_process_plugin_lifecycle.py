"""Process plugin tests: lifecycle."""

from __future__ import annotations

import asyncio

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.errors import ToolError
from lingcore.sandbox import ShellOptions, launch_shell
from tests.support.process_plugin import (
    BWRAP,
    TOOLS,
    _getenv,
    lines,
    make_ctx,
    pid_of,
    process,
    py,
    start,
)
from tests.support.process_plugin import host_launcher as host_launcher


def test_manifest_declares_tools_hooks_and_a_high_risk_start():
    from lingcore.plugins.discovery import discover_plugins

    plugin = discover_plugins(None)["process"]
    assert plugin.source == "bundled"
    assert plugin.manifest.provides == TOOLS
    assert plugin.manifest.hooks == "ProcessHooks"
    assert process.process_start.high_risk
    assert not process.process_logs.high_risk


def test_options_require_a_sandbox_and_prefer_their_own():
    with pytest.raises(ToolError, match="require a sandbox"):
        process.parse_process_options({})
    options, sandbox = process.parse_process_options({"run_shell": {"sandbox": BWRAP}})
    assert sandbox.backend == "bubblewrap" and options.max_processes == 3
    own = {"backend": "bubblewrap", "network": True}
    _, sandbox = process.parse_process_options(
        {"run_shell": {}, "process": {"sandbox": own}}
    )
    assert sandbox.network is True
    with pytest.raises(ToolError, match="max_ready_timeout"):
        process.parse_process_options(
            {"run_shell": {"sandbox": BWRAP}, "process": {"ready_timeout": 500}}
        )
    with pytest.raises(ToolError, match="tool_options.process"):
        process.parse_process_options(
            {"run_shell": {"sandbox": BWRAP}, "process": {"max_procs": 2}}
        )


def test_log_buffer_normalizes_terminal_output():
    log = process.LogBuffer(64 * 1024, 200)
    log.feed(b"\x1b[32m  VITE\x1b[0m ready\x1b]0;title\x07\r\n")
    log.feed(b"\x1b[2Kbar 10%\rbar 50%\rbar 100%\r")
    log.feed(b"\ndone \xe2\x9c")  # a CRLF and a UTF-8 sequence split across chunks
    log.feed(b"\x93\n")
    log.feed(b"no newline at exit")
    log.close()
    log.close()
    assert lines(log) == [
        "  VITE ready",
        "bar 10%",
        "bar 50%",
        "bar 100%",
        "done ✓",
        "no newline at exit",
    ]


def test_log_buffer_cuts_long_lines_and_drops_old_ones_by_bytes():
    log = process.LogBuffer(64 * 1024, 200)
    log.feed(b"x" * 5000 + b"\nnext\n")  # minified output on one line
    first, second = lines(log)
    assert first == "x" * 200 + " …[line truncated]" and second == "next"

    small = process.LogBuffer(64 * 1024, 200)
    for number in range(2000):
        small.feed(f"line {number:04d} {'.' * 60}\n".encode())
    assert small.total == 2000
    assert small.first > 1  # older lines were dropped to stay within the cap
    retained = small.lines_from(1)
    assert retained[0][0] == small.first and retained[-1] == (
        2000,
        f"line 1999 {'.' * 60}",
    )
    assert small.lines_from(1999) == retained[-2:]


async def test_every_launcher_gives_commands_an_empty_stdin(tmp_path):
    # An inherited stdin is the frontend's terminal: `cat` would block on it.
    execution = await launch_shell(
        "cat; echo $NO_COLOR",
        workspace=tmp_path,
        options=ShellOptions(),
        getenv=_getenv,
        extra_env={"NO_COLOR": "1"},
    )
    assert execution.process.stdout is not None
    async with asyncio.timeout(5):
        output = await execution.process.stdout.read()
        assert await execution.finish() == 0
    assert output == b"1\n"


async def test_only_bubblewrap_can_pass_descriptors(tmp_path):
    with pytest.raises(ToolError, match="only the bubblewrap runner"):
        await launch_shell(
            "true",
            workspace=tmp_path,
            options=ShellOptions(),
            getenv=_getenv,
            pass_fds=(0,),
        )


async def test_start_waits_for_ready_then_logs_list_and_stop(tmp_path, host_launcher):
    ctx, hooks, asked = make_ctx(tmp_path)
    code = "import time\nprint('booting')\nprint('Local: http://localhost:5173/')\ntime.sleep(60)"
    result = await start(ctx, py(code), name="dev", ready_pattern=r"Local:\s+http")
    pid = pid_of(result)
    assert asked == [py(code)]
    assert result.startswith(f"Started {pid} (dev): ready")
    assert "Ready: log line 2 matched ready_pattern." in result
    assert "booting" in result
    assert host_launcher[0]["env"]["NO_COLOR"] == "1"

    logs = await process.process_logs(process.LogsArgs(id=pid, since=2), ctx)
    assert "next cursor: since=3" in logs and "booting" not in logs
    grep = await process.process_logs(process.LogsArgs(id=pid, grep="boot"), ctx)
    assert "1| booting" in grep

    listing = await process.process_list(process.NoArgs(), ctx)
    assert f"{pid} (dev) | running" in listing

    stopped = await process.process_stop(process.IdArgs(id=pid), ctx)
    assert stopped.startswith(f"Stopped {pid} (dev)")
    with pytest.raises(ToolError, match="no background process"):
        await process.process_logs(process.LogsArgs(id=pid), ctx)
    assert "No background processes" in await process.process_list(
        process.NoArgs(), ctx
    )
    await hooks.aclose()


async def test_an_early_exit_is_a_result_with_its_output(tmp_path, host_launcher):
    ctx, hooks, _ = make_ctx(tmp_path)
    result = await start(
        ctx,
        py("import sys\nprint('Error: config missing')\nsys.exit(3)"),
        ready_pattern="ready",
        ready_timeout=10,
    )
    assert "exited (code 3)" in result and "before it was ready" in result
    assert "Error: config missing" in result
    listing = await process.process_list(process.NoArgs(), ctx)
    assert "exited (code 3)" in listing
    removed = await process.process_stop(process.IdArgs(id=pid_of(result)), ctx)
    assert removed.startswith("Removed") and "already exited" in removed
    await hooks.aclose()


async def test_not_ready_in_time_keeps_it_running(tmp_path, host_launcher):
    ctx, hooks, _ = make_ctx(tmp_path)
    result = await start(
        ctx, py("import time\ntime.sleep(60)"), ready_pattern="never", ready_timeout=0.3
    )
    assert "Started" in result and ": running" in result
    assert "Not ready after 0.3s" in result
    assert hooks.get(pid_of(result)).state == "running"
    await hooks.aclose()


async def test_confirmation_is_required_and_uses_its_own_allowlist(
    tmp_path, host_launcher
):
    ctx, hooks, _ = make_ctx(tmp_path, confirm=False)
    with pytest.raises(ToolError, match="user declined to start process"):
        await start(ctx, "echo hi")
    ctx, hooks, _ = make_ctx(tmp_path, confirm=None)
    with pytest.raises(ToolError, match="no confirmation handler"):
        await start(ctx, "echo hi")
    # run_shell's allowlist never authorizes a background process.
    ctx.options["run_shell"]["allow_patterns"] = ["echo hi"]
    with pytest.raises(ToolError, match="no confirmation handler"):
        await start(ctx, "echo hi")
    ctx.options["process"] = {"allow_patterns": ["echo hi"]}
    assert "Started" in await start(ctx, "echo hi")
    assert host_launcher[-1]["command"] == "echo hi"
    await hooks.aclose()


async def test_limits_are_checked_before_asking(tmp_path, host_launcher):
    ctx, hooks, asked = make_ctx(tmp_path, max_processes=1)
    await start(ctx, py("import time\ntime.sleep(60)"))
    with pytest.raises(ToolError, match="max_processes"):
        await start(ctx, "echo second")
    assert len(asked) == 1  # the refused start never prompted
    await hooks.aclose()


async def test_ids_are_unique_and_never_reused(tmp_path, host_launcher):
    ctx, hooks, _ = make_ctx(tmp_path)
    ids = set()
    for _ in range(4):
        result = await start(ctx, "true")
        ids.add(pid_of(result))
        await process.process_stop(process.IdArgs(id=pid_of(result)), ctx)
    assert len(ids) == 4 and all(len(i) == 8 and i.startswith("p-") for i in ids)
    await hooks.aclose()


async def test_lifetime_limit_stops_a_process(tmp_path, host_launcher):
    ctx, hooks, _ = make_ctx(tmp_path, max_lifetime=0.5)
    pid = pid_of(await start(ctx, py("import time\nprint('up')\ntime.sleep(60)")))
    proc = hooks.get(pid)
    async with asyncio.timeout(5):
        while proc.state != "exited":
            await asyncio.sleep(0.05)
    assert "max_lifetime" in (proc.note or "")
    listing = await process.process_list(process.NoArgs(), ctx)
    assert "max_lifetime" in listing
    await hooks.aclose()


async def test_exited_entries_are_capped(tmp_path, host_launcher):
    ctx, hooks, _ = make_ctx(tmp_path, max_exited=1)
    for _ in range(3):
        await start(ctx, "true")
    listing = await process.process_list(process.NoArgs(), ctx)
    assert listing.count("exited (code 0)") == 1
    assert "2 older exited process(es) were removed" in listing
    await hooks.aclose()
