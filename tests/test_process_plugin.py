"""Process plugin tests: core behavior."""

from __future__ import annotations

import asyncio
import time

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.errors import ToolError
from lingcore.plugins import TurnRolledBackEvent
from tests.support.process_plugin import host_launcher as host_launcher
from tests.support.process_plugin import make_ctx as make_ctx
from tests.support.process_plugin import process
from tests.support.process_plugin import py as py
from tests.support.process_plugin import start as start


async def test_monitor_delivers_lines_before_exit_and_one_end_notice(
    tmp_path, host_launcher
):
    ctx, hooks, asked = make_ctx(tmp_path)
    ctx.background.enabled = True
    command = py(
        "import time; from pathlib import Path; print('first'); "
        "exec(\"while not Path('release').exists(): time.sleep(.01)\"); "
        "print('last without newline', end='')"
    )
    try:
        result = await process.process_monitor(
            process.MonitorArgs(command=command), ctx
        )
        assert result.startswith("Started monitor p-") and asked == [command]
        proc = hooks.processes()[0]
        await asyncio.wait_for(ctx.background.wait(), 3)
        incoming = ctx.background.take()
        assert "first" in incoming.text and proc.state == "running"
        (tmp_path / "release").write_text("go")
        await asyncio.wait_for(proc.supervisor, 3)
        incoming = ctx.background.take()
        assert "last without newline" in incoming.text
        assert incoming.text.count("Monitor ended:") == 1
        assert ctx.background.take() is None
    finally:
        await hooks.aclose()


@pytest.mark.parametrize("action", ["stop", "rollback", "close"])
async def test_monitor_cancellation_discards_queued_events(
    tmp_path, host_launcher, action
):
    ctx, hooks, _ = make_ctx(tmp_path)
    ctx.background.enabled = True
    before = time.monotonic()
    try:
        await process.process_monitor(
            process.MonitorArgs(
                command=py("import time; print('queued'); time.sleep(300)")
            ),
            ctx,
        )
        proc = hooks.processes()[0]
        await asyncio.wait_for(ctx.background.wait(), 3)
        if action == "stop":
            await process.process_stop(process.IdArgs(id=proc.id), ctx)
        elif action == "rollback":
            hooks.turn_rolled_back(TurnRolledBackEvent("cancelled", before))
            await asyncio.gather(*hooks._reaping)
        else:
            await hooks.aclose()
        assert proc.execution.process.returncode is not None
        assert ctx.background.take() is None
    finally:
        await hooks.aclose()


async def test_stopping_an_exited_monitor_keeps_its_last_events(
    tmp_path, host_launcher
):
    ctx, hooks, _ = make_ctx(tmp_path)
    ctx.background.enabled = True
    try:
        await process.process_monitor(
            process.MonitorArgs(command=py("print('final line')")), ctx
        )
        proc = hooks.processes()[0]
        await asyncio.wait_for(proc.supervisor, 3)
        assert ctx.background.busy  # ended, but its events are still queued
        await process.process_stop(process.IdArgs(id=proc.id), ctx)
        incoming = ctx.background.take()
        assert "final line" in incoming.text
        assert incoming.text.count("Monitor ended:") == 1
        assert not ctx.background.busy
    finally:
        await hooks.aclose()


@pytest.mark.parametrize("removal", ["prune", "process_stop"])
async def test_rollback_discards_removed_monitor_events_but_keeps_earlier_watches(
    tmp_path, host_launcher, removal
):
    ctx, hooks, _ = make_ctx(tmp_path, max_exited=0)
    ctx.background.enabled = True
    try:
        for text in ("earlier watch", "rolled back watch"):
            started = time.monotonic()
            await process.process_monitor(
                process.MonitorArgs(command=py(f"print({text!r})")), ctx
            )
            proc = hooks.processes()[0]
            await asyncio.wait_for(proc.supervisor, 3)
            if removal == "prune":
                await process.process_list(process.NoArgs(), ctx)
            else:
                await process.process_stop(process.IdArgs(id=proc.id), ctx)
            assert not hooks.processes()
        hooks.turn_rolled_back(TurnRolledBackEvent("cancelled", started))
        if hooks._reaping:
            await asyncio.gather(*hooks._reaping)
        incoming = ctx.background.take()
        assert "earlier watch" in incoming.text
        assert "rolled back watch" not in incoming.text
        assert incoming.text.count("Monitor ended:") == 1
        assert ctx.background.take() is None
    finally:
        await hooks.aclose()


async def test_removed_monitor_sources_are_released_after_event_delivery(
    tmp_path, host_launcher
):
    import weakref

    ctx, hooks, _ = make_ctx(tmp_path)
    ctx.background.enabled = True
    try:
        await process.process_monitor(
            process.MonitorArgs(command=py("print('finished')")), ctx
        )
        proc = hooks.processes()[0]
        await asyncio.wait_for(proc.supervisor, 3)
        source = weakref.ref(proc.source)
        await process.process_stop(process.IdArgs(id=proc.id), ctx)
        del proc
        assert source() is not None  # the inbox still owns its pending events
        assert ctx.background.take() is not None
        assert source() is None
    finally:
        await hooks.aclose()


async def test_monitor_deadline_is_capped_and_notifies_once(tmp_path, host_launcher):
    ctx, hooks, _ = make_ctx(tmp_path, max_lifetime=0.1)
    ctx.background.enabled = True
    try:
        await process.process_monitor(
            process.MonitorArgs(command=py("import time; time.sleep(300)"), timeout=10),
            ctx,
        )
        proc = hooks.processes()[0]
        assert proc.lifetime == 0.1
        await asyncio.wait_for(proc.supervisor, 3)
        incoming = ctx.background.take()
        assert "max_lifetime" in incoming.text
        assert incoming.text.count("Monitor ended:") == 1
        assert ctx.background.take() is None
    finally:
        await hooks.aclose()


async def test_monitor_refuses_unsupported_frontend_before_approval(
    tmp_path, host_launcher
):
    ctx, hooks, asked = make_ctx(tmp_path)
    try:
        with pytest.raises(ToolError, match="event delivery"):
            await process.process_monitor(
                process.MonitorArgs(command="echo hello"), ctx
            )
        assert not asked and not host_launcher
    finally:
        await hooks.aclose()
