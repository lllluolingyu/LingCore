"""Process plugin tests: bridge."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.errors import ToolError
from lingcore.plugins import TurnRolledBackEvent
from tests.support.process_plugin import (
    OCI,
    RELAY,
    SERVER,
    WINDOWS_OCI,
    free_port,
    http_get,
    make_ctx,
    pid_of,
    process,
    py,
    start,
)
from tests.support.process_plugin import host_launcher as host_launcher


@pytest.mark.parametrize(
    ("sandbox", "message"),
    [
        ({"backend": "bubblewrap", "network": True}, "network off"),
        ({**OCI, "network": True}, "network off"),
        (WINDOWS_OCI, "Linux containers on a Linux host"),
    ],
)
async def test_port_is_refused_where_it_cannot_be_bridged(
    tmp_path, host_launcher, sandbox, message
):
    ctx, hooks, asked = make_ctx(tmp_path)
    ctx.options["process"] = {"sandbox": sandbox}
    with pytest.raises(ToolError, match=message):
        await start(ctx, "npm run dev", port=5173)
    assert asked == [] and host_launcher == []
    # Without a port the same sandbox still runs watchers.
    assert "Started" in await start(ctx, "true")
    await hooks.aclose()


async def test_the_bridge_carries_connections_to_the_declared_port(
    tmp_path, host_launcher
):
    ctx, hooks, _ = make_ctx(tmp_path)
    port = free_port()
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    result = await start(
        ctx,
        f"{sys.executable} -u {script} {port}",
        port=port,
        ready_pattern="Local:",
        ready_timeout=10,
    )
    assert f"Browser URL: http://localhost:{port}/" in result
    assert "Warning" not in result
    # The command runs under the relay, which received the control socket.
    assert host_launcher[0]["command"].startswith(f"exec {sys.executable} -I -S ")
    for host in ("localhost", "127.0.0.1", "::1", "[::1]", "LOCALHOST."):
        assert hooks.bridge_opener(host, port) is not None
    assert hooks.bridge_opener("example.com", port) is None
    assert hooks.bridge_opener("localhost", port + 1) is None
    body = await http_get(hooks.bridge_opener("localhost", port), f"localhost:{port}")
    assert body.startswith(b"HTTP/1.0 200") and body.endswith(
        f"localhost:{port}".encode()
    )

    with pytest.raises(ToolError, match="already bridged"):
        await start(ctx, "true", port=port)
    await process.process_stop(process.IdArgs(id=pid_of(result)), ctx)
    await hooks.aclose()


async def test_the_bridge_refuses_when_nothing_listens_or_after_stop(
    tmp_path, host_launcher
):
    ctx, hooks, _ = make_ctx(tmp_path)
    port = free_port()
    result = await start(ctx, py("import time\ntime.sleep(60)"), port=port)
    opener = hooks.bridge_opener("localhost", port)
    with pytest.raises(ConnectionRefusedError, match="nothing is listening"):
        await opener()
    await hooks.stop(hooks.get(pid_of(result)))
    # A stopped (still listed) process keeps shadowing its port, refusing.
    with pytest.raises(ConnectionRefusedError):
        await hooks.bridge_opener("localhost", port)()
    await hooks.aclose()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux OCI only")
async def test_an_oci_relay_connects_once_to_a_private_listener(
    tmp_path, host_launcher
):
    ctx, hooks, _ = make_ctx(tmp_path)
    ctx.options["process"] = {"sandbox": OCI}
    port = free_port()
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    result = await start(
        ctx,
        f"{sys.executable} -u {script} {port}",
        port=port,
        ready_pattern="Local:",
        ready_timeout=10,
    )
    assert f"Browser URL: http://localhost:{port}/" in result
    launched = host_launcher[0]
    # The relay is told the in-container path of the mounted listener.
    assert f" {process.CONTROL_TARGET} {port} " in launched["command"]
    mounts = {m.target: m.source for m in launched["sandbox"].read_only_mounts}
    listener = Path(mounts[process.CONTROL_TARGET])
    body = await http_get(hooks.bridge_opener("localhost", port), f"localhost:{port}")
    assert body.startswith(b"HTTP/1.0 200")
    # Once the relay is in, nothing else can connect: the listener is gone.
    assert not listener.parent.exists()
    body = await http_get(hooks.bridge_opener("localhost", port), f"localhost:{port}")
    assert body.startswith(b"HTTP/1.0 200")
    await hooks.aclose()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux OCI only")
async def test_an_oci_listener_is_removed_when_the_relay_never_connects(
    tmp_path, host_launcher, monkeypatch
):
    monkeypatch.setattr(process, "_BRIDGE_TIMEOUT", 0.2)
    ctx, hooks, _ = make_ctx(tmp_path)
    ctx.options["process"] = {"sandbox": OCI}
    port = free_port()
    # No relay: the "image" has no python3, so the wrapper exits at once.
    monkeypatch.setattr(process, "OCI_GUEST_PYTHON", "/nonexistent/python3")
    result = await start(ctx, "true", port=port)
    assert "exited (code 127)" in result
    assert "the port bridge runs" in result
    mounts = {m.target: m.source for m in host_launcher[0]["sandbox"].read_only_mounts}
    listener = Path(mounts[process.CONTROL_TARGET])
    with pytest.raises(ConnectionRefusedError):
        await hooks.bridge_opener("localhost", port)()
    assert not listener.parent.exists()
    await hooks.aclose()


async def test_the_relay_never_runs_the_command_without_its_bridge(tmp_path):
    marker = tmp_path / "ran"
    relay = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-S",
        str(RELAY),
        str(tmp_path / "missing.sock"),
        "1",
        f"touch {marker}",
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await relay.communicate()
    assert relay.returncode == 1
    assert b"the command was not started" in err
    assert not marker.exists()


async def test_a_moved_port_is_reported(tmp_path, host_launcher):
    ctx, hooks, _ = make_ctx(tmp_path)
    result = await start(
        ctx,
        py("import time\nprint('Local: http://localhost:5174/')\ntime.sleep(60)"),
        port=5173,
        ready_pattern="Local:",
    )
    assert "announces port 5174, not the declared 5173" in result
    await hooks.aclose()


async def test_rollback_kills_only_processes_of_the_voided_turn(
    tmp_path, host_launcher
):
    ctx, hooks, _ = make_ctx(tmp_path)
    sleeper = py("import time\ntime.sleep(60)")
    kept = hooks.get(pid_of(await start(ctx, sleeper)))
    turn_start = time.monotonic()
    doomed = hooks.get(pid_of(await start(ctx, sleeper)))
    hooks.turn_rolled_back(TurnRolledBackEvent("cancelled", turn_start))
    # Gone from the registry at once; the next tool call waits for the kill.
    assert hooks.processes() == [kept]
    listing = await process.process_list(process.NoArgs(), ctx)
    assert doomed.id not in listing and doomed.state == "exited"
    assert doomed.execution.process.returncode is not None
    assert kept.state == "running"
    await hooks.aclose()


async def test_aclose_stops_everything_in_reverse_order_and_refuses_later(
    tmp_path, host_launcher
):
    ctx, hooks, _ = make_ctx(tmp_path)
    sleeper = py("import time\ntime.sleep(60)")
    procs = [hooks.get(pid_of(await start(ctx, sleeper))) for _ in range(2)]
    stopped: list[str] = []
    original = hooks._stop

    async def record(proc):
        stopped.append(proc.id)
        await original(proc)

    hooks._stop = record
    await hooks.aclose()
    await hooks.aclose()
    assert stopped == [procs[1].id, procs[0].id]
    assert all(p.execution.process.returncode is not None for p in procs)
    with pytest.raises(ToolError, match="closed"):
        await process.process_list(process.NoArgs(), ctx)
