"""Process plugin tests: integration."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.events import Final, ToolResultEvent, TurnCancelled
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.process_plugin import (
    BWRAP,
    LIVE_SERVER,
    OCI,
    OCI_E2E,
    SERVER,
    TOOLS,
    _chromium_available,
    _drain,
    _pgrep,
    approve,
    call,
    e2e,
    free_port,
    http_get,
    make_ctx,
    process,
    profile_for,
    py,
    start,
)
from tests.support.process_plugin import host_launcher as host_launcher


async def test_tools_need_the_plugin_and_a_sandbox(tmp_path, host_launcher):
    llm = FakeLLMClient([call("process_list"), ScriptedTurn(text="done")])
    async with Agent.from_profile(
        profile_for(tmp_path, plugins=()), llm=llm, confirm=approve
    ) as agent:
        events = [e async for e in agent.run("go")]
    (result,) = [e.result for e in events if isinstance(e, ToolResultEvent)]
    assert not result.ok and "plugins:" in result.content

    llm = FakeLLMClient([call("process_start", command="true"), ScriptedTurn("done")])
    async with Agent.from_profile(
        profile_for(tmp_path, sandbox=None), llm=llm, confirm=approve
    ) as agent:
        assert isinstance(agent.tool_ctx.plugins["process"], process.ProcessHooks)
        events = [e async for e in agent.run("go")]
    (result,) = [e.result for e in events if isinstance(e, ToolResultEvent)]
    assert not result.ok and "require a sandbox" in result.content


async def test_stop_of_the_starting_turn_kills_its_process_but_not_earlier_ones(
    tmp_path, host_launcher
):
    sleeper = py("import time\ntime.sleep(60)")
    llm = FakeLLMClient(
        [
            call("process_start", command=sleeper, name="kept"),
            ScriptedTurn(text="started"),
            # Waits for a line that never comes, so Stop lands mid-tool.
            call("process_start", command=sleeper, ready_pattern="x", ready_timeout=60),
        ]
    )
    async with Agent.from_profile(
        profile_for(tmp_path), llm=llm, confirm=approve
    ) as agent:
        hooks = agent.tool_ctx.plugins["process"]
        assert isinstance([e async for e in agent.run("one")][-1], Final)
        (kept,) = hooks.processes()

        task = asyncio.create_task(_drain(agent, "two"))
        async with asyncio.timeout(10):
            while len(hooks.processes()) < 2:
                await asyncio.sleep(0.02)
        doomed = hooks.processes()[1]
        assert agent.cancel_turn()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert isinstance(agent.finalize_cancelled_turn(), TurnCancelled)
        assert hooks.processes() == [kept]
        async with hooks.use():
            pass  # waits for the reap
        assert doomed.execution.process.returncode is not None
        assert kept.state == "running"
    # Closing the Agent stopped the committed one too.
    assert kept.execution.process.returncode is not None


async def test_frontend_profile_engages_processes(tmp_path, monkeypatch):
    from lingcore.composer import ComposeContext

    monkeypatch.setenv("LINGCORE_WORKSPACE", str(tmp_path / "workspace"))
    root = Path(__file__).parents[1] / "profiles" / "frontend"
    profile = AgentProfile.load(root)
    assert "process" in profile.plugins and "process" in profile.skills
    assert set(TOOLS) <= profile.initial_tool_set()
    async with Agent.from_profile(profile, llm=FakeLLMClient([])) as agent:
        assert isinstance(agent.tool_ctx.plugins["process"], process.ProcessHooks)
        advertised = {s["function"]["name"] for s in agent._effective_tool_schemas()}
        assert set(TOOLS) <= advertised
        prompt = await agent.composer.compose(
            ComposeContext(user_message="hi", turn_index=0)
        )
    assert "--strictPort" in prompt  # the process skill's guidance
    options, sandbox = process.parse_process_options(profile.tool_options)
    assert sandbox.backend == "bubblewrap" and sandbox.network is False


@e2e
async def test_bubblewrap_server_is_bridged_and_offline(tmp_path):
    ctx, hooks, _ = make_ctx(tmp_path)
    port = 8123
    (tmp_path / "server.py").write_text(SERVER)
    result = await start(
        ctx,
        f"python3 server.py {port}",
        port=port,
        ready_pattern="Local:",
        ready_timeout=15,
    )
    assert "(runner: bubblewrap" in result and ": ready" in result
    body = await http_get(hooks.bridge_opener("localhost", port), f"localhost:{port}")
    assert body.startswith(b"HTTP/1.0 200")
    # The server's own network namespace has no route out.
    offline = await start(
        ctx,
        "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), 3)\"",
    )
    assert "Network is unreachable" in offline
    # The bridged command never inherits the relay's control socket: it sees
    # only stdio plus the descriptor listdir itself opens.
    fds = await start(
        ctx,
        "python3 -c \"import os; print('FDS', len(os.listdir('/proc/self/fd')))\"",
        port=8124,
        ready_pattern="FDS",
    )
    assert "FDS 4" in fds
    await hooks.aclose()
    assert hooks.processes() == []


@e2e
async def test_bubblewrap_processes_die_with_a_killed_parent(tmp_path):
    import secrets

    # A per-run duration doubles as a marker no unrelated command line contains.
    marker = f"sleep {700000 + secrets.randbelow(99999)}"
    script = tmp_path / "parent.py"
    script.write_text(
        f"""
import asyncio, sys
sys.path.insert(0, {str(Path(__file__).parents[1])!r})
from tests.test_process_plugin import make_ctx, start, py
async def main():
    ctx, hooks, _ = make_ctx(__import__("pathlib").Path({str(tmp_path)!r}))
    await start(ctx, "exec {marker}")
    print("started", flush=True)
    await asyncio.sleep(600)
asyncio.run(main())
"""
    )
    parent = await asyncio.create_subprocess_exec(
        sys.executable, str(script), stdout=asyncio.subprocess.PIPE
    )
    assert parent.stdout is not None
    async with asyncio.timeout(20):
        assert await parent.stdout.readline() == b"started\n"
    assert await _pgrep(marker)  # running before its parent dies
    parent.kill()
    await parent.wait()
    async with asyncio.timeout(10):
        while await _pgrep(marker):
            await asyncio.sleep(0.1)


@pytest.mark.skipif(
    OCI_E2E not in {"docker-linux", "podman-linux"},
    reason="set LINGCORE_SANDBOX_E2E=docker-linux or podman-linux",
)
async def test_oci_server_is_bridged_and_offline(tmp_path):
    import shutil

    runtime = OCI_E2E.removesuffix("-linux")
    executable = os.environ.get("LINGCORE_SANDBOX_EXECUTABLE") or shutil.which(runtime)
    assert executable, f"{runtime} is required by LINGCORE_SANDBOX_E2E={OCI_E2E}"
    ctx, hooks, _ = make_ctx(tmp_path)
    ctx.options["process"] = {
        "sandbox": {
            **OCI,
            "runtime": runtime,
            "executable": str(Path(executable).resolve()),
            "image": os.environ.get("LINGCORE_SANDBOX_IMAGE", "python:3.13-slim"),
            "user": os.environ.get(
                "LINGCORE_SANDBOX_USER", f"{os.getuid()}:{os.getgid()}"
            ),
        }
    }
    port = 8123
    (tmp_path / "server.py").write_text(SERVER)
    result = await start(
        ctx,
        f"python3 server.py {port}",
        port=port,
        ready_pattern="Local:",
        ready_timeout=60,
    )
    assert f"(runner: oci/{runtime}:linux" in result and ": ready" in result, result
    body = await http_get(hooks.bridge_opener("localhost", port), f"localhost:{port}")
    assert body.startswith(b"HTTP/1.0 200")
    offline = await start(
        ctx,
        "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), 3)\"",
        ready_timeout=30,
    )
    assert "Network is unreachable" in offline, offline
    # The bridged command inherits neither the control connection nor the
    # listener, and it cannot reconnect through the mounted socket.
    # The pause lets LingCore accept the relay first; a connection queued
    # before that is still never accepted, but its connect() would succeed.
    probe = (
        "import os, socket, time\n"
        "print('FDS', len(os.listdir('/proc/self/fd')))\n"
        "time.sleep(1)\n"
        "s = socket.socket(socket.AF_UNIX)\n"
        "try:\n"
        f"    s.connect({process.CONTROL_TARGET!r}); print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('REFUSED', type(e).__name__)\n"
    )
    (tmp_path / "probe.py").write_text(probe)
    fds = await start(
        ctx,
        "python3 probe.py",
        port=8124,
        ready_pattern="REFUSED|CONNECTED",
        ready_timeout=60,
    )
    assert "FDS 4" in fds and "REFUSED" in fds, fds
    await hooks.aclose()
    assert hooks.processes() == []


async def test_live_browser_opens_a_bridged_dev_server_without_asking(
    tmp_path, host_launcher
):
    if not await _chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    port = free_port()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "server.py").write_text(LIVE_SERVER)
    asked: list[str] = []

    async def confirm(text: str) -> bool:
        asked.append(text)
        return text.startswith(sys.executable)  # only the process start

    command = f"{sys.executable} -u server.py {port}"
    profile = AgentProfile(
        llm={"model": "fake"},
        workspace=str(tmp_path / "workspace"),
        plugins=["browser", "process"],
        tools=[*TOOLS, "browser_navigate", "browser_snapshot"],
        tool_options={"run_shell": {"sandbox": BWRAP}},
    )
    profile._source_dir = tmp_path
    llm = FakeLLMClient(
        [
            call("process_start", command=command, port=port, ready_pattern="Local:"),
            call("browser_navigate", url=f"http://localhost:{port}/"),
            call("browser_snapshot"),
            ScriptedTurn(text="done"),
        ]
    )
    async with Agent.from_profile(profile, llm=llm, confirm=confirm) as agent:
        events = [e async for e in agent.run("check the app")]
        hooks = agent.tool_ctx.plugins["process"]
        (proc,) = hooks.processes()
    results = [e.result for e in events if isinstance(e, ToolResultEvent)]
    assert all(r.ok for r in results), [r.content for r in results]
    assert asked == [command]  # the bridged origin needed no approval
    snapshot = results[-1].content
    assert f"localhost:{port}" in snapshot and "hmr connected" in snapshot
    served = [line for _, line in proc.log.lines_from(1)]
    assert any("GET / HTTP/1.1" in line for line in served)
    assert proc.execution.process.returncode is not None  # closed with the Agent
