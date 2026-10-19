"""Tests for the bundled ``process`` plugin (background processes).

Most tests swap the sandbox launcher for a host launcher, so they exercise the
registry, logs, limits, bridge and lifecycle with real processes but without
Bubblewrap. ``LINGCORE_SANDBOX_E2E=bubblewrap`` adds end-to-end runs inside a
real Bubblewrap sandbox.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.events import Final, ToolResultEvent, TurnCancelled
from lingcore.message import ToolCall
from lingcore.plugins import PluginContext, TurnRolledBackEvent
from lingcore.sandbox import ShellExecution, ShellOptions, _kill_and_reap, launch_shell
from lingcore.tools import ToolContext
from tests.fakes import FakeLLMClient, ScriptedTurn

TOOLS = ["process_start", "process_logs", "process_stop", "process_list"]
BWRAP = {"backend": "bubblewrap"}
E2E = os.environ.get("LINGCORE_SANDBOX_E2E") == "bubblewrap"


def _load_process():
    from lingcore.plugins.discovery import discover_plugins
    from lingcore.skills import _load_tool_module

    plugin = discover_plugins(None)["process"]
    assert plugin.manifest.module is not None
    _load_tool_module(
        "process",
        plugin.root / plugin.manifest.module,
        plugin.manifest.provides,
        prefix="process",
    )
    name = next(n for n in sys.modules if n.startswith("lingcore_skill_tools.process."))
    return sys.modules[name]


process = _load_process()
RELAY = Path(process.__file__).with_name("relay.py")


@pytest.fixture
def host_launcher(monkeypatch):
    """Run "sandboxed" commands on the host, recording what was launched."""
    launched: list[dict] = []

    async def fake_launch(
        command, *, workspace, options, getenv, pass_fds=(), extra_env=None
    ):
        launched.append(
            {
                "command": command,
                "sandbox": options.sandbox,
                "env": dict(extra_env or {}),
            }
        )
        host_command = command
        # The guest sees mounts at their targets; on the host use the sources.
        for mount in getattr(options.sandbox, "read_only_mounts", []):
            host_command = host_command.replace(mount.target, mount.source)
        proc = await asyncio.create_subprocess_shell(
            host_command,
            cwd=str(workspace),
            env={**os.environ, **(extra_env or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            pass_fds=pass_fds,
            start_new_session=True,
        )

        async def exit_code():
            return await proc.wait()

        async def cleanup(abort):
            if abort:
                await _kill_and_reap(proc)

        oci = getattr(options.sandbox, "backend", None) == "oci"
        return ShellExecution(proc, "oci/fake" if oci else "fake", exit_code, cleanup)

    monkeypatch.setattr(process, "launch_shell", fake_launch)
    monkeypatch.setattr(process, "GUEST_PYTHON", sys.executable)
    monkeypatch.setattr(process, "OCI_GUEST_PYTHON", sys.executable)
    return launched


def make_ctx(tmp_path, *, confirm=True, **options):
    """A tool context with the plugin enabled; ``confirm`` approves or denies."""
    asked: list[str] = []

    async def approve(text: str) -> bool:
        asked.append(text)
        return confirm

    tool_options = {"run_shell": {"sandbox": BWRAP}, "process": options}
    hooks = process.ProcessHooks(
        PluginContext("process", tmp_path, tmp_path, None, None, options, _getenv)
    )
    ctx = ToolContext(
        workspace=tmp_path, confirm=approve if confirm is not None else None
    )
    ctx.options = tool_options
    ctx.plugins = {"process": hooks}
    return ctx, hooks, asked


def _getenv(name, default=None):
    return default


def py(code: str) -> str:
    """A shell command running ``code`` with this interpreter."""
    import shlex

    return f"{shlex.quote(sys.executable)} -u -c {shlex.quote(code)}"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def start(ctx, command, **kwargs):
    return await process.process_start(
        process.StartArgs(command=command, **kwargs), ctx
    )


def pid_of(result: str) -> str:
    return result.split()[1].rstrip(":")


# --------------------------------------------------------------------------- #
# Manifest, options and the log buffer                                        #
# --------------------------------------------------------------------------- #


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


def lines(log):
    return [line for _, line in log.lines_from(1)]


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


# --------------------------------------------------------------------------- #
# Tools                                                                       #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Ports and the bridge                                                        #
# --------------------------------------------------------------------------- #


OCI = {
    "backend": "oci",
    "runtime": "podman",
    "executable": "/usr/bin/podman",
    "image": "node:22",
    "user": "1000:1000",
    "resources": {
        "cpus": 1,
        "memory_mb": 512,
        "pids": 128,
        "ephemeral_storage_mb": 256,
    },
}
WINDOWS_OCI = {
    "backend": "oci",
    "runtime": "docker",
    "executable": "/usr/bin/docker",
    "image": "mcr.microsoft.com/windows/nanoserver:ltsc2022",
    "container_os": "windows",
    "resources": {"cpus": 1, "memory_mb": 512, "ephemeral_storage_mb": 256},
}


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


SERVER = """
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"hello " + self.headers["Host"].encode())
    def log_message(self, *a):
        print("served", self.path, flush=True)
server = http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H)
print("Local: http://localhost:%s/" % sys.argv[1], flush=True)
server.serve_forever()
"""


async def http_get(opener, host="localhost:1") -> bytes:
    reader, writer = await opener()
    writer.write(f"GET / HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
    await writer.drain()
    try:
        return await reader.read()
    finally:
        writer.close()


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


# --------------------------------------------------------------------------- #
# Lifecycle                                                                   #
# --------------------------------------------------------------------------- #


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


def profile_for(tmp_path, *, plugins=("process",), sandbox=BWRAP):
    profile = AgentProfile(
        llm={"model": "fake"},
        workspace=str(tmp_path / "workspace"),
        plugins=list(plugins),
        tools=TOOLS,
        tool_options={"run_shell": {"sandbox": sandbox}} if sandbox else {},
    )
    profile._source_dir = tmp_path
    return profile


def call(tool_name, **arguments):
    return ScriptedTurn(
        tool_calls=[ToolCall(id=tool_name, name=tool_name, arguments=arguments)]
    )


async def approve(text):
    return True


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


async def _drain(agent, text):
    return [event async for event in agent.run(text)]


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


# --------------------------------------------------------------------------- #
# Bubblewrap end to end                                                       #
# --------------------------------------------------------------------------- #

e2e = pytest.mark.skipif(not E2E, reason="set LINGCORE_SANDBOX_E2E=bubblewrap")


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


OCI_E2E = os.environ.get("LINGCORE_SANDBOX_E2E", "")


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


async def _pgrep(pattern: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "pgrep", "-f", pattern, stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    return bool(out.strip())


# --------------------------------------------------------------------------- #
# Live Chromium through the bridge                                            #
# --------------------------------------------------------------------------- #

LIVE_SERVER = r"""
import base64, hashlib, http.server, socketserver, sys
PAGE = b'''<!doctype html><title>Dev</title><h1 id=h>waiting</h1>
<script>
const ws = new WebSocket("ws://" + location.host + "/hmr");
ws.onopen = () => { document.getElementById("h").textContent = "hmr connected"; };
ws.onerror = () => { document.getElementById("h").textContent = "hmr failed"; };
</script>'''
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        if self.path == "/hmr":
            key = self.headers["Sec-WebSocket-Key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            accept = base64.b64encode(hashlib.sha1(key.encode()).digest()).decode()
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.rfile.read(1)  # hold the socket open until the client goes
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)
    def log_message(self, *args):
        print("served", self.requestline, flush=True)
class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
server = S(("127.0.0.1", int(sys.argv[1])), H)
print("Local: http://localhost:%s/" % sys.argv[1], flush=True)
server.serve_forever()
"""


async def _chromium_available() -> bool:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return False
    try:
        async with async_playwright() as playwright:
            await (await playwright.chromium.launch()).close()
    except Exception:
        return False
    return True


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
