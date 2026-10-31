"""Fixtures and builders for process plugin tests."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.config import AgentProfile
from lingcore.message import ToolCall
from lingcore.plugins import PluginContext
from lingcore.sandbox import ShellExecution, _kill_and_reap
from lingcore.tools import ToolContext
from tests.fakes import ScriptedTurn

TOOLS = [
    "process_start",
    "process_logs",
    "process_stop",
    "process_list",
    "process_monitor",
]


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


def lines(log):
    return [line for _, line in log.lines_from(1)]


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


async def _drain(agent, text):
    return [event async for event in agent.run(text)]


e2e = pytest.mark.skipif(not E2E, reason="set LINGCORE_SANDBOX_E2E=bubblewrap")


OCI_E2E = os.environ.get("LINGCORE_SANDBOX_E2E", "")


async def _pgrep(pattern: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "pgrep", "-f", pattern, stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    return bool(out.strip())


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
