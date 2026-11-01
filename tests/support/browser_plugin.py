"""Fixtures and builders for browser plugin tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import lingcore.tools.builtin  # noqa: F401
import lingcore.tools.builtin.web as web
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.events import Final, ToolResultEvent
from lingcore.message import ToolCall
from lingcore.tools import ToolContext
from tests.fakes import FakeLLMClient, ScriptedTurn

TOOLS = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_select",
    "browser_press",
    "browser_scroll",
    "browser_back",
    "browser_screenshot",
    "browser_resize",
    "browser_close",
]


def _load_browser():
    """Load the module under its canonical synthetic name (see test_canvas_skill)."""
    from lingcore.plugins.discovery import discover_plugins
    from lingcore.skills import _load_tool_module

    plugin = discover_plugins(None)["browser"]
    assert plugin.manifest.module is not None
    _load_tool_module(
        "browser",
        plugin.root / plugin.manifest.module,
        plugin.manifest.provides,
        prefix="browser",
    )
    name = next(n for n in sys.modules if n.startswith("lingcore_skill_tools.browser."))
    return sys.modules[name]


browser = _load_browser()


def profile_for(tmp_path, *, plugins=("browser",), tools=TOOLS, fetch=None, **options):
    tool_options = {"browser": options} if options else {}
    if fetch is not None:
        tool_options["fetch_url"] = fetch
    profile = AgentProfile(
        llm={"model": "fake"},
        workspace=str(tmp_path / "workspace"),
        plugins=list(plugins),
        tools=list(tools),
        tool_options=tool_options,
    )
    profile._source_dir = tmp_path
    return profile


def call(name, **arguments):
    return ScriptedTurn(tool_calls=[ToolCall(id=name, name=name, arguments=arguments)])


def results(events):
    return [e.result for e in events if isinstance(e, ToolResultEvent)]


async def run_tools(profile, turns, *, confirm=None):
    llm = FakeLLMClient([*turns, ScriptedTurn(text="done")])
    async with Agent.from_profile(profile, llm=llm, confirm=confirm) as agent:
        events = [event async for event in agent.run("browse")]
    assert isinstance(events[-1], Final)
    return results(events)


def detached_session(tmp_path, options):
    """Configure a session the way a tool call does, without an Agent."""
    hooks = browser.BrowserHooks.__new__(browser.BrowserHooks)
    hooks._session = None
    return hooks.session(ToolContext(workspace=tmp_path, options=options))


@pytest.fixture
def resolve(monkeypatch):
    answers = {
        "public.test": "93.184.216.34",
        "intranet.test": "10.0.0.5",
        "localhost": "127.0.0.1",
    }

    async def fake(host, port):
        if host not in answers:
            raise ToolError(f"could not resolve host {host!r}")
        return [answers[host]]

    monkeypatch.setattr(web, "_resolve_ips", fake)
    return answers


async def _socks_connect(port, address: bytes):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"\x05\x01\x00")
    assert await reader.readexactly(2) == b"\x05\x00"
    writer.write(b"\x05\x01\x00" + address)
    reply = await reader.readexactly(10)
    return reader, writer, reply[1]


class _FakeError(Exception):
    pass


class _FakeTimeout(_FakeError):
    pass


class _FakeHandle:
    def __init__(self, name, log, gate=None):
        self.name, self.log, self.gate = name, log, gate
        self.closed = False

    async def close(self):
        self.log.append(self.name)
        if self.gate is not None:
            gate, self.gate = self.gate, None
            await gate.wait()
        self.closed = True

    stop = close


class _FakePage:
    url = "https://public.test/"

    def __init__(self, delay=0.0):
        self.delay = delay

    def on(self, event, handler):
        pass

    async def goto(self, url, **kwargs):
        await asyncio.sleep(self.delay)

    async def wait_for_load_state(self, *args, **kwargs):
        pass

    async def evaluate(self, expression, args):
        return True

    async def title(self):
        return "Fake"

    async def aria_snapshot(self, mode, **kwargs):
        return "- heading"


class _FakeContext(_FakeHandle):
    def __init__(self, log, page):
        super().__init__("context", log)
        self.page = page

    async def new_page(self):
        return self.page


@pytest.fixture
def fake_playwright(monkeypatch):
    """Patch launches to install fake handles; returns the launched sessions."""
    from lingcore.bundled_plugins.browser import session

    launched = []
    monkeypatch.setattr(
        browser, "_playwright_errors", lambda: (_FakeError, _FakeTimeout)
    )
    monkeypatch.setattr(
        session, "_playwright_errors", lambda: (_FakeError, _FakeTimeout)
    )

    async def launch(self):
        log = []
        self.log = log
        self._playwright = _FakeHandle("playwright", log)
        self._browser = _FakeHandle("browser", log)
        self._context = _FakeContext(log, _FakePage(delay=0.2 if not launched else 0))
        launched.append(self)

    monkeypatch.setattr(browser.BrowserSession, "_launch", launch)
    return launched


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


_LIVE: bool | None = None


@pytest.fixture
async def live():
    global _LIVE
    if _LIVE is None:
        _LIVE = await _chromium_available()
    if not _LIVE:
        pytest.skip("Playwright Chromium is not installed")


REDIRECTS = {
    "/hop": "http://localhost:{port}/next",
    "/script-hop": "http://localhost:{port}/app.js",
}


PAGES = {
    # A parser-blocking script, so its redirect lands before the snapshot.
    "/redirected-script": """<!doctype html><title>Hop</title><p>Script below</p>
<script src="/script-hop"></script>""",
    "/": """<!doctype html><title>Home</title><h1>Welcome</h1>
<form action="/result"><input aria-label="Query" name="q"></form>
<a href="/next">Next page</a>
<select aria-label="Colour"><option value="r">red</option><option value="g">green</option></select>
<ul><li>Nested item</li></ul>
<img src="http://localhost:{port}/pixel.png" alt="pixel">""",
    "/next": "<!doctype html><title>Next</title><p>Second page</p>",
    "/responsive": """<!doctype html><title>Responsive</title><style>
.narrow {{ display: none }} @media (max-width: 600px) {{
.wide {{ display: none }} .narrow {{ display: block }} }}</style>
<p class="wide">Desktop layout</p><p class="narrow">Mobile layout</p>""",
    "/broken": """<!doctype html><title>Broken</title><div id="root"></div>
<script src="/missing.js"></script>
<script>console.error("render failed: boom"); console.log("plain-log-line");</script>
<script>undefinedFunction();</script>""",
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition("?")
        self.server.hits.append(f"{self.headers['Host'].split(':')[0]}{path}")
        if path in REDIRECTS:
            self.send_response(302)
            self.send_header(
                "Location", REDIRECTS[path].format(port=self.server.server_port)
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/result":
            body = f"<!doctype html><title>Result</title><p>You searched {query}</p>"
        elif path in PAGES:
            body = PAGES[path].format(port=self.server.server_port)
        else:
            self.send_error(404)
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.hits = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _ref(snapshot: str, label: str) -> str:
    line = next(
        line for line in snapshot.splitlines() if label in line and "[ref=" in line
    )
    return line.split("[ref=", 1)[1].split("]", 1)[0]


@pytest.fixture
async def owner_browser(live):
    """A Chromium someone else launched with a DevTools port, like a harness."""
    from playwright.async_api import async_playwright

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    async with async_playwright() as playwright:
        with tempfile.TemporaryDirectory() as profile_dir:
            context = await playwright.chromium.launch_persistent_context(
                profile_dir, args=[f"--remote-debugging-port={port}"]
            )
            await context.add_init_script(
                "addEventListener('DOMContentLoaded', () => document.body"
                ".insertAdjacentHTML('beforeend', '<p>Injected by owner</p>'))"
            )
            try:
                yield context, f"http://127.0.0.1:{port}"
            finally:
                await context.close()


@pytest.fixture
async def websocket_site():
    """A local echo server supporting the small text/binary frames used below."""
    hits, writers, tasks = [], set(), set()

    async def echo(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        writers.add(writer)
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            hits.append(request.split(b"\r\n", 1)[0])
            headers = dict(
                line.split(b": ", 1)
                for line in request.split(b"\r\n")[1:]
                if b": " in line
            )
            accept = base64.b64encode(
                hashlib.sha1(
                    headers[b"Sec-WebSocket-Key"]
                    + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                ).digest()
            )
            writer.write(
                b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                b"Connection: Upgrade\r\nSec-WebSocket-Accept: "
                + accept
                + b"\r\nSec-WebSocket-Protocol: lingcore-test\r\n\r\n"
            )
            await writer.drain()
            while True:
                opcode, length = await reader.readexactly(2)
                length &= 127
                assert length < 126
                mask = await reader.readexactly(4)
                payload = await reader.readexactly(length)
                decoded = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
                writer.write(bytes([opcode, length]) + decoded)
                await writer.drain()
                if opcode & 15 == 8:
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            writers.discard(writer)
            tasks.discard(task)

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"ws://127.0.0.1:{port}/socket", hits
    finally:
        server.close()
        await server.wait_closed()
        for writer in list(writers):
            writer.close()
        await asyncio.gather(*tasks)


async def _exchange_web_socket(frame, url):
    result = await frame.evaluate(
        """url => new Promise((resolve, reject) => {
          const socket = new WebSocket(url, ['lingcore-test']);
          socket.binaryType = 'arraybuffer';
          const result = { messages: [] };
          const timer = setTimeout(() => reject(Error('WebSocket timed out')), 3000);
          socket.onerror = () => { clearTimeout(timer); reject(Error('WebSocket failed')); };
          socket.onopen = () => {
            result.instance = socket instanceof WebSocket;
            result.tag = Object.prototype.toString.call(socket);
            result.protocol = socket.protocol;
            result.open = socket.readyState === WebSocket.OPEN;
            socket.send('hello');
            socket.send(new Uint8Array([1, 2, 3]));
          };
          socket.addEventListener('message', event => {
            result.messages.push(typeof event.data === 'string' ? event.data : [...new Uint8Array(event.data)]);
            if (result.messages.length === 2) socket.close(1000, 'done');
          });
          socket.onclose = event => {
            clearTimeout(timer);
            result.code = event.code;
            resolve(result);
          };
        })""",
        url,
    )
    assert result == {
        "messages": ["hello", [1, 2, 3]],
        "instance": True,
        "tag": "[object WebSocket]",
        "protocol": "lingcore-test",
        "open": True,
        "code": 1000,
    }


class _EventPage:
    """A page that records its handlers so a test can emit events."""

    def __init__(self):
        self.handlers = {}

    def on(self, event, handler):
        self.handlers[event] = handler


def _diagnostics_session(tmp_path):
    session = detached_session(tmp_path, {})
    page = _EventPage()
    session._track(page)
    return session, page.handlers
