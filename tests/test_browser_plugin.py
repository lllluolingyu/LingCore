"""Tests for the bundled Playwright ``browser`` plugin.

Policy and wiring tests run without a browser. The live tests drive a real
headless Chromium against a local HTTP server and are skipped when Playwright
or its Chromium build is not installed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import lingcore.tools.builtin  # noqa: F401
import lingcore.tools.builtin.web as web
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.doctor import diagnose_profile
from lingcore.errors import ConfigError, ToolError
from lingcore.events import Final, ToolResultEvent
from lingcore.message import ToolCall
from lingcore.tool_options import NetworkPolicy, parse_network_policy
from lingcore.tools import ToolContext
from tests.fakes import FakeLLMClient, ScriptedTurn

TOOLS = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_select",
    "browser_press",
    "browser_back",
    "browser_screenshot",
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


# --------------------------------------------------------------------------- #
# Manifest, options and wiring                                                #
# --------------------------------------------------------------------------- #


def test_bundled_manifest_declares_tools_hooks_and_module_requirement():
    from lingcore.plugins.discovery import discover_plugins

    plugin = discover_plugins(None)["browser"]
    assert plugin.source == "bundled"
    assert plugin.manifest.provides == TOOLS
    assert plugin.manifest.hooks == "BrowserHooks"
    assert [m.name for m in plugin.manifest.requires.modules] == ["playwright"]


def test_options_are_strict():
    assert browser.parse_browser_options(None) == browser.BrowserOptions()
    parsed = browser.parse_browser_options({"timeout": 5, "headless": False})
    assert parsed.timeout_ms == 5000 and not parsed.headless
    with pytest.raises(ConfigError, match="unknown tool_options.browser option"):
        browser.parse_browser_options({"headles": True})
    with pytest.raises(ConfigError, match="must be a boolean"):
        browser.parse_browser_options({"headless": "false"})


@pytest.mark.parametrize(
    "key", ["allow_private_hosts", "confirm_private_hosts", "allowed_networks"]
)
def test_network_policy_keys_point_to_the_shared_fetch_url_setting(key):
    with pytest.raises(ConfigError, match=r"set it under tool_options\.fetch_url"):
        browser.parse_browser_options({key: True})


def detached_session(tmp_path, options):
    """Configure a session the way a tool call does, without an Agent."""
    hooks = browser.BrowserHooks.__new__(browser.BrowserHooks)
    hooks._session = None
    return hooks.session(ToolContext(workspace=tmp_path, options=options))


def test_browser_reads_the_fetch_url_network_policy(tmp_path):
    session = detached_session(
        tmp_path,
        {
            "browser": {"timeout": 3},
            "fetch_url": {
                "allowed_networks": ["198.18.0.0/15"],
                "confirm_private_hosts": False,
            },
        },
    )
    assert session.options.timeout_ms == 3000
    assert [str(n) for n in session.policy.allowed_networks] == ["198.18.0.0/15"]
    assert not session.policy.confirm_private_hosts
    assert detached_session(tmp_path, {}).policy == NetworkPolicy()
    # An env expansion like "${VAR:-false}" arrives as a string; never truthy.
    with pytest.raises(ToolError, match=r"fetch_url\.allow_private_hosts"):
        detached_session(tmp_path, {"fetch_url": {"allow_private_hosts": "false"}})


async def test_tools_without_plugin_consent_explain_the_plugins_entry(tmp_path):
    profile = profile_for(tmp_path, plugins=())
    (result,) = await run_tools(profile, [call("browser_snapshot")])
    assert not result.ok
    assert "plugins:" in result.content


def test_doctor_reports_missing_playwright_and_missing_consent(tmp_path, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "playwright" else real(name, *a),
    )
    report = diagnose_profile(profile_for(tmp_path, plugins=()))
    messages = [(f.level, f.message) for f in report.findings]
    assert any(
        level == "error" and "lingcore[browser]" in message
        for level, message in messages
    )
    assert any(
        level == "warning" and "without explicit consent" in message
        for level, message in messages
    )
    enabled = diagnose_profile(profile_for(tmp_path))
    assert not any("without explicit consent" in f.message for f in enabled.findings)


# --------------------------------------------------------------------------- #
# Network policy                                                              #
# --------------------------------------------------------------------------- #


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


async def test_requests_follow_the_fetch_url_public_web_policy(resolve):
    session = browser.BrowserSession(browser.BrowserOptions())
    assert await session.blocked_reason("https://public.test/a.js") is None
    assert await session.blocked_reason("wss://public.test/socket") is None
    local = await session.blocked_reason("http://localhost:8000/")
    assert isinstance(local, web._NonPublic) and local.addr is None
    private = await session.blocked_reason("http://intranet.test/")
    assert isinstance(private, web._NonPublic) and private.addr == "10.0.0.5"
    resolve["proxied.test"] = "198.18.0.35"
    proxied = await session.blocked_reason("http://proxied.test/")
    assert "tool_options.fetch_url.allowed_networks" in proxied.error()
    assert "unsupported scheme" in await session.blocked_reason("file:///etc/passwd")
    assert "could not resolve" in await session.blocked_reason("http://nowhere.test/")
    session.approved_hosts.add("intranet.test")
    assert await session.blocked_reason("http://intranet.test/") is None
    exempt = browser.BrowserSession(
        browser.BrowserOptions(),
        parse_network_policy({"allowed_networks": ["10.0.0.0/8"]}),
    )
    assert await exempt.blocked_reason("http://intranet.test/") is None
    open_ = browser.BrowserSession(
        browser.BrowserOptions(), NetworkPolicy(allow_private_hosts=True)
    )
    assert await open_.blocked_reason("http://localhost/") is None
    assert await open_.blocked_reason("chrome://settings") is not None


@pytest.mark.parametrize("answer", [True, False, None])
async def test_private_navigation_asks_and_records_the_approval(
    tmp_path, resolve, answer
):
    prompts = []

    async def confirm(prompt):
        prompts.append(prompt)
        return answer

    ctx = ToolContext(
        workspace=tmp_path, confirm=confirm if answer is not None else None
    )
    session = browser.BrowserSession(browser.BrowserOptions())
    if answer:
        await session.authorize_navigation(ctx, "http://intranet.test/")
        assert session.approved_hosts == {"intranet.test"}
        assert "browser_navigate" in prompts[0] and "10.0.0.5" in prompts[0]
    else:
        with pytest.raises(ToolError, match="private/local address"):
            await session.authorize_navigation(ctx, "http://intranet.test/")
        assert session.approved_hosts == set()
        assert len(prompts) == (0 if answer is None else 1)
    quiet = browser.BrowserSession(
        browser.BrowserOptions(), NetworkPolicy(confirm_private_hosts=False)
    )
    with pytest.raises(ToolError):
        await quiet.authorize_navigation(ctx, "http://localhost/")


async def test_whole_url_is_validated_before_any_approval(resolve):
    session = browser.BrowserSession(browser.BrowserOptions())
    session.approved_hosts.add("intranet.test")
    # A clean request first: no verdict is cached for later URLs on the host.
    assert await session.blocked_reason("http://intranet.test/") is None
    assert await session.blocked_reason("https://public.test/") is None
    for url in (
        "http://user:pw@intranet.test/",
        "https://user:pw@public.test/",
        "wss://user@public.test/socket",
        "http://intranet.test:99999/",
    ):
        reason = await session.blocked_reason(url)
        assert isinstance(reason, str), url
    open_ = browser.BrowserSession(
        browser.BrowserOptions(), NetworkPolicy(allow_private_hosts=True)
    )
    assert "credentials" in await open_.blocked_reason("http://u:p@localhost/")

    class Request:
        url = "http://user:pw@intranet.test/"

    class Route:
        verdict = None

        async def continue_(self):
            self.verdict = "continue"

        async def abort(self, code):
            self.verdict = code

    route = Route()
    await session._route(route, Request())
    assert route.verdict == "blockedbyclient"
    assert session.drain_notes() == [
        "Blocked requests to non-public or unsupported hosts: intranet.test."
    ]


async def test_proxy_vets_and_pins_every_connection(resolve):
    resolve.update({"10.0.0.5": "10.0.0.5", "::1": "::1"})  # literals parse
    session = browser.BrowserSession(browser.BrowserOptions())
    assert await session.connect_address("public.test", 443) == "93.184.216.34"
    assert await session.connect_address("localhost", 80) is None
    assert await session.connect_address("10.0.0.5", 80) is None
    assert await session.connect_address("::1", 80) is None
    assert await session.connect_address("intranet.test", 80) is None
    session.approved_hosts.add("intranet.test")
    assert await session.connect_address("intranet.test", 80) == "10.0.0.5"
    with pytest.raises(ToolError, match="could not resolve"):
        await session.connect_address("nowhere.test", 80)
    assert session.drain_notes() == [
        "Blocked requests to non-public or unsupported hosts: "
        "localhost, 10.0.0.5, ::1, intranet.test."
    ]
    open_ = browser.BrowserSession(
        browser.BrowserOptions(), NetworkPolicy(allow_private_hosts=True)
    )
    assert await open_.connect_address("localhost", 80) == "localhost"


async def _socks_connect(port, address: bytes):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"\x05\x01\x00")
    assert await reader.readexactly(2) == b"\x05\x00"
    writer.write(b"\x05\x01\x00" + address)
    reply = await reader.readexactly(10)
    return reader, writer, reply[1]


async def test_socks_proxy_tunnels_only_vetted_hosts():
    async def echo(reader, writer):
        writer.write(await reader.read(100))
        await writer.drain()
        writer.close()

    upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
    upstream_port = upstream.sockets[0].getsockname()[1]
    port_bytes = upstream_port.to_bytes(2, "big")
    vetted = []

    async def vet(host, port):
        vetted.append((host, port))
        if host == "nowhere.test":
            raise ToolError("could not resolve host")
        return "127.0.0.1" if host == "allowed.test" else None

    proxy = browser._VettingProxy(vet, connect_timeout=5)
    url = await proxy.start()
    port = int(url.rsplit(":", 1)[1])
    try:
        name = b"allowed.test"
        reader, writer, code = await _socks_connect(
            port, b"\x03" + bytes([len(name)]) + name + port_bytes
        )
        assert code == 0
        writer.write(b"ping")
        assert await reader.read(100) == b"ping"
        writer.close()
        for address, expected in (
            (b"\x03\x0cblocked.test" + port_bytes, 2),
            (b"\x03\x0cnowhere.test" + port_bytes, 4),
            (b"\x01\x7f\x00\x00\x01" + port_bytes, 2),
            (b"\x04" + bytes(15) + b"\x01" + port_bytes, 2),
        ):
            _, writer, code = await _socks_connect(port, address)
            assert code == expected, address
            writer.close()
        assert vetted == [
            ("allowed.test", upstream_port),
            ("blocked.test", upstream_port),
            ("nowhere.test", upstream_port),
            ("127.0.0.1", upstream_port),
            ("::1", upstream_port),
        ]
        # An open tunnel is dropped when the proxy closes.
        reader, writer, _ = await _socks_connect(
            port, b"\x03" + bytes([len(name)]) + name + port_bytes
        )
        await proxy.close()
        assert await asyncio.wait_for(reader.read(), 5) == b""
        writer.close()
    finally:
        await proxy.close()
        upstream.close()
        await upstream.wait_closed()


# --------------------------------------------------------------------------- #
# Session lifecycle (fake Playwright handles)                                 #
# --------------------------------------------------------------------------- #


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

    async def title(self):
        return "Fake"

    async def aria_snapshot(self, mode):
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
    launched = []
    monkeypatch.setattr(
        browser, "_playwright_errors", lambda: (_FakeError, _FakeTimeout)
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


async def test_calls_queued_behind_close_get_a_fresh_owned_session(
    tmp_path, fake_playwright
):
    profile = profile_for(tmp_path, fetch={"allow_private_hosts": True})
    assert profile.loop.parallel_tools
    batch = ScriptedTurn(
        tool_calls=[
            ToolCall(
                id="go",
                name="browser_navigate",
                arguments={"url": "https://public.test/"},
            ),
            ToolCall(id="close", name="browser_close", arguments={}),
            ToolCall(id="look", name="browser_snapshot", arguments={}),
        ]
    )
    llm = FakeLLMClient([batch, ScriptedTurn(text="done")])
    agent = Agent.from_profile(profile, llm=llm)
    hooks = agent.tool_ctx.plugins["browser"]
    events = [event async for event in agent.run("browse")]
    assert [r.ok for r in results(events)] == [True, True, True]
    first, second = fake_playwright
    assert first.log == ["context", "browser", "playwright"]
    assert hooks._session is second  # the snapshot relaunched, still owned
    await agent.aclose()
    assert second.log == ["context", "browser", "playwright"]
    assert hooks._session is None
    # Calls arriving after aclose refuse instead of launching an orphan.
    ctx = ToolContext(workspace=tmp_path, plugins={"browser": hooks})
    with pytest.raises(ToolError, match="closed"):
        await browser.browser_snapshot.run(browser.NoArgs(), ctx)
    assert len(fake_playwright) == 2


async def test_cancelled_close_keeps_its_handles_and_resumes(tmp_path, fake_playwright):
    agent = Agent.from_profile(profile_for(tmp_path), llm=FakeLLMClient([]))
    hooks = agent.tool_ctx.plugins["browser"]
    ctx = agent.tool_ctx
    async with hooks.use(ctx) as session:
        await session.page()
    gate = asyncio.Event()
    session._context.gate = gate
    closing = asyncio.create_task(agent.aclose())
    while session.log != ["context"]:
        await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    # The owned close task survives its caller; nothing was dropped early.
    assert hooks._session is session and session._browser is not None
    gate.set()
    await agent.aclose()
    assert session.log == ["context", "browser", "playwright"]
    assert hooks._session is None and session._browser is None

    # A close cancelled inside the session (e.g. loop teardown) also resumes.
    direct = browser.BrowserSession(browser.BrowserOptions())
    log = []
    direct._context = _FakeHandle("context", log, gate=asyncio.Event())
    direct._browser = _FakeHandle("browser", log)
    direct._playwright = _FakeHandle("playwright", log)
    task = asyncio.create_task(direct.close())
    while log != ["context"]:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await direct.close()
    assert log == ["context", "context", "browser", "playwright"]
    assert direct._context is direct._browser is direct._playwright is None


def test_element_refs_are_validated():
    class Page:
        def locator(self, selector):
            return selector

    assert browser._locator(Page(), "e12") == "aria-ref=e12"
    assert browser._locator(Page(), "ref=f1e3") == "aria-ref=f1e3"
    for bad in ["", "12", "e1 >> text=x", "css=button"]:
        with pytest.raises(ToolError, match="invalid element ref"):
            browser._locator(Page(), bad)


# --------------------------------------------------------------------------- #
# Live Chromium                                                               #
# --------------------------------------------------------------------------- #


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
<img src="http://localhost:{port}/pixel.png" alt="pixel">""",
    "/next": "<!doctype html><title>Next</title><p>Second page</p>",
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
    line = next(line for line in snapshot.splitlines() if label in line)
    return line.split("[ref=", 1)[1].split("]", 1)[0]


async def test_live_navigate_type_click_select_screenshot(tmp_path, live, site):
    profile = profile_for(tmp_path, fetch={"allow_private_hosts": True})
    base = f"http://127.0.0.1:{site.server_port}"
    llm = FakeLLMClient([])
    async with Agent.from_profile(profile, llm=llm) as agent:

        async def step(name, **arguments):
            llm._turns = [call(name, **arguments), ScriptedTurn(text="ok")]
            events = [e async for e in agent.run("go")]
            (result,) = results(events)
            assert result.ok, result.content
            return result

        home = (await step("browser_navigate", url=base + "/")).content
        assert home.startswith(f"[Home] {base}/")
        assert 'heading "Welcome"' in home
        searched = await step(
            "browser_type", ref=_ref(home, "Query"), text="cats", submit=True
        )
        assert "[Result]" in searched.content and "q=cats" in searched.content
        home = (await step("browser_back")).content
        selected = await step("browser_select", ref=_ref(home, "Colour"), values=["g"])
        assert 'option "green" [selected]' in selected.content
        clicked = await step("browser_click", ref=_ref(selected.content, "Next page"))
        assert "Second page" in clicked.content
        shot = await step("browser_screenshot")
        assert shot.attachments[0].media_type == "image/png"
        stale = await step("browser_snapshot")
        assert "[Next]" in stale.content
        hooks = agent.tool_ctx.plugins["browser"]
        assert hooks._session is not None and hooks._session._browser is not None
    assert hooks._session is None


async def test_live_private_hosts_need_approval_and_subresources_are_dropped(
    tmp_path, live, site
):
    prompts = []

    async def confirm(prompt):
        prompts.append(prompt)
        return True

    base = f"http://127.0.0.1:{site.server_port}"
    (result,) = await run_tools(
        profile_for(tmp_path),
        [call("browser_navigate", url=base + "/")],
        confirm=confirm,
    )
    assert result.ok, result.content
    assert len(prompts) == 1 and "127.0.0.1" in prompts[0]
    # The page's image points at localhost, which was not approved.
    assert "Blocked requests to non-public or unsupported hosts: localhost" in (
        result.content
    )
    assert "127.0.0.1/" in site.hits
    assert not any(hit.startswith("localhost") for hit in site.hits)

    (denied,) = await run_tools(
        profile_for(tmp_path), [call("browser_navigate", url=base + "/next")]
    )
    assert not denied.ok and "private/local" in denied.content
    assert "127.0.0.1/next" not in site.hits


async def test_live_redirects_to_unapproved_hosts_never_connect(tmp_path, live, site):
    async def confirm(prompt):
        return True

    base = f"http://127.0.0.1:{site.server_port}"
    # Separate browsers: after a failed navigation Chromium's error page can
    # interrupt the next goto.
    (navigation,) = await run_tools(
        profile_for(tmp_path),
        [call("browser_navigate", url=base + "/hop")],
        confirm=confirm,
    )
    (subresource,) = await run_tools(
        profile_for(tmp_path),
        [call("browser_navigate", url=base + "/redirected-script")],
        confirm=confirm,
    )
    assert "127.0.0.1/hop" in site.hits and "127.0.0.1/script-hop" in site.hits
    assert not any(hit.startswith("localhost") for hit in site.hits)
    assert not navigation.ok
    assert "Blocked requests to non-public or unsupported hosts: localhost" in (
        navigation.content
    )
    assert subresource.ok, subresource.content
    assert "Script below" in subresource.content
    assert "Blocked requests to non-public or unsupported hosts: localhost" in (
        subresource.content
    )


async def test_live_agents_get_separate_browsers(tmp_path, live, site):
    profile = profile_for(tmp_path, fetch={"allow_private_hosts": True})
    base = f"http://127.0.0.1:{site.server_port}"
    agents = [
        Agent.from_profile(
            profile,
            llm=FakeLLMClient(
                [call("browser_navigate", url=f"{base}{path}"), ScriptedTurn(text="ok")]
            ),
        )
        for path in ("/", "/next")
    ]

    async def drain(agent):
        return [e async for e in agent.run("go")]

    outcomes = await asyncio.gather(*(drain(agent) for agent in agents))
    titles = [results(events)[0].content.split("]", 1)[0] for events in outcomes]
    assert titles == ["[Home", "[Next"]
    sessions = [agent.tool_ctx.plugins["browser"]._session for agent in agents]
    assert sessions[0]._browser is not sessions[1]._browser
    await asyncio.gather(*(agent.aclose() for agent in agents))
    assert all(s._browser is None and s._proxy is None for s in sessions)
