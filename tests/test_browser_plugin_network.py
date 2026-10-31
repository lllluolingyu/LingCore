"""Browser plugin tests: network."""

from __future__ import annotations

import asyncio

import pytest

import lingcore.tools.builtin  # noqa: F401
import lingcore.tools.builtin.web as web
from lingcore.agent import Agent
from lingcore.errors import ToolError
from lingcore.message import ToolCall
from lingcore.tool_options import NetworkPolicy, parse_network_policy
from lingcore.tools import ToolContext
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.browser_plugin import (
    _FakeHandle,
    _socks_connect,
    browser,
    call,
    profile_for,
    results,
    run_tools,
)
from tests.support.browser_plugin import fake_playwright as fake_playwright
from tests.support.browser_plugin import live as live
from tests.support.browser_plugin import owner_browser as owner_browser
from tests.support.browser_plugin import resolve as resolve
from tests.support.browser_plugin import site as site
from tests.support.browser_plugin import websocket_site as websocket_site


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


async def test_attached_routing_applies_the_host_policy_without_a_proxy(resolve):
    attached = browser.BrowserOptions(cdp_url="http://127.0.0.1:9222")
    session = browser.BrowserSession(attached)
    assert session.attached
    assert not await session.request_blocked("https://public.test/app.js")
    assert await session.request_blocked("http://localhost:8080/")
    assert await session.request_blocked("http://intranet.test/")
    assert await session.request_blocked("http://nowhere.test/")
    session.approved_hosts.add("intranet.test")
    assert not await session.request_blocked("http://intranet.test/")
    # A launched session leaves hosts to its proxy and checks only the URL.
    launched = browser.BrowserSession(browser.BrowserOptions())
    assert not await launched.request_blocked("http://localhost:8080/")
    assert await launched.request_blocked("http://u:p@public.test/")
    open_ = browser.BrowserSession(attached, NetworkPolicy(allow_private_hosts=True))
    assert not await open_.request_blocked("http://localhost:8080/")
    assert await open_.request_blocked("chrome://settings")


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
        await browser.browser_snapshot.run(browser.SnapshotArgs(), ctx)
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


async def test_attached_close_disconnects_and_closes_only_opened_tabs():
    log = []

    class Tab(_FakeHandle):
        def on(self, event, handler):
            if event == "close":
                self.on_close = handler

        async def close(self):
            await super().close()
            self.on_close(self)

    class Context(_FakeHandle):
        pages = []

        async def unroute_all(self, behavior):
            log.append(f"unroute:{behavior}")

    session = browser.BrowserSession(
        browser.BrowserOptions(cdp_url="http://127.0.0.1:9222")
    )
    owner_tab, opened = Tab("owner-tab", log), Tab("opened-tab", log)
    session._context = Context("context", log)
    session._browser = _FakeHandle("browser", log)
    session._playwright = _FakeHandle("playwright", log)
    session._track(owner_tab, announce=False)
    session._track(opened)
    session._opened.append(opened)
    assert session.drain_notes() == ["A new tab opened and is now active."]
    await session.close()
    # The owner's context and tab survive; disconnecting is browser.close().
    assert log == ["opened-tab", "unroute:ignoreErrors", "browser", "playwright"]
    assert not owner_tab.closed and not session._context


async def test_resize_is_bounded_and_refused_for_an_attached_browser(tmp_path):
    from pydantic import ValidationError

    browser.ResizeArgs(width=320, height=240)
    for width, height in [(319, 800), (1280, 239), (3841, 800), (1280, 2161)]:
        with pytest.raises(ValidationError):
            browser.ResizeArgs(width=width, height=height)

    profile = profile_for(tmp_path, cdp_url="http://127.0.0.1:9")
    (result,) = await run_tools(
        profile, [call("browser_resize", width=390, height=844)]
    )
    assert not result.ok
    # Refused before attaching: nothing listens on port 9.
    assert "not available for an attached browser" in result.content
    assert "cannot attach" not in result.content
