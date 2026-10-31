"""Browser plugin tests: attachment."""

from __future__ import annotations

import asyncio
import socket

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.agent import Agent
from lingcore.tool_options import NetworkPolicy
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.browser_plugin import (
    _exchange_web_socket,
    _ref,
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


async def test_live_attach_preserves_owners_media_settings(owner_browser):
    owner, cdp_url = owner_browser
    page = owner.pages[0]
    await page.emulate_media(color_scheme="dark", reduced_motion="reduce")
    expression = """() => ({
        dark: matchMedia('(prefers-color-scheme: dark)').matches,
        reduced: matchMedia('(prefers-reduced-motion: reduce)').matches,
    })"""
    before = await page.evaluate(expression)
    session = browser.BrowserSession(browser.BrowserOptions(cdp_url=cdp_url))
    try:
        await session.page()
        assert (
            await page.evaluate(expression) == before == {"dark": True, "reduced": True}
        )
    finally:
        await session.close()


@pytest.mark.parametrize("in_frame", [False, True])
async def test_live_attached_web_sockets_are_vetted_in_loaded_frames(
    owner_browser, site, websocket_site, in_frame
):
    owner, cdp_url = owner_browser
    page = owner.pages[0]
    base = f"http://127.0.0.1:{site.server_port}/next"
    await page.goto(base)
    if in_frame:
        await page.set_content(f'<iframe src="{base}"></iframe>')
        frame = page.frames[1]
        await frame.wait_for_load_state()
    else:
        frame = page.main_frame
    url, hits = websocket_site
    session = browser.BrowserSession(browser.BrowserOptions(cdp_url=cdp_url))
    try:
        await session.page()
        code = await frame.evaluate(
            """url => new Promise((resolve, reject) => {
              const socket = new WebSocket(url, ['lingcore-test']);
              const timer = setTimeout(() => reject(Error('WebSocket timed out')), 3000);
              socket.onopen = () => socket.close();
              socket.onclose = event => { clearTimeout(timer); resolve(event.code); };
            })""",
            url,
        )
        assert code == 1008 and hits == []
        assert session._blocked == ["127.0.0.1"]
        session.approved_hosts.add("127.0.0.1")
        await _exchange_web_socket(frame, url)
        assert len(hits) == 1
    finally:
        await session.close()


async def test_live_attached_web_sockets_survive_disconnect_and_reattach(
    owner_browser, site, websocket_site
):
    owner, cdp_url = owner_browser
    page = owner.pages[0]
    await page.goto(f"http://127.0.0.1:{site.server_port}/next")
    url, _ = websocket_site
    for _ in range(2):
        await page.evaluate("() => { window.originalSocket = WebSocket; }")
        session = browser.BrowserSession(
            browser.BrowserOptions(cdp_url=cdp_url),
            NetworkPolicy(allow_private_hosts=True),
        )
        try:
            controlled = await session.page()
            await _exchange_web_socket(page, url)
            await session.close()
            assert await page.evaluate("WebSocket === originalSocket")
            await _exchange_web_socket(page, url)
            # New documents are guarded too, and survive cleanup without reload.
            controlled = await session.page()
            await controlled.reload()
            await _exchange_web_socket(page, url)
        finally:
            await session.close()
        await _exchange_web_socket(page, url)


@pytest.mark.parametrize("pending", [False, True])
async def test_live_attached_open_and_pending_web_sockets_survive_disconnect(
    owner_browser, site, websocket_site, pending
):
    owner, cdp_url = owner_browser
    page = owner.pages[0]
    await page.goto(f"http://127.0.0.1:{site.server_port}/next")
    url, _ = websocket_site
    session = browser.BrowserSession(
        browser.BrowserOptions(cdp_url=cdp_url),
        NetworkPolicy(allow_private_hosts=True),
    )
    checking, release = asyncio.Event(), asyncio.Event()

    async def delayed_check(source, url):
        checking.set()
        await release.wait()
        return True

    if pending:
        session._allow_web_socket = delayed_check
    try:
        await session.page()
        await page.evaluate(
            """url => {
              window.savedSocket = new WebSocket(url, ['lingcore-test']);
              window.socketOpened = new Promise((resolve, reject) => {
                savedSocket.onopen = resolve;
                savedSocket.onerror = () => reject(Error('WebSocket failed'));
              });
            }""",
            url,
        )
        if pending:
            await asyncio.wait_for(checking.wait(), 3)
        else:
            await asyncio.wait_for(
                page.evaluate("() => socketOpened.then(() => true)"), 3
            )
        await asyncio.wait_for(session.close(), 3)
        release.set()
        await asyncio.wait_for(page.evaluate("() => socketOpened.then(() => true)"), 3)
        message = await asyncio.wait_for(
            page.evaluate(
                """() => new Promise(resolve => {
              savedSocket.onmessage = event => { savedSocket.close(); resolve(event.data); };
              savedSocket.send('after disconnect');
            })"""
            ),
            3,
        )
        assert message == "after disconnect"
    finally:
        release.set()
        await asyncio.wait_for(session.close(), 3)


async def test_live_attached_session_drives_the_owners_tab_and_leaves_it_open(
    tmp_path, owner_browser, site
):
    owner, cdp_url = owner_browser
    base = f"http://127.0.0.1:{site.server_port}"
    await owner.pages[0].goto(base + "/")
    profile = profile_for(
        tmp_path, fetch={"allow_private_hosts": True}, cdp_url=cdp_url, timeout=10
    )
    llm = FakeLLMClient([])
    async with Agent.from_profile(profile, llm=llm) as agent:

        async def step(name, **arguments):
            llm._turns = [call(name, **arguments), ScriptedTurn(text="ok")]
            (result,) = results([e async for e in agent.run("go")])
            assert result.ok, result.content
            return result.content

        # The owner's tab is adopted as is, with its page already loaded.
        home = await step("browser_snapshot")
        assert home.startswith(f"[Home] {base}/") and "new tab" not in home
        assert 'heading "Welcome"' in home
        assert "Nested item" in home
        shallow = await step("browser_snapshot", depth=1)
        assert "- list" in shallow and "Nested item" not in shallow
        clicked = await step("browser_click", ref=_ref(home, "Next page"))
        # The owner's init script still runs on the session's navigations.
        assert "Second page" in clicked and "Injected by owner" in clicked
        back = await step("browser_back")
        query = _ref(back, "Query")
        scoped = await step("browser_snapshot", ref=query)
        assert f"(Only the subtree of {query}.)" in scoped
        assert "Next page" not in scoped
        typed = await step(
            "browser_type", ref=_ref(scoped, "Query"), text="cats", submit=True
        )
        assert "q=cats" in typed
        assert await step("browser_close") == "browser closed"
        # A fresh session re-attaches to the same tab.
        assert "[Result]" in await step("browser_snapshot")
    assert len(owner.pages) == 1 and "q=cats" in owner.pages[0].url
    assert "Second page" in await owner.pages[0].evaluate(
        "fetch('/next').then(r => r.text())"
    )


async def test_live_attached_session_refuses_private_hosts_per_request(
    tmp_path, owner_browser, site
):
    owner, cdp_url = owner_browser
    prompts = []

    async def confirm(prompt):
        prompts.append(prompt)
        return True

    base = f"http://127.0.0.1:{site.server_port}"
    (result,) = await run_tools(
        profile_for(tmp_path, cdp_url=cdp_url),
        [call("browser_navigate", url=base + "/")],
        confirm=confirm,
    )
    assert result.ok, result.content
    assert len(prompts) == 1 and "127.0.0.1" in prompts[0]
    # Without the proxy, routing still drops the unapproved localhost image.
    assert "Blocked requests to non-public or unsupported hosts: localhost" in (
        result.content
    )
    assert not any(hit.startswith("localhost") for hit in site.hits)
    # Disconnecting removed the routes: the owner's own requests are unaffected.
    await owner.pages[0].goto(f"http://localhost:{site.server_port}/next")
    assert any(hit.startswith("localhost") for hit in site.hits)


async def test_live_attach_failure_does_not_reveal_the_endpoint(tmp_path, live):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    secret = f"ws://127.0.0.1:{port}/devtools/browser/secret-session-id"
    (result,) = await run_tools(
        profile_for(tmp_path, cdp_url=secret, timeout=2),
        [call("browser_snapshot")],
    )
    assert not result.ok and "cannot attach" in result.content
    assert "secret-session-id" not in result.content


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
