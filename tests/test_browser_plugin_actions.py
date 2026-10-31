"""Browser plugin tests: actions."""

from __future__ import annotations

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.agent import Agent
from lingcore.errors import ToolError
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.browser_plugin import _ref, browser, call, profile_for, results
from tests.support.browser_plugin import fake_playwright as fake_playwright
from tests.support.browser_plugin import live as live
from tests.support.browser_plugin import owner_browser as owner_browser
from tests.support.browser_plugin import resolve as resolve
from tests.support.browser_plugin import site as site
from tests.support.browser_plugin import websocket_site as websocket_site


def test_element_refs_are_validated():
    class Page:
        def locator(self, selector):
            return selector

    assert browser._locator(Page(), "e12") == "aria-ref=e12"
    assert browser._locator(Page(), "ref=f1e3") == "aria-ref=f1e3"
    for bad in ["", "12", "e1 >> text=x", "css=button"]:
        with pytest.raises(ToolError, match="invalid element ref"):
            browser._locator(Page(), bad)


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
        wide = (await step("browser_navigate", url=base + "/responsive")).content
        assert "Desktop layout" in wide and "Mobile layout" not in wide
        narrow = (await step("browser_resize", width=390, height=844)).content
        assert narrow.startswith(f"Viewport is now 390x844.\n[Responsive] {base}/")
        assert "Mobile layout" in narrow and "Desktop layout" not in narrow
        session = agent.tool_ctx.plugins["browser"]._session
        assert session._pages[-1].viewport_size == {"width": 390, "height": 844}
        restored = (await step("browser_resize", width=1280, height=800)).content
        assert "Desktop layout" in restored
        hooks = agent.tool_ctx.plugins["browser"]
        assert hooks._session is not None and hooks._session._browser is not None
    assert hooks._session is None


async def test_live_attached_close_closes_a_session_created_tab(owner_browser):
    owner, cdp_url = owner_browser
    await owner.pages[0].close()
    session = browser.BrowserSession(browser.BrowserOptions(cdp_url=cdp_url))
    try:
        page = await session.page()
        assert page in session._opened
        await session.close()
        assert session._context is session._browser is session._playwright is None
        assert not owner.pages
    finally:
        await session.close()


async def test_live_attached_close_closes_only_tool_opened_popups(
    tmp_path, owner_browser, site
):
    owner, cdp_url = owner_browser
    owner_tab = owner.pages[0]
    await owner_tab.set_content(
        f'<a href="http://127.0.0.1:{site.server_port}/next" target="_blank">Open</a>'
    )
    profile = profile_for(
        tmp_path, cdp_url=cdp_url, fetch={"allow_private_hosts": True}
    )
    async with Agent.from_profile(profile, llm=FakeLLMClient([])) as agent:
        ctx = agent.tool_ctx
        snapshot = await browser.browser_snapshot.run(browser.SnapshotArgs(), ctx)
        await browser.browser_click.run(
            browser.RefArgs(ref=_ref(snapshot, "Open")), ctx
        )
        session = ctx.plugins["browser"]._session
        assert len(session._popup_targets) == 1 and not session._acting
        independent_tab = await owner.new_page()
        async with owner_tab.expect_popup() as pending:
            await owner_tab.evaluate("window.open('about:blank')")
        owner_popup = await pending.value
        await browser.browser_close.run(browser.NoArgs(), ctx)
        assert set(owner.pages) == {owner_tab, independent_tab, owner_popup}
