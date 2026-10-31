"""Browser plugin tests: core behavior."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

import lingcore.tools.builtin  # noqa: F401
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.doctor import diagnose_profile
from lingcore.errors import ConfigError, ToolError
from lingcore.tool_options import NetworkPolicy
from tests.fakes import FakeLLMClient
from tests.support.browser_plugin import (
    TOOLS,
    _FakeError,
    _FakePage,
    _FakeTimeout,
    browser,
    call,
    detached_session,
    profile_for,
    run_tools,
)
from tests.support.browser_plugin import fake_playwright as fake_playwright
from tests.support.browser_plugin import live as live
from tests.support.browser_plugin import owner_browser as owner_browser
from tests.support.browser_plugin import resolve as resolve
from tests.support.browser_plugin import site as site
from tests.support.browser_plugin import websocket_site as websocket_site


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
    "options",
    [
        {"settle_quiet_ms": -1},
        {"settle_timeout_ms": 0},
        {"settle_quiet_ms": True},
        {"settle_timeout_ms": "soon"},
        {"settle_timeout_ms": 10_001},
        {"settle_quiet_ms": 501, "settle_timeout_ms": 500},
    ],
)
def test_settling_options_reject_invalid_budgets(options):
    with pytest.raises(ConfigError):
        browser.parse_browser_options(options)


async def test_settling_follows_popup_and_renders_observed_page(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        browser, "_playwright_errors", lambda: (_FakeError, _FakeTimeout)
    )
    observed = []
    pages = []

    class Page(_FakePage):
        async def evaluate(self, expression, args):
            observed.append(self)
            pages.append(Page())  # each page opens another while settling
            return True

    async def active():
        return pages[-1]

    pages.append(Page())
    session = SimpleNamespace(options=browser.BrowserOptions(), page=active)
    shown, notes = await browser._settled_page(session, pages[0])
    assert shown is observed[-1]
    assert shown is not pages[-1]  # never claim the last, unobserved popup was settled
    assert "repeated tab changes" in notes[0] and "timed out" not in notes[0]


async def test_settling_timeout_and_cancellation_are_bounded(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        browser, "_playwright_errors", lambda: (_FakeError, _FakeTimeout)
    )
    entered = asyncio.Event()

    class Page(_FakePage):
        async def evaluate(self, expression, args):
            entered.set()
            await asyncio.Event().wait()

    page = Page()

    async def active():
        return page

    session = SimpleNamespace(
        options=browser.BrowserOptions(settle_quiet_ms=5, settle_timeout_ms=20),
        page=active,
    )
    shown, notes = await asyncio.wait_for(browser._settled_page(session, page), 1)
    assert shown is page and "timed out" in notes[0]
    entered.clear()
    session.options = browser.BrowserOptions()
    task = asyncio.create_task(browser._settled_page(session, page))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    session.options = browser.BrowserOptions(settle_quiet_ms=0)
    assert await browser._settled_page(session, page) == (page, [])


async def test_settling_recovers_when_navigation_destroys_context(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        browser, "_playwright_errors", lambda: (_FakeError, _FakeTimeout)
    )
    calls = []

    class Page(_FakePage):
        async def evaluate(self, expression, args):
            calls.append(args)
            if len(calls) == 1:
                raise _FakeError("Execution context was destroyed")
            return True

    page = Page()

    async def active():
        return page

    session = SimpleNamespace(options=browser.BrowserOptions(), page=active)
    assert await browser._settled_page(session, page) == (page, [])
    assert len(calls) == 2
    assert calls[1][1] <= calls[0][1]  # navigation cannot reset the deadline


async def test_settling_reports_repeated_page_errors(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        browser, "_playwright_errors", lambda: (_FakeError, _FakeTimeout)
    )

    class Page(_FakePage):
        async def evaluate(self, expression, args):
            raise _FakeError("Target page has been closed")

    page = Page()

    async def active():
        return page

    session = SimpleNamespace(options=browser.BrowserOptions(), page=active)
    _, notes = await browser._settled_page(session, page)
    assert "page/navigation errors" in notes[0] and "timed out" not in notes[0]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9222",
        "ws://localhost:9222/devtools/browser/abc",
        "http://[::1]:9222/",
        "https://127.0.0.2:9222",
    ],
)
def test_cdp_url_accepts_loopback_endpoints(url):
    assert browser.parse_browser_options({"cdp_url": url}).cdp_url == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        42,
        "file:///tmp/devtools",
        "http://10.0.0.5:9222",
        "http://example.com:9222",
        "http://user:secret@localhost:9222",
        "http://localhost:99999",
        "ws://169.254.169.254/devtools/browser/secret-id",
    ],
)
def test_cdp_url_must_be_a_credential_free_loopback_endpoint(url):
    with pytest.raises(ConfigError, match="cdp_url must be") as error:
        browser.parse_browser_options({"cdp_url": url})
    if isinstance(url, str) and url:
        assert url not in str(error.value)  # a ws path is the session id


@pytest.mark.parametrize(
    "key", ["headless", "executable_path", "viewport_width", "chromium_sandbox"]
)
def test_launch_options_cannot_be_combined_with_cdp_url(key):
    value = {"executable_path": "/bin/chromium", "viewport_width": 800}.get(key, True)
    with pytest.raises(ConfigError, match=f"{key} cannot be combined with"):
        browser.parse_browser_options({"cdp_url": "http://127.0.0.1:9", key: value})
    # Options that shape the session rather than the launch still apply.
    attached = browser.parse_browser_options(
        {"cdp_url": "http://127.0.0.1:9", "timeout": 4, "max_chars": 2_000}
    )
    assert attached.timeout_ms == 4000 and attached.max_chars == 2_000


@pytest.mark.parametrize(
    "key", ["allow_private_hosts", "confirm_private_hosts", "allowed_networks"]
)
def test_network_policy_keys_point_to_the_shared_fetch_url_setting(key):
    with pytest.raises(ConfigError, match=r"set it under tool_options\.fetch_url"):
        browser.parse_browser_options({key: True})


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


async def test_frontend_profile_ships_the_browser_engaged(tmp_path, monkeypatch):
    from pathlib import Path

    from lingcore.composer import ComposeContext

    monkeypatch.setenv("LINGCORE_WORKSPACE", str(tmp_path / "workspace"))
    root = Path(__file__).parents[1] / "profiles" / "frontend"
    profile = AgentProfile.load(root)
    assert profile.plugins == ["browser", "process"]
    assert profile.skills == ["browser", "process"]
    assert set(TOOLS) <= profile.initial_tool_set()
    # The sandbox stays offline unless the operator opts in.
    assert profile.tool_options["run_shell"]["sandbox"]["network"] is False
    async with Agent.from_profile(profile, llm=FakeLLMClient([])) as agent:
        assert isinstance(agent.tool_ctx.plugins["browser"], browser.BrowserHooks)
        advertised = {s["function"]["name"] for s in agent._effective_tool_schemas()}
        assert set(TOOLS) <= advertised
        prompt = await agent.composer.compose(
            ComposeContext(user_message="hi", turn_index=0)
        )
    assert "browser_resize" in prompt  # the browser skill's guidance
    assert "dev server" in prompt  # world.md's notes on reaching one


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
