"""Headless Chromium tools for the bundled ``browser`` plugin (Playwright).

One browser belongs to one Agent: ``BrowserHooks`` is the per-Agent plugin
instance, launches Chromium lazily on the first tool call and closes it from
``Agent.aclose``. Tools reach it through ``ctx.plugins["browser"]``, so the
plugin must be enabled with ``plugins: [browser]``; listing the tools alone
loads this module but leaves them without a session.

Design notes:
- Playwright is the optional ``lingcore[browser]`` extra and is imported only
  when a session starts, so a profile without it still assembles and the tools
  report how to install it.
- Pages are read through Playwright's AI-mode aria snapshot. Its ``[ref=eN]``
  handles address elements for the action tools and are valid until the next
  snapshot; every action returns a fresh one.
- Network policy is ``fetch_url``'s own (invariant 9), read from
  ``tool_options.fetch_url`` so the two tools share one setting. It is enforced
  at two layers. Playwright routing validates every request URL (scheme,
  embedded credentials, port), but it never sees redirect hops, so the host
  check lives in ``_VettingProxy``: Chromium reaches the network only through
  a per-session loopback SOCKS5 proxy (loopback included, local DNS disabled)
  that vets every connection — page loads, subresources, redirect hops and
  WebSockets alike — refuses a local or non-public host and pins the rest to
  the vetted address, as ``fetch_url`` does. Only a model-requested navigation
  can put such a host to the user; an approval covers that hostname for the
  rest of the session.
- A target the ``process`` plugin bridges (a sandboxed dev server's declared
  port on ``localhost``/``127.0.0.1``/``::1``) is connected through that
  plugin's relay instead of TCP, and needs no private-host approval: the user
  confirmed the command serving it. Attached browsers never use a bridge.
- The context is ephemeral (no stored cookies or credentials), service
  workers are blocked so every request stays routable, downloads are refused,
  and JavaScript dialogs are dismissed and reported.
- Console errors/warnings, uncaught exceptions, failed requests and HTTP error
  responses are collected per result (bounded and deduplicated) so a blank or
  broken page explains itself. The text is page-authored, so it is quoted and
  labelled as data for the model, never as instructions.
- ``cdp_url`` attaches to an already running Chromium on loopback instead
  (e.g. one a test harness prepared and logged in). The session adopts that
  browser's default context and tabs, and closing only disconnects: it never
  closes the owner's browser or tabs, only the tabs the session opened itself.
  The proxy can be set only at launch, so an attached session enforces the
  host policy in HTTP routing and a reversible page WebSocket guard instead:
  it still refuses non-public hosts per request, but cannot pin the vetted
  address, see redirect hops, cover existing connections/cached constructors/
  worker sockets, or block service workers and downloads in an owner's context.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from pydantic import BaseModel, Field

from lingcore.bundled_plugins.browser.options import _MAX_HEIGHT as _MAX_HEIGHT
from lingcore.bundled_plugins.browser.options import _MAX_WIDTH as _MAX_WIDTH
from lingcore.bundled_plugins.browser.options import _MIN_HEIGHT as _MIN_HEIGHT
from lingcore.bundled_plugins.browser.options import _MIN_WIDTH as _MIN_WIDTH
from lingcore.bundled_plugins.browser.options import OPTION_PATH
from lingcore.bundled_plugins.browser.options import BrowserOptions as BrowserOptions
from lingcore.bundled_plugins.browser.options import (
    parse_browser_options as parse_browser_options,
)
from lingcore.bundled_plugins.browser.proxy import PortBridges
from lingcore.bundled_plugins.browser.proxy import _VettingProxy as _VettingProxy
from lingcore.bundled_plugins.browser.session import (
    _MAX_PAGE_PROBLEMS as _MAX_PAGE_PROBLEMS,
)
from lingcore.bundled_plugins.browser.session import (
    _MAX_PROBLEM_CHARS as _MAX_PROBLEM_CHARS,
)
from lingcore.bundled_plugins.browser.session import BrowserSession as BrowserSession
from lingcore.bundled_plugins.browser.session import _error_text, _locator
from lingcore.bundled_plugins.browser.session import (
    _playwright_errors as _playwright_errors,
)
from lingcore.errors import ConfigError, ToolError
from lingcore.media import attachment_from_bytes
from lingcore.media_types import IMAGE_MAX_BYTES
from lingcore.plugins import PluginContext, PluginHooks
from lingcore.tool_options import (
    parse_network_policy,
)
from lingcore.tools import ToolContext, ToolOutput, tool

if TYPE_CHECKING:
    from playwright.async_api import (
        Page,
    )


_SETTLE_JS = """([quietMs, capMs]) => new Promise(resolve => {
    let quiet;
    const done = stable => {
        observer.disconnect(); clearTimeout(quiet); clearTimeout(cap); resolve(stable);
    };
    const observer = new MutationObserver(() => {
        clearTimeout(quiet); quiet = setTimeout(() => done(true), quietMs);
    });
    observer.observe(document, {
        subtree: true, childList: true, attributes: true, characterData: true
    });
    quiet = setTimeout(() => done(true), quietMs);
    const cap = setTimeout(() => done(false), capMs);
})"""


class BrowserHooks(PluginHooks):
    """Owns this Agent's browser session; overrides only ``aclose``.

    One lock serializes every tool call with ``reset``/``aclose``, and the
    session is chosen under it, so a call queued behind ``browser_close``
    gets the fresh session instead of reviving the closed one. Closing runs
    as an owned task: when a caller is cancelled (or times out) mid-close the
    task keeps running, and the next ``use``/``reset`` awaits its completion.
    """

    def __init__(self, ctx: PluginContext) -> None:
        super().__init__(ctx)
        self._session: BrowserSession | None = None
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @asynccontextmanager
    async def use(self, ctx: ToolContext) -> AsyncIterator[BrowserSession]:
        """Hold the browser for one tool call."""
        async with self._lock:
            if self._close_task is not None:
                await self._close_session()
            if self._closed:
                raise ToolError("the browser plugin is closed")
            yield self.session(ctx)

    def session(self, ctx: ToolContext) -> BrowserSession:
        """This Agent's session, configured from the calling tool's options.

        Callers hold the lock (see ``use``).
        """
        if self._session is None:
            options = ctx.options or {}
            try:
                browser = parse_browser_options(options.get("browser"))
                policy = parse_network_policy(options.get("fetch_url", {}))
            except ConfigError as exc:
                raise ToolError(str(exc)) from None
            # The process plugin's dev-server ports, when it is enabled.
            bridges = ctx.plugins.get("process")
            if not callable(getattr(bridges, "bridge_opener", None)):
                bridges = None
            self._session = BrowserSession(
                browser, policy, cast("PortBridges | None", bridges)
            )
        return self._session

    async def _close_session(self) -> None:
        session = self._session
        if session is None:
            return
        if self._close_task is None or self._close_task.done():
            # A done task was interrupted or finished after its caller was
            # cancelled; BrowserSession.close resumes or no-ops.
            self._close_task = asyncio.create_task(session.close())
        await asyncio.shield(self._close_task)
        self._session = None
        self._close_task = None

    async def reset(self) -> None:
        """Close the session; the next tool call starts a fresh browser."""
        async with self._lock:
            await self._close_session()

    async def aclose(self) -> None:
        self._closed = True  # calls still queued for the lock now refuse
        await self.reset()


def _hooks(ctx: ToolContext) -> BrowserHooks:
    hooks = ctx.plugins.get("browser")
    if not isinstance(hooks, BrowserHooks):
        raise ToolError(
            "the browser plugin is not enabled: add `browser` to the profile's "
            "plugins: list so this agent gets its own browser session"
        )
    return hooks


async def _settled_page(session: BrowserSession, page: Page) -> tuple[Page, list[str]]:
    """Settle and select a page under one deadline, even across popup/navigation races."""
    options = session.options
    if options.settle_quiet_ms == 0:
        return await session.page(), []
    playwright_error, playwright_timeout = _playwright_errors()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + options.settle_timeout_ms / 1000
    current = page
    for attempt in range(3):
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            async with asyncio.timeout(remaining):
                await current.wait_for_load_state(
                    "domcontentloaded", timeout=max(1, int(remaining * 1000))
                )
                cap_ms = max(1, int((deadline - loop.time()) * 1000))
                stable = await current.evaluate(
                    _SETTLE_JS, [options.settle_quiet_ms, cap_ms]
                )
            newest = await session.page()
            if newest is current:
                return current, [] if stable else [
                    "DOM settling timed out; the page may still be changing."
                ]
            # If the budget is spent, render the page we actually observed,
            # rather than silently switching to an unobserved popup.
            if loop.time() >= deadline:
                break
            if attempt == 2:
                return current, [
                    "DOM settling stopped after repeated tab changes; "
                    "showing the last observed page."
                ]
            current = newest
        except (TimeoutError, playwright_timeout):
            break
        except playwright_error:
            # Navigation can destroy evaluate's execution context. Retry the
            # selected page under the original deadline; cancellation propagates.
            current = await session.page()
            if attempt == 2:
                return current, [
                    "DOM settling interrupted by repeated page/navigation errors; "
                    "the current page may still be changing."
                ]
    return current, ["DOM settling timed out; the page may still be changing."]


@overload
async def _act(
    ctx: ToolContext,
    action: Any,
    *,
    navigate_to: str | None = None,
    launched_only: str | None = None,
    screenshot: Literal[False] = False,
) -> str: ...


@overload
async def _act(
    ctx: ToolContext,
    action: Any,
    *,
    navigate_to: str | None = None,
    launched_only: str | None = None,
    screenshot: Literal[True],
) -> ToolOutput: ...


async def _act(
    ctx: ToolContext,
    action: Any,
    *,
    navigate_to: str | None = None,
    launched_only: str | None = None,
    screenshot: bool = False,
) -> str | ToolOutput:
    """Run ``action(page)`` holding the browser and return a fresh snapshot.

    ``navigate_to`` is authorized against the session the action will use,
    before the browser launches. ``launched_only`` names a tool an attached
    browser refuses, before anything attaches.
    """
    playwright_error, playwright_timeout = _playwright_errors()
    async with _hooks(ctx).use(ctx) as session:
        if launched_only is not None and session.attached:
            raise ToolError(
                f"{launched_only} is not available for an attached browser "
                f"({OPTION_PATH}.cdp_url): its tabs belong to the browser's owner"
            )
        if navigate_to is not None:
            await session.authorize_navigation(ctx, navigate_to)
        page = await session.page()
        notes: list[str] = []
        session._action_target = session._page_targets.get(page)
        session._acting = True
        try:
            try:
                note = await action(page)
                if note:
                    notes.append(note)
            except playwright_timeout as exc:
                notes.append(f"Timed out: {_error_text(exc)}")
            except playwright_error as exc:
                raise ToolError(
                    " ".join([_error_text(exc), *session.drain_notes()])
                ) from None
            shown, settling_notes = await _settled_page(session, page)
            text = await session.render(ctx, [*notes, *settling_notes], page=shown)
            if screenshot:
                view = await _screenshot(shown)
                return ToolOutput(
                    text=f"{text}\n\n{view.text}", attachments=view.attachments
                )
            return text
        finally:
            session._acting = False
            session._action_target = None


class NavigateArgs(BaseModel):
    url: str = Field(description="The http(s) URL to open in the active tab.")


class ScrollArgs(BaseModel):
    direction: Literal["up", "down", "top", "bottom"] = Field(
        default="down",
        description="Scroll the page up/down one viewport, or jump to its top/bottom.",
    )


class RefArgs(BaseModel):
    ref: str = Field(
        description="Element ref from the latest snapshot, e.g. 'e12' (from [ref=e12])."
    )


class TypeArgs(RefArgs):
    text: str = Field(description="Text that replaces the field's current value.")
    submit: bool = Field(default=False, description="Press Enter after typing.")


class SelectArgs(RefArgs):
    values: list[str] = Field(
        min_length=1, description="Option values or labels to select."
    )


class PressArgs(BaseModel):
    key: str = Field(
        description="Key or chord for the focused element, e.g. 'Enter', 'Escape', 'Control+A'."
    )


class ScreenshotArgs(BaseModel):
    full_page: bool = Field(
        default=False, description="Capture the whole page instead of the viewport."
    )


class ResizeArgs(BaseModel):
    width: int = Field(
        ge=_MIN_WIDTH,
        le=_MAX_WIDTH,
        description="Viewport width in CSS pixels, e.g. 390 (phone), 768 (tablet), 1280 (desktop).",
    )
    height: int = Field(
        ge=_MIN_HEIGHT,
        le=_MAX_HEIGHT,
        description="Viewport height in CSS pixels, e.g. 844 (phone), 1024 (tablet), 800 (desktop).",
    )


class SnapshotArgs(BaseModel):
    ref: str | None = Field(
        default=None,
        description=(
            "Only return this element's subtree (a ref from the latest "
            "snapshot), e.g. one list or form instead of the whole page."
        ),
    )
    depth: int | None = Field(
        default=None,
        ge=1,
        description="Only return this many levels of the tree below the root.",
    )


class NoArgs(BaseModel):
    pass


@tool(
    name="browser_navigate",
    description=(
        "Open a URL in the headless browser and return the page as an "
        "accessibility snapshot whose [ref=eN] handles the other browser tools "
        "accept. Use for pages that need JavaScript or interaction; prefer "
        "fetch_url for plain documents."
    ),
)
async def browser_navigate(args: NavigateArgs, ctx: ToolContext) -> str:
    async def go(page: Page) -> None:
        await page.goto(args.url, wait_until="domcontentloaded")

    return await _act(ctx, go, navigate_to=args.url)


@tool(
    name="browser_snapshot",
    description=(
        "Return the active page's settled accessibility snapshot, including "
        "offscreen content, optionally "
        "only one element's subtree (ref) or only the top levels (depth)."
    ),
)
async def browser_snapshot(args: SnapshotArgs, ctx: ToolContext) -> str:
    async with _hooks(ctx).use(ctx) as session:
        page, notes = await _settled_page(session, await session.page())
        return await session.render(
            ctx, notes, ref=args.ref, depth=args.depth, page=page
        )


@tool(
    name="browser_click",
    description="Click the element with the given ref and return the new snapshot.",
)
async def browser_click(args: RefArgs, ctx: ToolContext) -> str:
    async def click(page: Page) -> None:
        await _locator(page, args.ref).click()

    return await _act(ctx, click)


@tool(
    name="browser_type",
    description=(
        "Focus and fill the input or rich-text editor with the given ref, optionally pressing "
        "Enter, and return the new snapshot."
    ),
)
async def browser_type(args: TypeArgs, ctx: ToolContext) -> str:
    async def fill(page: Page) -> None:
        field = _locator(page, args.ref)
        if await field.evaluate("el => ['IFRAME', 'FRAME'].includes(el.tagName)"):
            field = field.content_frame.locator("body")
        await field.focus()
        await field.fill(args.text)
        if args.submit:
            await field.press("Enter")

    return await _act(ctx, fill)


@tool(
    name="browser_select",
    description="Choose options in the select element with the given ref.",
)
async def browser_select(args: SelectArgs, ctx: ToolContext) -> str:
    async def select(page: Page) -> None:
        await _locator(page, args.ref).select_option(args.values)

    return await _act(ctx, select)


@tool(
    name="browser_press",
    description="Press a key in the active page and return the new snapshot.",
)
async def browser_press(args: PressArgs, ctx: ToolContext) -> str:
    async def press(page: Page) -> None:
        await page.keyboard.press(args.key)

    return await _act(ctx, press)


@tool(
    name="browser_scroll",
    description=(
        "Scroll the page up/down one viewport or to top/bottom, independent of "
        "keyboard focus. Return the actual movement, a fresh snapshot and a "
        "viewport screenshot. This scrolls the page, not an inner panel."
    ),
)
async def browser_scroll(args: ScrollArgs, ctx: ToolContext) -> ToolOutput:
    async def scroll(page: Page) -> str:
        before, after, bottom = await page.evaluate(
            """direction => {
                const before = window.scrollY;
                const end = Math.max(0, document.scrollingElement.scrollHeight - innerHeight);
                const top = direction === 'top' ? 0 : direction === 'bottom' ? end
                    : before + (direction === 'up' ? -1 : 1) * innerHeight;
                window.scrollTo({top, behavior: 'instant'});
                return [before, window.scrollY, end].map(Math.round);
            }""",
            args.direction,
        )
        if before != after:
            return f"Scrolled {args.direction}: page offset {before} -> {after} px (bottom is {bottom})."
        if bottom <= 0:
            return (
                "The page did not move: it does not scroll itself (an inner panel may)."
            )
        edge = "top" if args.direction in {"up", "top"} else "bottom"
        return f"The page did not move: it is already at the {edge}, or page scrolling is blocked."

    return await _act(ctx, scroll, screenshot=True)


@tool(
    name="browser_back",
    description="Go back one page in the active tab's history.",
)
async def browser_back(args: NoArgs, ctx: ToolContext) -> str:
    async def back(page: Page) -> None:
        await page.go_back(wait_until="domcontentloaded")

    return await _act(ctx, back)


@tool(
    name="browser_screenshot",
    description=(
        "Attach a PNG screenshot of the active page so the model can see its "
        "layout. Viewport only unless full_page is true: missing content may be "
        "offscreen. Use the snapshot for text and refs."
    ),
)
async def browser_screenshot(args: ScreenshotArgs, ctx: ToolContext) -> ToolOutput:
    async with _hooks(ctx).use(ctx) as session:
        page, notes = await _settled_page(session, await session.page())
        result = await _screenshot(page, full_page=args.full_page)
        return ToolOutput(
            text="\n".join([*notes, result.text]), attachments=result.attachments
        )


async def _screenshot(page: Page, *, full_page: bool = False) -> ToolOutput:
    playwright_error, _ = _playwright_errors()
    try:
        data = await page.screenshot(type="png", full_page=full_page)
        media_type, name = "image/png", "screenshot.png"
        if len(data) > IMAGE_MAX_BYTES:
            data = await page.screenshot(type="jpeg", quality=70, full_page=full_page)
            media_type, name = "image/jpeg", "screenshot.jpg"
    except playwright_error as exc:
        raise ToolError(f"screenshot failed: {_error_text(exc)}") from None
    attachment = attachment_from_bytes(
        data, name=name, media_type=media_type, kind="image"
    )
    return ToolOutput(
        text=f"attached a screenshot of {page.url} ({len(data)} bytes)",
        attachments=[attachment],
    )


@tool(
    name="browser_resize",
    description=(
        "Resize the active tab's viewport to check a responsive layout (CSS "
        "media queries, breakpoints) and return the new snapshot; follow with "
        "browser_screenshot to see it. Only the size changes: there is no touch "
        "or mobile user-agent emulation, and new tabs open at the configured "
        "size."
    ),
)
async def browser_resize(args: ResizeArgs, ctx: ToolContext) -> str:
    async def resize(page: Page) -> None:
        await page.set_viewport_size({"width": args.width, "height": args.height})

    snapshot = await _act(ctx, resize, launched_only="browser_resize")
    return f"Viewport is now {args.width}x{args.height}.\n{snapshot}"


@tool(
    name="browser_close",
    description=(
        "Close the browser, discarding its tabs, cookies and approvals (an "
        "attached browser is only disconnected). The next browser tool starts "
        "a fresh session."
    ),
)
async def browser_close(args: NoArgs, ctx: ToolContext) -> str:
    await _hooks(ctx).reset()
    return "browser closed"
