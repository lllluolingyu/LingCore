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
- The context is ephemeral (no stored cookies or credentials), service
  workers are blocked so every request stays routable, downloads are refused,
  and JavaScript dialogs are dismissed and reported.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from lingcore.errors import ConfigError, ToolError
from lingcore.media import attachment_from_bytes
from lingcore.media_types import IMAGE_MAX_BYTES
from lingcore.plugins import PluginContext, PluginHooks
from lingcore.tool_options import (
    FETCH_OPTION_PATH,
    NETWORK_POLICY_KEYS,
    NetworkPolicy,
    bool_option,
    int_option,
    parse_network_policy,
)
from lingcore.tools import ToolContext, ToolOutput, tool
from lingcore.tools.builtin._offload import offload_text
from lingcore.tools.builtin.web import _NonPublic, _vet_url

if TYPE_CHECKING:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Dialog,
        Page,
        Playwright,
        Request,
        Route,
        WebSocketRoute,
    )

OPTION_PATH = "tool_options.browser"
_OPTIONS = frozenset(
    {
        "headless",
        "timeout",
        "executable_path",
        "chromium_sandbox",
        "viewport_width",
        "viewport_height",
        "offload_over_chars",
        "max_chars",
    }
)
_INSTALL_HINT = (
    "install the browser extra and Chromium: "
    "pip install 'lingcore[browser]' && playwright install chromium"
)
_REF = re.compile(r"(?:f[0-9]+)?e[0-9]+")  # iframe refs carry an fN prefix
_MAX_NOTED_HOSTS = 5
_SOCKS_HANDSHAKE_TIMEOUT = 10.0
_RELAY_CHUNK = 64 * 1024
# SOCKS5 reply codes (RFC 1928).
_SOCKS_OK, _SOCKS_NOT_ALLOWED, _SOCKS_UNREACHABLE = 0, 2, 4
_SOCKS_BAD_COMMAND, _SOCKS_BAD_ADDRESS = 7, 8


@dataclass(frozen=True, slots=True)
class BrowserOptions:
    headless: bool = True
    timeout_ms: int = 15_000
    executable_path: str | None = None
    chromium_sandbox: bool = False
    viewport: tuple[int, int] = (1280, 800)
    offload_over_chars: int = 20_000
    max_chars: int = 40_000


def parse_browser_options(raw: object) -> BrowserOptions:
    """Validate the complete ``tool_options.browser`` mapping."""
    if raw is None:
        return BrowserOptions()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{OPTION_PATH} must be a mapping")
    moved = sorted(str(name) for name in raw if name in NETWORK_POLICY_KEYS)
    if moved:
        raise ConfigError(
            f"{OPTION_PATH}.{moved[0]} is not supported: the network policy is "
            f"shared with fetch_url, so set it under {FETCH_OPTION_PATH}"
        )
    unknown = sorted(str(name) for name in raw if name not in _OPTIONS)
    if unknown:
        raise ConfigError(f"unknown {OPTION_PATH} option(s): {', '.join(unknown)}")
    path = OPTION_PATH
    executable = raw.get("executable_path")
    if executable is not None and (not isinstance(executable, str) or not executable):
        raise ConfigError(f"{path}.executable_path must be a non-empty string")
    defaults = BrowserOptions()
    return BrowserOptions(
        headless=bool_option(raw, "headless", True, option_path=path),
        timeout_ms=int_option(
            raw, "timeout", 15, minimum=1, maximum=120, option_path=path
        )
        * 1000,
        executable_path=executable,
        chromium_sandbox=bool_option(raw, "chromium_sandbox", False, option_path=path),
        viewport=(
            int_option(
                raw, "viewport_width", 1280, minimum=320, maximum=3840, option_path=path
            ),
            int_option(
                raw, "viewport_height", 800, minimum=240, maximum=2160, option_path=path
            ),
        ),
        offload_over_chars=int_option(
            raw, "offload_over_chars", defaults.offload_over_chars, option_path=path
        ),
        max_chars=int_option(
            raw, "max_chars", defaults.max_chars, minimum=1_000, option_path=path
        ),
    )


def _playwright_errors() -> tuple[type[Exception], type[Exception]]:
    """Playwright's ``(Error, TimeoutError)``, or a ToolError with install advice."""
    try:
        from playwright.async_api import Error, TimeoutError
    except ImportError:
        raise ToolError(f"Playwright is not installed; {_INSTALL_HINT}") from None
    return Error, TimeoutError


def _error_text(exc: BaseException) -> str:
    """First line of a Playwright error; its call log can be very long."""
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0]


def _normal_host(host: str) -> str:
    return host.rstrip(".").lower()


def _http_form(url: str) -> str | None:
    """``url`` with ws/wss mapped to http/https; ``None`` for any other scheme."""
    parsed = urlparse(url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    if scheme not in ("http", "https"):
        return None
    return parsed._replace(scheme=scheme).geturl()


async def _relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy one direction of a tunnel, passing on a half-close."""
    try:
        while data := await reader.read(_RELAY_CHUNK):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except OSError:
        writer.transport.abort()


class _VettingProxy:
    """A loopback SOCKS5 proxy that applies the network policy per connection.

    Chromium sends it unresolved hostnames, so ``vet`` sees every host the
    browser connects to — including redirect targets Playwright's routing
    never reports — and returns the address to pin, or ``None`` to refuse.
    """

    def __init__(
        self,
        vet: Callable[[str, int], Awaitable[str | None]],
        connect_timeout: float,
    ) -> None:
        self._vet = vet
        self._connect_timeout = connect_timeout
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> str:
        """Listen on an ephemeral loopback port; return the proxy URL."""
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"socks5://127.0.0.1:{port}"

    async def close(self) -> None:
        """Stop listening and drop every tunnel; safe to retry if cancelled."""
        if self._server is not None:
            self._server.close()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        upstream: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(_SOCKS_HANDSHAKE_TIMEOUT):
                target = await self._handshake(reader, writer)
            if target is None:
                return
            host, port = target
            code = _SOCKS_NOT_ALLOWED
            try:
                address = await self._vet(host, port)
            except ToolError:  # unresolvable
                address, code = None, _SOCKS_UNREACHABLE
            if address is None:
                await self._reply(writer, code)
                return
            try:
                async with asyncio.timeout(self._connect_timeout):
                    upstream_reader, upstream = await asyncio.open_connection(
                        address, port
                    )
            except (OSError, TimeoutError):
                await self._reply(writer, _SOCKS_UNREACHABLE)
                return
            await self._reply(writer, _SOCKS_OK)
            await asyncio.gather(
                _relay(reader, upstream), _relay(upstream_reader, writer)
            )
        except (OSError, TimeoutError, asyncio.IncompleteReadError):
            pass
        finally:
            for stream in (writer, upstream):
                if stream is not None:
                    stream.close()

    async def _handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> tuple[str, int] | None:
        """Read a no-auth SOCKS5 CONNECT; ``None`` after refusing anything else."""
        version, count = await reader.readexactly(2)
        methods = await reader.readexactly(count)
        if version != 5 or 0 not in methods:
            writer.write(b"\x05\xff")
            await writer.drain()
            return None
        writer.write(b"\x05\x00")
        await writer.drain()
        version, command, _, kind = await reader.readexactly(4)
        host: str | None = None
        if kind == 1:
            host = str(ipaddress.IPv4Address(await reader.readexactly(4)))
        elif kind == 4:
            host = str(ipaddress.IPv6Address(await reader.readexactly(16)))
        elif kind == 3:
            name = await reader.readexactly((await reader.readexactly(1))[0])
            if name.isascii():  # Chromium sends IDNA (punycode) names
                host = name.decode("ascii")
        port = int.from_bytes(await reader.readexactly(2), "big")
        if version != 5 or command != 1:
            await self._reply(writer, _SOCKS_BAD_COMMAND)
            return None
        if not host:
            await self._reply(writer, _SOCKS_BAD_ADDRESS)
            return None
        return host, port

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, code: int) -> None:
        writer.write(bytes([5, code, 0, 1, 0, 0, 0, 0, 0, 0]))
        await writer.drain()


class BrowserSession:
    """A lazily launched Chromium plus the request policy guarding it.

    Not safe for concurrent use: ``BrowserHooks.use`` serializes callers.
    """

    def __init__(
        self, options: BrowserOptions, policy: NetworkPolicy | None = None
    ) -> None:
        self.options = options
        self.policy = policy or NetworkPolicy()
        self.approved_hosts: set[str] = set()
        self._blocked: list[str] = []
        self._notes: list[str] = []
        self._proxy: _VettingProxy | None = None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: list[Page] = []

    # -- lifecycle ---------------------------------------------------------

    async def page(self) -> Page:
        """The active page, launching the browser on first use."""
        if self._context is None:
            await self._launch()
        if not self._pages:
            assert self._context is not None
            self._track(await self._context.new_page())
        return self._pages[-1]

    async def _launch(self) -> None:
        playwright_error, _ = _playwright_errors()
        from playwright.async_api import async_playwright

        options = self.options
        try:
            # Each handle is stored as soon as it exists so close() can release
            # a launch that fails or is cancelled midway.
            self._proxy = _VettingProxy(self.connect_address, options.timeout_ms / 1000)
            try:
                proxy_url = await self._proxy.start()
            except OSError as exc:
                raise ToolError(
                    f"cannot start the browser's network proxy: {exc}"
                ) from None
            self._playwright = await async_playwright().start()
            # With a SOCKS proxy Playwright also disables Chromium's own DNS,
            # so names resolve only in the proxy. "<-loopback>" removes
            # Chromium's implicit loopback bypass whatever the environment says.
            self._browser = await self._playwright.chromium.launch(
                headless=options.headless,
                executable_path=options.executable_path,
                chromium_sandbox=options.chromium_sandbox,
                proxy={"server": proxy_url, "bypass": "<-loopback>"},
            )
            width, height = options.viewport
            self._context = await self._browser.new_context(
                accept_downloads=False,
                service_workers="block",
                viewport={"width": width, "height": height},
            )
            self._context.set_default_timeout(options.timeout_ms)
            self._context.on("page", self._track)
            await self._context.route("**/*", self._route)
            await self._context.route_web_socket("**/*", self._route_web_socket)
        except playwright_error as exc:
            await self.close()
            hint = "" if options.executable_path else f" ({_INSTALL_HINT})"
            raise ToolError(
                f"cannot start the browser: {_error_text(exc)}{hint}"
            ) from None
        except BaseException:
            await self.close()
            raise

    def _track(self, page: Page) -> None:
        if page in self._pages:
            return
        if self._pages:
            self._notes.append("A new tab opened and is now active.")
        self._pages.append(page)
        page.on("close", self._untrack)
        page.on("dialog", self._dismiss)

    def _untrack(self, page: Page) -> None:
        if page in self._pages:
            self._pages.remove(page)

    async def _dismiss(self, dialog: Dialog) -> None:
        self._notes.append(
            f"Dismissed a JavaScript {dialog.type} dialog: {dialog.message[:200]!r}"
        )
        try:
            await dialog.dismiss()
        except Exception:
            pass

    async def close(self) -> None:
        """Release every handle; safe to call repeatedly or after a failed launch.

        Each handle is dropped only once its own cleanup has finished, so a
        close interrupted by cancellation resumes where it stopped when called
        again.
        """
        self._pages = []
        if self._context is not None:
            await _quietly(self._context.close())
            self._context = None
        if self._browser is not None:
            await _quietly(self._browser.close())
            self._browser = None
        if self._playwright is not None:
            await _quietly(self._playwright.stop())
            self._playwright = None
        if self._proxy is not None:
            await self._proxy.close()
            self._proxy = None

    # -- network policy ----------------------------------------------------

    @staticmethod
    async def url_error(url: str) -> str | None:
        """Why ``url`` is malformed for the policy (scheme, credentials, port).

        Purely syntactic, so it never resolves the host: ``allow_private_hosts``
        here only skips ``_vet_url``'s DNS step, never its URL validation.
        """
        http_url = _http_form(url)
        if http_url is None:
            return f"unsupported scheme {urlparse(url).scheme!r}"
        try:
            await _vet_url(http_url, allow_private_hosts=True)
        except ToolError as exc:
            return str(exc)
        return None

    async def blocked_reason(self, url: str) -> _NonPublic | str | None:
        """Why the page may not request ``url``; ``None`` when it may.

        Returns the non-public target for a local host, or a plain reason for a
        malformed URL or unresolvable host. The whole URL is validated before
        any approval applies, and nothing is cached.
        """
        reason = await self.url_error(url)
        if reason is not None or self.policy.allow_private_hosts:
            return reason
        try:
            target = await _vet_url(
                _http_form(url) or url,
                allow_private_hosts=False,
                allowed_networks=self.policy.allowed_networks,
            )
        except ToolError as exc:
            return str(exc)
        blocked = target.non_public
        if blocked is None or _normal_host(blocked.hostname) in self.approved_hosts:
            return None
        return blocked

    async def connect_address(self, host: str, port: int) -> str | None:
        """The address the proxy connects to for ``host:port``; ``None`` refuses.

        An unresolvable host raises ``ToolError``. A refused host is recorded
        for the next result.
        """
        literal = f"[{host}]" if ":" in host else host
        target = await _vet_url(
            f"http://{literal}:{port}/",
            allow_private_hosts=self.policy.allow_private_hosts,
            allowed_networks=self.policy.allowed_networks,
        )
        blocked = target.non_public
        if blocked is not None and _normal_host(host) not in self.approved_hosts:
            self._record_block(host)
            return None
        return target.pinned or host

    def _record_block(self, url_or_host: str) -> None:
        host = urlparse(url_or_host).hostname or url_or_host
        if host not in self._blocked:
            self._blocked.append(host)

    async def _route(self, route: Route, request: Request) -> None:
        try:
            if await self.url_error(request.url) is None:
                await route.continue_()
            else:
                self._record_block(request.url)
                await route.abort("blockedbyclient")
        except Exception:
            # The page may close while a request is in flight; Playwright would
            # otherwise log the handler error as unhandled.
            pass

    async def _route_web_socket(self, ws: WebSocketRoute) -> None:
        try:
            if await self.url_error(ws.url) is None:
                ws.connect_to_server()
            else:
                self._record_block(ws.url)
                await ws.close(code=1008, reason="blocked by LingCore")
        except Exception:
            pass

    async def authorize_navigation(self, ctx: ToolContext, url: str) -> None:
        """Refuse or put a non-public navigation target to the user."""
        reason = await self.blocked_reason(url)
        if reason is None:
            return
        if isinstance(reason, str):
            raise ToolError(f"cannot open {url!r}: {reason}")
        if not self.policy.confirm_private_hosts or ctx.confirm is None:
            raise ToolError(reason.error())
        if not await ctx.confirm(reason.prompt(url, tool="browser_navigate")):
            raise ToolError(f"user declined: {reason.error()}")
        self.approved_hosts.add(_normal_host(reason.hostname))

    # -- rendering ---------------------------------------------------------

    def drain_notes(self) -> list[str]:
        notes, self._notes = self._notes, []
        if self._blocked:
            hosts = ", ".join(self._blocked[:_MAX_NOTED_HOSTS])
            more = len(self._blocked) - _MAX_NOTED_HOSTS
            if more > 0:
                hosts += f" and {more} more"
            notes.append(
                f"Blocked requests to non-public or unsupported hosts: {hosts}."
            )
            self._blocked = []
        return notes

    async def render(self, ctx: ToolContext, notes: list[str] | None = None) -> str:
        """The active page as a titled AI snapshot plus any pending notes."""
        playwright_error, _ = _playwright_errors()
        page = await self.page()
        try:
            title = await page.title()
            snapshot = await page.aria_snapshot(mode="ai")
        except playwright_error as exc:
            raise ToolError(f"cannot read the page: {_error_text(exc)}") from None
        lines = [f"[{title or 'untitled'}] {page.url}"]
        if len(self._pages) > 1:
            lines.append(f"({len(self._pages)} tabs are open; the newest is active.)")
        lines.extend((notes or []) + self.drain_notes())
        text = "\n".join(lines) + "\n\n" + snapshot
        return offload_text(
            ctx,
            source="browser",
            text=text,
            threshold=self.options.offload_over_chars,
            fallback_max_chars=self.options.max_chars,
        )


async def _quietly(cleanup: Awaitable[object]) -> None:
    try:
        await cleanup
    except Exception:
        pass


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
            self._session = BrowserSession(browser, policy)
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


async def _act(
    ctx: ToolContext,
    action: Any,
    *,
    settle: bool = True,
    navigate_to: str | None = None,
) -> str:
    """Run ``action(page)`` holding the browser and return a fresh snapshot.

    ``navigate_to`` is authorized against the session the action will use,
    before the browser launches.
    """
    playwright_error, playwright_timeout = _playwright_errors()
    async with _hooks(ctx).use(ctx) as session:
        if navigate_to is not None:
            await session.authorize_navigation(ctx, navigate_to)
        page = await session.page()
        notes: list[str] = []
        try:
            await action(page)
            if settle:
                # A click may start a navigation; wait briefly for it to land.
                await page.wait_for_load_state("domcontentloaded", timeout=5_000)
        except playwright_timeout as exc:
            notes.append(f"Timed out: {_error_text(exc)}")
        except playwright_error as exc:
            raise ToolError(
                " ".join([_error_text(exc), *session.drain_notes()])
            ) from None
        return await session.render(ctx, notes)


class NavigateArgs(BaseModel):
    url: str = Field(description="The http(s) URL to open in the active tab.")


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


class NoArgs(BaseModel):
    pass


def _locator(page: Page, ref: str) -> Any:
    ref = ref.strip().removeprefix("ref=")
    if not _REF.fullmatch(ref):
        raise ToolError(f"invalid element ref {ref!r}; use a ref like 'e12'")
    return page.locator(f"aria-ref={ref}")


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

    return await _act(ctx, go, settle=False, navigate_to=args.url)


@tool(
    name="browser_snapshot",
    description="Return the active page's current accessibility snapshot.",
)
async def browser_snapshot(args: NoArgs, ctx: ToolContext) -> str:
    async def noop(page: Page) -> None:
        pass

    return await _act(ctx, noop, settle=False)


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
        "Replace the text of the input with the given ref, optionally pressing "
        "Enter, and return the new snapshot."
    ),
)
async def browser_type(args: TypeArgs, ctx: ToolContext) -> str:
    async def fill(page: Page) -> None:
        field = _locator(page, args.ref)
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
    name="browser_back",
    description="Go back one page in the active tab's history.",
)
async def browser_back(args: NoArgs, ctx: ToolContext) -> str:
    async def back(page: Page) -> None:
        await page.go_back(wait_until="domcontentloaded")

    return await _act(ctx, back, settle=False)


@tool(
    name="browser_screenshot",
    description=(
        "Attach a PNG screenshot of the active page so the model can see its "
        "layout. Use the snapshot for text and refs."
    ),
)
async def browser_screenshot(args: ScreenshotArgs, ctx: ToolContext) -> ToolOutput:
    playwright_error, _ = _playwright_errors()
    async with _hooks(ctx).use(ctx) as session:
        page = await session.page()
        try:
            data = await page.screenshot(type="png", full_page=args.full_page)
            media_type, name = "image/png", "screenshot.png"
            if len(data) > IMAGE_MAX_BYTES:
                data = await page.screenshot(
                    type="jpeg", quality=70, full_page=args.full_page
                )
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
    name="browser_close",
    description=(
        "Close the browser, discarding its tabs, cookies and approvals. The "
        "next browser tool starts a fresh one."
    ),
)
async def browser_close(args: NoArgs, ctx: ToolContext) -> str:
    await _hooks(ctx).reset()
    return "browser closed"
