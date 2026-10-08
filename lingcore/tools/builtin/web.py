"""fetch_url — retrieve a URL and return its text content.

HTML is stripped to readable text via a lightweight regex pass so the model
receives prose rather than tag soup. JSON, plain text, and Markdown are
returned as-is. Output is truncated to _MAX_CHARS to stay within context.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

from lingcore import __version__ as _lingcore_version
from lingcore.errors import ConfigError, ToolError
from lingcore.tool_options import IPNetwork, parse_fetch_allowed_networks
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._offload import DEFAULT_OFFLOAD_OVER_CHARS, offload_text

_MAX_CHARS = 32_000
_MAX_BYTES = 5_000_000  # hard cap on bytes pulled from the socket before we stop
_TIMEOUT = 20.0
_DNS_TIMEOUT = 5.0  # DNS has no other deadline until the HTTP client is built
_MAX_REDIRECTS = 5
_LOCAL_HOSTS = {"localhost", "localhost.localdomain"}
_USER_AGENT = f"LingCore/{_lingcore_version}"  # tracks the package version
# The RFC 2544/5180 benchmarking ranges. Fake-IP proxies (Clash/mihomo, Surge,
# ...) in TUN mode answer every DNS query from them and map the address back to
# the hostname, so a block here usually means such a proxy, not a private host.
_FAKE_IP_NETWORKS = (
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("2001:2::/48"),
)


def _html_to_text(html: str) -> str:
    """Very lightweight HTML → plain text: strip tags, collapse whitespace."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class FetchArgs(BaseModel):
    url: str = Field(description="The URL to fetch (http or https).")


def _is_blocked_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    allowed_networks: tuple[IPNetwork, ...] = (),
) -> bool:
    """True for addresses fetch_url must never reach when private hosts are off.

    ``allowed_networks`` are operator-configured exemptions, checked first.
    """
    # An IPv4-mapped IPv6 address (e.g. ::ffff:127.0.0.1) is judged by its
    # embedded IPv4 address, which is what the kernel actually routes to.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if any(ip in network for network in allowed_networks):
        return False
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def _resolve_ips(host: str, port: int) -> list[str]:
    """Resolve *host* to its IP literals.

    getaddrinfo also parses numeric literals — decimal/hex/octal IPv4 and IPv6 —
    so this is what closes encoding-based bypasses (e.g. http://2130706433/ for
    127.0.0.1); those never reach the network. Real hostnames do a DNS lookup.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, port, proto=socket.IPPROTO_TCP),
            timeout=_DNS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise ToolError(f"DNS resolution timed out for {host!r}") from None
    except socket.gaierror as e:
        raise ToolError(f"could not resolve host {host!r}: {e}") from None
    # Drop any IPv6 zone id (e.g. fe80::1%eth0) before the address is parsed.
    return [str(info[4][0]).split("%", 1)[0] for info in infos]


@dataclass(frozen=True, slots=True)
class _NonPublic:
    """Why a URL's host is not on the public web (named host, or an address)."""

    hostname: str
    addr: str | None = None  # None when the name itself is local (localhost)

    def _fake_ip_hint(self, advice: str) -> str:
        if self.addr is None:
            return ""
        ip = ipaddress.ip_address(self.addr)
        network = next((n for n in _FAKE_IP_NETWORKS if ip in n), None)
        if network is None:
            return ""
        ranges = ", ".join(map(str, _FAKE_IP_NETWORKS))
        return (
            f" ({network} is a fake-IP range of TUN-mode proxies such as"
            f" Clash/mihomo or Surge; {advice}, add {ranges} to"
            " tool_options.fetch_url.allowed_networks)"
        )

    def error(self) -> str:
        if self.addr is None:
            return f"private/local host is not allowed: {self.hostname!r}"
        return (
            f"private/local address is not allowed: {self.hostname!r} -> "
            f"{self.addr}{self._fake_ip_hint('to fetch through one')}"
        )

    def prompt(self, url: str) -> str:
        target = (
            f"{self.hostname} is a local host"
            if self.addr is None
            else f"{self.hostname} resolves to non-public address {self.addr}"
        )
        hint = self._fake_ip_hint("to stop asking")
        return f"Allow fetch_url to reach a non-public address? {url} — {target}{hint}"


@dataclass(frozen=True, slots=True)
class _Target:
    pinned: str | None  # the IP to connect to; None lets httpx resolve
    non_public: _NonPublic | None = None  # set when the host needs approval


async def _vet_url(
    url: str,
    *,
    allow_private_hosts: bool,
    allowed_networks: tuple[IPNetwork, ...] = (),
) -> _Target:
    """Validate *url* and choose the IP to connect to.

    With private hosts allowed nothing is pinned (httpx resolves normally).
    Otherwise the request is pinned to one vetted IP, and a host that is local
    or resolves to any non-public address is reported in ``non_public`` for
    the caller to refuse or put to the user. Raises ``ToolError`` for a
    disallowed scheme, credentials, or an unresolvable host.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError(f"only http/https URLs are supported: {url!r}")
    if not parsed.hostname:
        raise ToolError(f"url must include a hostname: {url!r}")
    if parsed.username or parsed.password:
        raise ToolError("URLs with embedded credentials are not supported")
    # Accessing .port validates the 0-65535 range; surface a clean tool error
    # (recoverable by the model) instead of leaking an internal ValueError.
    try:
        parsed.port
    except ValueError:
        raise ToolError(f"invalid port in url: {url!r}") from None
    if allow_private_hosts:
        return _Target(pinned=None)

    host = parsed.hostname.rstrip(".").lower()
    non_public: _NonPublic | None = None
    if host in _LOCAL_HOSTS or host.endswith(".localhost"):
        non_public = _NonPublic(parsed.hostname)

    port = 443 if parsed.scheme == "https" else 80
    pinned: str | None = None
    for addr in await _resolve_ips(host, port):
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if non_public is None and _is_blocked_ip(ip, allowed_networks):
            non_public = _NonPublic(parsed.hostname, addr)
        if pinned is None:
            pinned = addr
    if pinned is None:
        raise ToolError(
            f"could not resolve host {parsed.hostname!r} to a usable address"
        )
    return _Target(pinned=pinned, non_public=non_public)


async def _authorize(
    ctx: ToolContext,
    url: str,
    target: _Target,
    *,
    ask: bool,
    approved: set[tuple[str, str | None]],
) -> str | None:
    """Return the IP to pin, asking the user before a non-public target.

    Without a confirmation handler (or with asking turned off) a non-public
    target is refused, as before. An approval covers that host and address for
    the rest of this one fetch, so a redirect back to it doesn't ask twice; any
    other non-public hop asks again.
    """
    blocked = target.non_public
    if blocked is None:
        return target.pinned
    key = (blocked.hostname, target.pinned)
    if key in approved:
        return target.pinned
    if not ask or ctx.confirm is None:
        raise ToolError(blocked.error())
    if not await ctx.confirm(blocked.prompt(url)):
        raise ToolError(f"user declined: {blocked.error()}")
    approved.add(key)
    return target.pinned


def _build_request(
    client: httpx.AsyncClient, url: str, pinned_ip: str | None
) -> httpx.Request:
    """Build a GET request, pinning the connection to *pinned_ip* when set.

    Pinning rewrites the authority to the vetted IP but keeps the Host header
    and (for TLS) the SNI/cert-verification hostname as the original host. That
    closes the DNS-rebinding window — httpx never re-resolves the name — while
    keeping certificate verification bound to the hostname (verified live).
    """
    headers = {"User-Agent": _USER_AGENT}
    target = url
    sni: str | None = None
    if pinned_ip is not None:
        parsed = urlparse(url)
        authority = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
        if parsed.port:
            authority = f"{authority}:{parsed.port}"
        target = parsed._replace(netloc=authority).geturl()
        host_header = parsed.hostname or ""
        # ``urlparse().hostname`` strips the brackets from an IPv6 literal, but
        # an HTTP Host header must carry the bracketed authority.
        if ":" in host_header:
            host_header = f"[{host_header}]"
        if parsed.port:
            host_header = f"{host_header}:{parsed.port}"
        headers["Host"] = host_header
        if parsed.scheme == "https":
            sni = parsed.hostname
    req = client.build_request("GET", target, headers=headers)
    if sni is not None:
        req.extensions["sni_hostname"] = sni
    return req


async def _read_capped(resp: httpx.Response, max_bytes: int) -> bytes:
    """Stream the body, stopping once ``max_bytes`` have been read."""
    buf = bytearray()
    async for chunk in resp.aiter_bytes():
        buf.extend(chunk)
        if len(buf) >= max_bytes:
            break
    return bytes(buf[:max_bytes])


def _is_redirect(status_code: int) -> bool:
    return 300 <= status_code < 400


@tool(
    description="Fetch a URL and return its text content (HTML is converted to plain text)."
)
async def fetch_url(args: FetchArgs, ctx: ToolContext) -> str:
    opts = ctx.options.get("fetch_url", {}) if ctx.options else {}
    allow_private_hosts = bool(opts.get("allow_private_hosts", False))
    # Ask the user before a local/non-public target instead of refusing it.
    ask = bool(opts.get("confirm_private_hosts", True))
    try:
        allowed_networks = parse_fetch_allowed_networks(opts)
    except ConfigError as exc:
        raise ToolError(str(exc)) from None
    # Cap on bytes pulled from the socket. Honors the profile's
    # tool_options.fetch_url.max_bytes (falling back to the module default),
    # so a profile can tighten the fetch size without a code change.
    max_bytes = int(opts.get("max_bytes", _MAX_BYTES))
    url = args.url
    approved: set[tuple[str, str | None]] = set()

    async def vet(url: str) -> str | None:
        target = await _vet_url(
            url,
            allow_private_hosts=allow_private_hosts,
            allowed_networks=allowed_networks,
        )
        return await _authorize(ctx, url, target, ask=ask, approved=approved)

    pinned = await vet(url)

    status: int | None = None
    content_type = ""
    charset = "utf-8"
    body = b""
    # No keep-alive: SNI is consumed when a TLS connection is opened, not per
    # request. Reusing a pooled connection across redirect hops that pin to the
    # same IP would skip the new hop's SNI/cert check, so force a fresh
    # connection (and handshake) for every hop.
    limits = httpx.Limits(max_keepalive_connections=0)
    try:
        async with httpx.AsyncClient(
            follow_redirects=False, timeout=_TIMEOUT, limits=limits
        ) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                resp = await client.send(
                    _build_request(client, url, pinned), stream=True
                )
                try:
                    if _is_redirect(resp.status_code):
                        location = resp.headers.get("location")
                        if not location:
                            raise ToolError(
                                f"redirect response missing Location header: {url!r}"
                            )
                    else:
                        status = resp.status_code
                        content_type = resp.headers.get("content-type", "")
                        charset = resp.charset_encoding or "utf-8"
                        body = await _read_capped(resp, max_bytes)
                        location = None
                finally:
                    await resp.aclose()

                if location is None:
                    break
                url = urljoin(url, location)
                pinned = await vet(url)
            else:
                raise ToolError(f"too many redirects fetching {args.url!r}")
    except httpx.HTTPError as e:
        raise ToolError(f"request failed: {e}") from None

    text = body.decode(charset, errors="replace")
    if "html" in content_type:
        text = _html_to_text(text)

    # Large pages are staged to a workspace file (read the rest with read_file)
    # rather than dumped inline; small ones stay inline.
    body_text = offload_text(
        ctx,
        source="fetch",
        text=text,
        threshold=int(opts.get("offload_over_chars", DEFAULT_OFFLOAD_OVER_CHARS)),
        fallback_max_chars=int(opts.get("max_chars", _MAX_CHARS)),
    )
    return f"[{status} {url}]\n\n{body_text}"
