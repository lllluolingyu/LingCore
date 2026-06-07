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
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool

_MAX_CHARS = 32_000
_MAX_BYTES = 5_000_000  # hard cap on bytes pulled from the socket before we stop
_TIMEOUT = 20.0
_DNS_TIMEOUT = 5.0  # DNS has no other deadline until the HTTP client is built
_MAX_REDIRECTS = 5
_LOCAL_HOSTS = {"localhost", "localhost.localdomain"}
_USER_AGENT = "LingCore/0.0.1"


def _html_to_text(html: str) -> str:
    """Very lightweight HTML → plain text: strip tags, collapse whitespace."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class FetchArgs(BaseModel):
    url: str = Field(description="The URL to fetch (http or https).")


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for addresses fetch_url must never reach when private hosts are off."""
    # An IPv4-mapped IPv6 address (e.g. ::ffff:127.0.0.1) is judged by its
    # embedded IPv4 address, which is what the kernel actually routes to.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
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
    return [info[4][0].split("%", 1)[0] for info in infos]


async def _vet_url(url: str, *, allow_private_hosts: bool) -> str | None:
    """Validate *url* and return the IP to connect to.

    Returns ``None`` when private hosts are allowed (let httpx resolve normally),
    otherwise the single vetted IP the request must be pinned to. Raises
    ``ToolError`` for any disallowed scheme, credentials, or private/local host.
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
        return None

    host = parsed.hostname.rstrip(".").lower()
    if host in _LOCAL_HOSTS or host.endswith(".localhost"):
        raise ToolError(f"private/local host is not allowed: {parsed.hostname!r}")

    port = 443 if parsed.scheme == "https" else 80
    pinned: str | None = None
    for addr in await _resolve_ips(host, port):
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise ToolError(
                f"private/local address is not allowed: {parsed.hostname!r} -> {addr}"
            )
        if pinned is None:
            pinned = addr
    if pinned is None:
        raise ToolError(f"could not resolve host {parsed.hostname!r} to a usable address")
    return pinned


def _build_request(client: httpx.AsyncClient, url: str, pinned_ip: str | None) -> httpx.Request:
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
        if parsed.port:
            host_header = f"{host_header}:{parsed.port}"
        headers["Host"] = host_header
        if parsed.scheme == "https":
            sni = parsed.hostname
    req = client.build_request("GET", target, headers=headers)
    if sni is not None:
        req.extensions["sni_hostname"] = sni
    return req


async def _read_capped(resp: httpx.Response) -> bytes:
    """Stream the body, stopping once _MAX_BYTES have been read."""
    buf = bytearray()
    async for chunk in resp.aiter_bytes():
        buf.extend(chunk)
        if len(buf) >= _MAX_BYTES:
            break
    return bytes(buf[:_MAX_BYTES])


def _is_redirect(status_code: int) -> bool:
    return 300 <= status_code < 400


@tool(description="Fetch a URL and return its text content (HTML is converted to plain text).")
async def fetch_url(args: FetchArgs, ctx: ToolContext) -> str:
    opts = ctx.options.get("fetch_url", {}) if ctx.options else {}
    allow_private_hosts = bool(opts.get("allow_private_hosts", False))
    url = args.url
    pinned = await _vet_url(url, allow_private_hosts=allow_private_hosts)

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
                resp = await client.send(_build_request(client, url, pinned), stream=True)
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
                        body = await _read_capped(resp)
                        location = None
                finally:
                    await resp.aclose()

                if location is None:
                    break
                url = urljoin(url, location)
                pinned = await _vet_url(url, allow_private_hosts=allow_private_hosts)
            else:
                raise ToolError(f"too many redirects fetching {args.url!r}")
    except httpx.HTTPError as e:
        raise ToolError(f"request failed: {e}") from None

    text = body.decode(charset, errors="replace")
    if "html" in content_type:
        text = _html_to_text(text)

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n... (truncated, {len(text) - _MAX_CHARS} more chars)"

    return f"[{status} {url}]\n\n{text}"
