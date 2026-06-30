"""Tests for fetch_url."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from lingcore.errors import ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.web import FetchArgs, fetch_url


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Resolve hosts deterministically so tests never touch real DNS.

    Numeric IP literals (including decimal/hex/octal) are resolved by the real
    resolver — which never hits the network for them — so the literal-bypass
    tests exercise the production code path. Named hosts use a fixed mapping.
    """
    import socket as _socket

    names = {
        "example.com": ["93.184.216.34"],
        "private.example": ["10.0.0.5"],
    }

    async def _fake(host: str, port: int) -> list[str]:
        if host in names:
            return names[host]
        infos = _socket.getaddrinfo(host, port, proto=_socket.IPPROTO_TCP)
        return [info[4][0].split("%", 1)[0] for info in infos]

    monkeypatch.setattr("lingcore.tools.builtin.web._resolve_ips", _fake)


class _FakeResponse:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(
        self,
        text: str = "",
        *,
        content_type: str = "text/plain",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        charset: str = "utf-8",
    ):
        self._body = text.encode(charset)
        self.status_code = status_code
        self.headers = {"content-type": content_type, **(headers or {})}
        self.charset_encoding = charset
        self.closed = False

    async def aiter_bytes(self):
        yield self._body

    async def aclose(self):
        self.closed = True


class _FakeClient:
    """Async-context client that returns queued responses in order."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def build_request(self, method, url, headers=None):
        return httpx.Request(method, url, headers=headers)

    async def send(self, request, stream=False):
        self.requests.append(request)
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_client(responses: list[_FakeResponse]):
    client = _FakeClient(responses)
    return patch("lingcore.tools.builtin.web.httpx.AsyncClient", return_value=client), client


async def test_fetch_plain(ctx):
    p, _ = _patch_client([_FakeResponse("hello world")])
    with p:
        out = await fetch_url(FetchArgs(url="https://example.com/api"), ctx)
    assert "hello world" in out
    assert "200" in out


async def test_fetch_html_stripped(ctx):
    html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"
    p, _ = _patch_client([_FakeResponse(html, content_type="text/html")])
    with p:
        out = await fetch_url(FetchArgs(url="https://example.com/"), ctx)
    assert "<html>" not in out
    assert "Title" in out
    assert "Content here" in out


async def test_fetch_offloads_large_body(ctx):
    p, _ = _patch_client([_FakeResponse("x" * 40_000)])
    with p:
        out = await fetch_url(FetchArgs(url="https://example.com/big"), ctx)
    assert "full output" in out and ".lingcore/tool-output/fetch-" in out
    assert "200" in out  # status line stays inline


async def test_fetch_truncates_when_offload_disabled(tmp_path):
    ctx = ToolContext(
        workspace=tmp_path,
        options={"fetch_url": {"offload_over_chars": 0, "max_chars": 2000}},
    )
    p, _ = _patch_client([_FakeResponse("x" * 40_000)])
    with p:
        out = await fetch_url(FetchArgs(url="https://example.com/big"), ctx)
    assert "truncated" in out


async def test_fetch_caps_body_bytes(ctx, monkeypatch):
    # The body is streamed and capped at _MAX_BYTES before decoding, so a huge
    # response never lands fully in memory.
    monkeypatch.setattr("lingcore.tools.builtin.web._MAX_BYTES", 50)
    p, _ = _patch_client([_FakeResponse("x" * 10_000)])
    with p:
        out = await fetch_url(FetchArgs(url="https://example.com/big"), ctx)
    body = out.split("\n\n", 1)[1]
    assert body == "x" * 50


async def test_fetch_pins_connection_to_vetted_ip(ctx):
    # The connection targets the vetted IP, but Host + TLS SNI stay the hostname
    # so DNS can't be rebound between validation and connect.
    p, client = _patch_client([_FakeResponse("ok")])
    with p:
        await fetch_url(FetchArgs(url="https://example.com/path?q=1"), ctx)
    req = client.requests[0]
    assert req.url.host == "93.184.216.34"
    assert req.headers["Host"] == "example.com"
    assert req.extensions["sni_hostname"] == "example.com"


async def test_fetch_disables_keepalive(ctx):
    # Keep-alive must be off so a redirect to another host sharing the same IP
    # can't reuse the first hop's TLS connection (and skip the new hop's SNI).
    client = _FakeClient([_FakeResponse("ok")])
    with patch(
        "lingcore.tools.builtin.web.httpx.AsyncClient", return_value=client
    ) as MockClient:
        await fetch_url(FetchArgs(url="https://example.com/"), ctx)
    assert MockClient.call_args.kwargs["limits"].max_keepalive_connections == 0


async def test_fetch_rejects_non_http(ctx):
    with pytest.raises(ToolError, match="http"):
        await fetch_url(FetchArgs(url="ftp://example.com/file"), ctx)


async def test_fetch_rejects_invalid_port(ctx):
    with pytest.raises(ToolError, match="invalid port"):
        await fetch_url(FetchArgs(url="http://example.com:99999/"), ctx)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://user:pass@example.com/",
    ],
)
async def test_fetch_rejects_private_or_credentialed_urls(ctx, url):
    with pytest.raises(ToolError):
        await fetch_url(FetchArgs(url=url), ctx)


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/",  # decimal-encoded 127.0.0.1
        "http://0x7f000001/",  # hex-encoded 127.0.0.1
        "http://0177.0.0.1/",  # octal-encoded 127.0.0.1
        "http://0/",           # shorthand 0.0.0.0
    ],
)
async def test_fetch_rejects_alt_encoded_loopback(ctx, url):
    with pytest.raises(ToolError, match="private/local"):
        await fetch_url(FetchArgs(url=url), ctx)


async def test_fetch_rejects_dns_resolving_to_private(ctx):
    # A public-looking hostname that resolves to an RFC1918 address must be
    # refused — this is the DNS-resolution check, not a literal-IP check.
    with pytest.raises(ToolError, match="private/local"):
        await fetch_url(FetchArgs(url="http://private.example/data"), ctx)


async def test_fetch_allows_private_with_opt_in(tmp_path):
    ctx = ToolContext(
        workspace=tmp_path, options={"fetch_url": {"allow_private_hosts": True}}
    )
    p, client = _patch_client([_FakeResponse("ok")])
    with p:
        out = await fetch_url(FetchArgs(url="http://localhost:11434/api/tags"), ctx)
    assert "ok" in out
    # Opt-in skips pinning, so the request keeps the original host untouched.
    assert client.requests[0].url.host == "localhost"


async def test_fetch_rejects_redirect_to_private_host(ctx):
    redirect = _FakeResponse(
        "", status_code=302, headers={"location": "http://127.0.0.1/secret"}
    )
    p, client = _patch_client([redirect])
    with p:
        with pytest.raises(ToolError, match="private/local"):
            await fetch_url(FetchArgs(url="https://example.com/"), ctx)
    assert len(client.requests) == 1  # the redirect target is never requested


async def test_fetch_http_error(ctx):
    with patch("lingcore.tools.builtin.web.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="request failed"):
            await fetch_url(FetchArgs(url="https://example.com/"), ctx)
