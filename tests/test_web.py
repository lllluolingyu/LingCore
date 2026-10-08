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
        # What a TUN-mode fake-IP proxy (Clash/mihomo, Surge) hands out.
        "proxied.example": ["198.18.0.41", "2001:2::29"],
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
    return patch(
        "lingcore.tools.builtin.web.httpx.AsyncClient", return_value=client
    ), client


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


async def test_fetch_respects_max_bytes_option(tmp_path):
    # tool_options.fetch_url.max_bytes caps the body without touching the module
    # constant — the daily profile relies on this to bound fetch size.
    ctx = ToolContext(
        workspace=tmp_path,
        options={
            "fetch_url": {"max_bytes": 20, "offload_over_chars": 0, "max_chars": 5000}
        },
    )
    p, _ = _patch_client([_FakeResponse("y" * 10_000)])
    with p:
        out = await fetch_url(FetchArgs(url="https://example.com/big"), ctx)
    body = out.split("\n\n", 1)[1]
    assert body == "y" * 20


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


async def test_fetch_preserves_ipv6_brackets_in_host_header(ctx):
    # urlparse.hostname drops IPv6 brackets; the Host header must not.
    p, client = _patch_client([_FakeResponse("ok")])
    with p:
        await fetch_url(FetchArgs(url="http://[2606:4700:4700::1111]:8080/"), ctx)
    req = client.requests[0]
    assert req.url.host == "2606:4700:4700::1111"
    assert req.headers["Host"] == "[2606:4700:4700::1111]:8080"


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
        "http://0/",  # shorthand 0.0.0.0
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


async def test_fetch_names_fake_ip_proxy_when_blocked(ctx):
    with pytest.raises(ToolError, match="fake-IP range") as exc:
        await fetch_url(FetchArgs(url="https://proxied.example/"), ctx)
    assert "198.18.0.0/15, 2001:2::/48 to" in str(exc.value)
    assert "tool_options.fetch_url.allowed_networks" in str(exc.value)


async def test_fetch_allowed_networks_must_cover_every_answer(tmp_path):
    # Exempting only the IPv4 range still refuses the host: every resolved
    # address is vetted, and the IPv6 fake IP is not covered.
    ctx = _fake_ip_ctx(tmp_path, ["198.18.0.0/15"])
    with pytest.raises(ToolError, match="2001:2::29"):
        await fetch_url(FetchArgs(url="https://proxied.example/"), ctx)


def _fake_ip_ctx(tmp_path, networks):
    return ToolContext(
        workspace=tmp_path,
        options={"fetch_url": {"allowed_networks": networks}},
    )


async def test_fetch_allowed_networks_exempts_fake_ip_and_still_pins(tmp_path):
    ctx = _fake_ip_ctx(tmp_path, ["198.18.0.0/15", "2001:2::/48"])
    p, client = _patch_client([_FakeResponse("sunny")])
    with p:
        out = await fetch_url(FetchArgs(url="https://proxied.example/w"), ctx)
    assert "sunny" in out
    req = client.requests[0]
    # The proxy maps the fake IP back to the hostname; Host/SNI carry it.
    assert req.url.host == "198.18.0.41"
    assert req.headers["Host"] == "proxied.example"
    assert req.extensions["sni_hostname"] == "proxied.example"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://private.example/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
async def test_fetch_allowed_networks_exempts_only_listed_ranges(tmp_path, url):
    ctx = _fake_ip_ctx(tmp_path, ["198.18.0.0/15"])
    with pytest.raises(ToolError, match="private/local"):
        await fetch_url(FetchArgs(url=url), ctx)


async def test_fetch_allowed_networks_rechecks_redirects(tmp_path):
    ctx = _fake_ip_ctx(tmp_path, ["198.18.0.0/15", "2001:2::/48"])
    redirect = _FakeResponse(
        "", status_code=302, headers={"location": "http://private.example/x"}
    )
    p, client = _patch_client([redirect])
    with p:
        with pytest.raises(ToolError, match="private/local"):
            await fetch_url(FetchArgs(url="https://proxied.example/"), ctx)
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "networks", ["198.18.0.0/15", ["198.18.0.1/15"], ["not-a-net"], [""]]
)
async def test_fetch_rejects_invalid_allowed_networks(tmp_path, networks):
    with pytest.raises(ToolError, match="allowed_networks"):
        await fetch_url(
            FetchArgs(url="https://example.com/"), _fake_ip_ctx(tmp_path, networks)
        )


def test_profile_load_rejects_invalid_allowed_networks(tmp_path, monkeypatch):
    from lingcore.config import AgentProfile
    from lingcore.errors import ConfigError

    monkeypatch.setenv("TEST_KEY", "sk-test")
    path = tmp_path / "profile.yaml"
    path.write_text(
        "name: t\n"
        "llm: {model: m, base_url: http://localhost:1/v1, api_key_env: TEST_KEY}\n"
        "persona: {system_prompt: hi}\n"
        "tools: [fetch_url]\n"
        "tool_options: {fetch_url: {allowed_networks: [198.18.0.1/15]}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="allowed_networks"):
        AgentProfile.load(path)


class _Confirm:
    """Records confirmation prompts and answers each with ``answer``."""

    def __init__(self, answer: bool):
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.answer


def _confirm_ctx(tmp_path, answer, **fetch_options):
    confirm = _Confirm(answer)
    ctx = ToolContext(
        workspace=tmp_path, confirm=confirm, options={"fetch_url": fetch_options}
    )
    return ctx, confirm


async def test_fetch_asks_before_non_public_address_and_pins_it(tmp_path):
    ctx, confirm = _confirm_ctx(tmp_path, True)
    p, client = _patch_client([_FakeResponse("sunny")])
    with p:
        out = await fetch_url(FetchArgs(url="https://proxied.example/w"), ctx)
    assert "sunny" in out
    (prompt,) = confirm.prompts
    assert "https://proxied.example/w" in prompt
    assert "resolves to non-public address 198.18.0.41" in prompt
    assert "to stop asking" in prompt  # names the allowed_networks fix
    # Approval doesn't drop pinning: the request goes to the vetted address.
    req = client.requests[0]
    assert req.url.host == "198.18.0.41"
    assert req.extensions["sni_hostname"] == "proxied.example"


async def test_fetch_declined_non_public_address_is_refused(tmp_path):
    ctx, confirm = _confirm_ctx(tmp_path, False)
    p, client = _patch_client([_FakeResponse("secret")])
    with p:
        with pytest.raises(ToolError, match="user declined: private/local"):
            await fetch_url(FetchArgs(url="http://private.example/"), ctx)
    assert len(confirm.prompts) == 1
    assert client.requests == []


async def test_fetch_asks_for_localhost_and_pins_loopback(tmp_path):
    ctx, confirm = _confirm_ctx(tmp_path, True)
    p, client = _patch_client([_FakeResponse("ok")])
    with p:
        await fetch_url(FetchArgs(url="http://localhost:11434/api/tags"), ctx)
    assert "localhost is a local host" in confirm.prompts[0]
    req = client.requests[0]
    assert req.url.host in {"127.0.0.1", "::1"}
    assert req.headers["Host"] == "localhost:11434"


async def test_fetch_confirm_private_hosts_off_refuses_without_asking(tmp_path):
    ctx, confirm = _confirm_ctx(tmp_path, True, confirm_private_hosts=False)
    with pytest.raises(ToolError, match="private/local"):
        await fetch_url(FetchArgs(url="http://private.example/"), ctx)
    assert confirm.prompts == []


async def test_fetch_public_host_never_asks(tmp_path):
    ctx, confirm = _confirm_ctx(tmp_path, False)
    p, _ = _patch_client([_FakeResponse("ok")])
    with p:
        await fetch_url(FetchArgs(url="https://example.com/"), ctx)
    assert confirm.prompts == []


async def test_fetch_asks_again_for_each_new_non_public_redirect_hop(tmp_path):
    ctx, confirm = _confirm_ctx(tmp_path, True)
    hops = [
        _FakeResponse("", status_code=302, headers={"location": "/again"}),
        _FakeResponse(
            "", status_code=302, headers={"location": "http://private.example/"}
        ),
        _FakeResponse("done"),
    ]
    p, client = _patch_client(hops)
    with p:
        out = await fetch_url(FetchArgs(url="https://proxied.example/"), ctx)
    assert "done" in out
    # The same approved host is not re-asked; the new private host is.
    assert len(confirm.prompts) == 2
    assert "private.example" in confirm.prompts[1]
    assert len(client.requests) == 3


async def test_fetch_declined_redirect_hop_is_never_requested(tmp_path):
    confirm = _ConfirmSequence([True, False])
    ctx = ToolContext(workspace=tmp_path, confirm=confirm)
    redirect = _FakeResponse(
        "", status_code=302, headers={"location": "http://127.0.0.1/secret"}
    )
    p, client = _patch_client([redirect])
    with p:
        with pytest.raises(ToolError, match="user declined"):
            await fetch_url(FetchArgs(url="https://proxied.example/"), ctx)
    assert len(client.requests) == 1


class _ConfirmSequence:
    def __init__(self, answers: list[bool]):
        self._answers = list(answers)

    async def __call__(self, prompt: str) -> bool:
        return self._answers.pop(0)


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
