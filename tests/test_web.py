"""Tests for fetch_url."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lingcore.errors import ToolError
from lingcore.tools import ToolContext
from lingcore.tools.builtin.web import FetchArgs, fetch_url


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


def _mock_response(text: str, content_type: str = "text/plain", status_code: int = 200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.headers = {"content-type": content_type}
    return r


async def test_fetch_plain(ctx):
    resp = _mock_response("hello world")
    with patch("lingcore.tools.builtin.web.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=resp)))
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        out = await fetch_url(FetchArgs(url="https://example.com/api"), ctx)
    assert "hello world" in out
    assert "200" in out


async def test_fetch_html_stripped(ctx):
    html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"
    resp = _mock_response(html, content_type="text/html")
    with patch("lingcore.tools.builtin.web.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=resp)))
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        out = await fetch_url(FetchArgs(url="https://example.com/"), ctx)
    assert "<html>" not in out
    assert "Title" in out
    assert "Content here" in out


async def test_fetch_truncates(ctx):
    resp = _mock_response("x" * 40_000)
    with patch("lingcore.tools.builtin.web.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=resp)))
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        out = await fetch_url(FetchArgs(url="https://example.com/big"), ctx)
    assert "truncated" in out


async def test_fetch_rejects_non_http(ctx):
    with pytest.raises(ToolError, match="http"):
        await fetch_url(FetchArgs(url="ftp://example.com/file"), ctx)


async def test_fetch_http_error(ctx):
    import httpx
    with patch("lingcore.tools.builtin.web.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(ToolError, match="request failed"):
            await fetch_url(FetchArgs(url="https://example.com/"), ctx)
