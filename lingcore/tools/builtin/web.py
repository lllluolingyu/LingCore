"""fetch_url — retrieve a URL and return its text content.

HTML is stripped to readable text via a lightweight regex pass so the model
receives prose rather than tag soup. JSON, plain text, and Markdown are
returned as-is. Output is truncated to _MAX_CHARS to stay within context.
"""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool

_MAX_CHARS = 32_000
_TIMEOUT = 20.0


def _html_to_text(html: str) -> str:
    """Very lightweight HTML → plain text: strip tags, collapse whitespace."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class FetchArgs(BaseModel):
    url: str = Field(description="The URL to fetch (http or https).")


@tool(description="Fetch a URL and return its text content (HTML is converted to plain text).")
async def fetch_url(args: FetchArgs, ctx: ToolContext) -> str:
    if not args.url.startswith(("http://", "https://")):
        raise ToolError(f"only http/https URLs are supported: {args.url!r}")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT) as client:
            r = await client.get(args.url, headers={"User-Agent": "LingCore/0.0.1"})
    except httpx.HTTPError as e:
        raise ToolError(f"request failed: {e}") from None

    content_type = r.headers.get("content-type", "")
    text = r.text
    if "html" in content_type:
        text = _html_to_text(text)

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n... (truncated, {len(text) - _MAX_CHARS} more chars)"

    return f"[{r.status_code} {args.url}]\n\n{text}"
