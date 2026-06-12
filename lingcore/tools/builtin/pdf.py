"""PDF text extraction tool.

``pdf2md`` turns a workspace PDF into markdown-ish text. It is both a
deliberate model choice (extracted text costs far fewer tokens than a native
file part) and the manual counterpart of the automatic PDF modality fallback
(``media_fallback.pdf``) — the only PDF path for a model with no ``file``
modality. Needs the optional ``lingcore[pdf]`` extra (pymupdf; AGPL-3.0,
which is exactly why it is optional).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.media_types import (
    FALLBACK_TEXT_MAX_CHARS,
    FILE_MAX_BYTES,
    media_bytes_match,
)
from lingcore.modality import DEFAULT_PDF_MAX_CHARS, extract_pdf_markdown
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin.fs import _resolve


class Pdf2MdArgs(BaseModel):
    path: str = Field(description="PDF file path relative to the workspace root.")
    max_chars: int | None = Field(
        default=None,
        ge=200,
        le=FALLBACK_TEXT_MAX_CHARS,
        description=(
            f"Optional cap on extracted characters (default {DEFAULT_PDF_MAX_CHARS})."
        ),
    )


@tool(
    description=(
        "Extract a PDF's text as markdown (## Page N headings). Much cheaper "
        "than attaching the whole PDF, and the only PDF option when the model "
        "cannot read file attachments natively."
    )
)
async def pdf2md(args: Pdf2MdArgs, ctx: ToolContext) -> str:
    full = _resolve(ctx, args.path)
    if not full.is_file():
        raise ToolError(f"not a file: {args.path!r}")
    size = full.stat().st_size
    if size > FILE_MAX_BYTES:
        raise ToolError(f"file too large ({size} bytes; limit {FILE_MAX_BYTES})")
    data = full.read_bytes()
    if not media_bytes_match(data, "application/pdf"):
        raise ToolError(f"not a PDF: {args.path!r}")
    opts = ctx.options.get("pdf2md", {})
    raw_max_chars = (
        args.max_chars
        if args.max_chars is not None
        else opts.get("max_chars", DEFAULT_PDF_MAX_CHARS)
    )
    try:
        max_chars = int(raw_max_chars)
    except (TypeError, ValueError):
        raise ToolError("pdf2md.max_chars must be an integer") from None
    if not 200 <= max_chars <= FALLBACK_TEXT_MAX_CHARS:
        raise ToolError(
            f"pdf2md.max_chars must be between 200 and {FALLBACK_TEXT_MAX_CHARS}"
        )
    try:
        return await asyncio.to_thread(extract_pdf_markdown, data, max_chars)
    except ToolError:
        raise
    except Exception as e:
        # A corrupt document is an in-domain failure the model can react to,
        # not an internal error.
        raise ToolError(f"could not extract PDF text: {e}") from None
