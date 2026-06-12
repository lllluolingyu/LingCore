"""Modality fallbacks — degrade attachments a model can't receive natively.

A profile *registers* its model's modalities in config (``llm.modalities``);
this module supplies the text stand-ins for everything else. ``MediaAdapter``
computes ``Attachment.fallback_text`` once, when an attachment enters the
conversation: PDFs become extracted text (via the optional ``lingcore[pdf]``
extra — pymupdf), images become a description produced by a secondary vision
model (``media_fallback.image``), and anything unconvertible becomes a short
diagnostic note. The original payload is never stripped, so sessions keep
full fidelity and a later modality upgrade restores native delivery; the
actual native-vs-text decision happens per render in ``Message.to_openai``.

This module never imports ``llm.py`` or the OpenAI SDK: the vision client is
duck-typed (``stream``-shaped, like the agent loop's own seam), and
``Agent.from_profile`` builds the real one.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any, AsyncIterator, Protocol

from lingcore.errors import ToolError
from lingcore.media_types import FALLBACK_TEXT_MAX_CHARS
from lingcore.message import Attachment, Message

PDF_INSTALL_HINT = (
    "PDF text extraction needs the optional dependency pymupdf — "
    "install it with: pip install 'lingcore[pdf]'"
)

DEFAULT_PDF_MAX_CHARS = 262_144

# pymupdf is not thread-safe. Extraction runs in worker threads
# (``asyncio.to_thread``), so a plain threading.Lock serializes every
# extraction in the process — across agents and event loops alike.
_PYMUPDF_LOCK = threading.Lock()


def _import_pymupdf() -> Any:
    """Import pymupdf lazily; the base install deliberately ships without it."""
    try:
        import pymupdf
    except ImportError:
        raise ToolError(PDF_INSTALL_HINT) from None
    return pymupdf


def _summarize(exc: BaseException) -> str:
    """Compact one-line cause description for fallback notes."""
    msg = " ".join(str(exc).split())
    if len(msg) > 200:
        msg = msg[:200] + "…"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _cap_fallback_text(text: str) -> str:
    if len(text) <= FALLBACK_TEXT_MAX_CHARS:
        return text
    marker = "\n[fallback text truncated]"
    return text[: FALLBACK_TEXT_MAX_CHARS - len(marker)] + marker


def extract_pdf_markdown(data: bytes, max_chars: int = DEFAULT_PDF_MAX_CHARS) -> str:
    """Extract a PDF's text as markdown-ish pages (``## Page N`` headings).

    Synchronous and CPU/IO-bound — call via ``asyncio.to_thread``. Raises
    ``ToolError`` for a missing pymupdf or a password-protected document, and
    whatever pymupdf raises for a broken one; callers own containment.
    """
    pymupdf = _import_pymupdf()
    chunks: list[str] = []
    used = 0
    truncated = False
    any_text = False
    with _PYMUPDF_LOCK:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ToolError("PDF is password-protected; cannot extract text")
            total_pages = doc.page_count
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                any_text = any_text or bool(text)
                block = f"## Page {index}\n\n{text or '(no extractable text)'}"
                if used + len(block) > max_chars:
                    remaining = max_chars - used
                    if remaining > 0:
                        chunks.append(block[:remaining])
                    truncated = True
                    break
                chunks.append(block)
                used += len(block) + 2  # the joining blank line
    if not any_text:
        return (
            "(no extractable text — this PDF may be scanned images "
            "without a text layer)"
        )
    out = "\n\n".join(chunks)
    if truncated:
        out += f"\n\n[truncated at {max_chars} characters; {total_pages} pages total]"
    return out


class _VisionClient(Protocol):
    """The duck-typed seam for the fallback vision model (LLMClient-shaped)."""

    def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Any]: ...


class MediaAdapter:
    """Computes text stand-ins for attachment kinds outside ``native``.

    ``prepare`` is the single entry point and **never raises** — a conversion
    failure becomes a diagnostic note in ``fallback_text`` so the turn
    proceeds (the loop's invariant 5 extends here); only cancellation
    propagates. Attachments that are natively supported, or already carry a
    ``fallback_text``, pass through untouched, so ``prepare`` is idempotent
    and a conversion is paid at most once per attachment.
    """

    def __init__(
        self,
        native: frozenset[str],
        *,
        pdf_mode: str = "markdown",
        pdf_max_chars: int = DEFAULT_PDF_MAX_CHARS,
        vision: _VisionClient | None = None,
        vision_prompt: str = (
            "Describe this image in detail for a text-only assistant. "
            "Transcribe any visible text verbatim."
        ),
        vision_max_chars: int = 4_000,
    ) -> None:
        self.native = frozenset(native)
        self.pdf_mode = pdf_mode
        self.pdf_max_chars = pdf_max_chars
        self.vision = vision
        self.vision_prompt = vision_prompt
        self.vision_max_chars = vision_max_chars

    async def prepare(self, attachments: list[Attachment]) -> list[Attachment]:
        """Return attachments ready for this model; copies carry the fallback."""
        pending = [
            (i, a)
            for i, a in enumerate(attachments)
            if a.kind not in self.native and a.fallback_text is None
        ]
        if not pending:
            return attachments
        texts = await asyncio.gather(*(self._fallback_for(a) for _, a in pending))
        out = list(attachments)
        for (i, attachment), text in zip(pending, texts):
            # model_copy skips validators; payload data is unchanged, and this
            # cap keeps the derived fallback under the Attachment ceiling.
            out[i] = attachment.model_copy(
                update={"fallback_text": _cap_fallback_text(text)}
            )
        return out

    async def _fallback_for(self, attachment: Attachment) -> str:
        label = attachment.name or attachment.media_type
        try:
            if attachment.kind == "file":
                return await self._pdf_text(attachment)
            return await self._describe_image(attachment)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            what = "PDF text" if attachment.kind == "file" else "image description"
            return f"[{what} unavailable for {label!r}: {_summarize(e)}]"

    async def _pdf_text(self, attachment: Attachment) -> str:
        if self.pdf_mode != "markdown":
            return (
                "[PDF attached; this model cannot read PDFs natively and the "
                "PDF text fallback is disabled (media_fallback.pdf: none)]"
            )
        data = base64.b64decode(attachment.data)
        return await asyncio.to_thread(
            extract_pdf_markdown, data, self.pdf_max_chars
        )

    async def _describe_image(self, attachment: Attachment) -> str:
        if self.vision is None:
            return (
                "[image attached; this model cannot view images and no "
                "media_fallback.image vision model is configured]"
            )
        request = Message.user(self.vision_prompt, attachments=[attachment])
        parts: list[str] = []
        async for chunk in self.vision.stream([request]):
            if getattr(chunk, "text_delta", ""):
                parts.append(chunk.text_delta)
        description = "".join(parts).strip()
        if not description:
            return "[image description unavailable: the vision model returned no text]"
        if len(description) > self.vision_max_chars:
            description = description[: self.vision_max_chars] + "…"
        return f"(described by a fallback vision model)\n{description}"
