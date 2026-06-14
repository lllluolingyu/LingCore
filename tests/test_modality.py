"""Tests for modality fallbacks — MediaAdapter + PDF extraction.

The vision client is doubled by ``FakeLLMClient`` (it has the same duck-typed
``stream`` shape the adapter expects). PDF fixtures are real documents built
with pymupdf itself — a bare ``%PDF-`` magic prefix is not an openable file.
"""

from __future__ import annotations

import base64

import pymupdf
import pytest

import lingcore.modality as modality_mod
from lingcore.errors import ToolError
from lingcore.llm import LLMChunk
from lingcore.media_types import FALLBACK_TEXT_MAX_CHARS
from lingcore.message import Attachment, Message
from lingcore.modality import MediaAdapter, PDF_INSTALL_HINT, extract_pdf_markdown
from tests.fakes import FakeLLMClient, ScriptedTurn


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def make_pdf(*page_texts: str, **save_kwargs) -> bytes:
    """Build a real PDF with one page per text ('' = blank page)."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes(**save_kwargs)
    doc.close()
    return data


def _pdf_attachment(*page_texts: str, name: str = "doc.pdf") -> Attachment:
    return Attachment(
        kind="file",
        media_type="application/pdf",
        data=_b64(make_pdf(*page_texts)),
        name=name,
    )


def _image_attachment(name: str = "pic.png") -> Attachment:
    return Attachment(
        kind="image",
        media_type="image/png",
        data=_b64(b"\x89PNG\r\n\x1a\nrest"),
        name=name,
    )


# --------------------------------------------------------------------------- #
# extract_pdf_markdown                                                         #
# --------------------------------------------------------------------------- #


def test_extract_pages_with_headings():
    out = extract_pdf_markdown(make_pdf("alpha words", "beta words"))
    assert "## Page 1" in out and "alpha words" in out
    assert "## Page 2" in out and "beta words" in out


def test_extract_truncates_at_max_chars():
    out = extract_pdf_markdown(make_pdf("alpha words here", "beta"), max_chars=20)
    assert out.startswith("## Page 1")
    assert "[truncated at 20 characters; 2 pages total]" in out
    assert "beta" not in out


def test_extract_textless_pdf_notes_scanned_images():
    out = extract_pdf_markdown(make_pdf("", ""))
    assert "no extractable text" in out
    assert "scanned images" in out


def test_extract_encrypted_pdf_raises_tool_error():
    data = make_pdf(
        "secret",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="pw",
        user_pw="pw",
    )
    with pytest.raises(ToolError, match="password-protected"):
        extract_pdf_markdown(data)


def test_extract_without_pymupdf_names_the_extra(monkeypatch):
    def boom():
        raise ToolError(PDF_INSTALL_HINT)

    monkeypatch.setattr(modality_mod, "_import_pymupdf", boom)
    with pytest.raises(ToolError, match=r"lingcore\[pdf\]"):
        extract_pdf_markdown(make_pdf("x"))


# --------------------------------------------------------------------------- #
# MediaAdapter.prepare                                                         #
# --------------------------------------------------------------------------- #


async def test_prepare_pdf_gets_extracted_text():
    att = _pdf_attachment("the rent is due Friday")
    out = await MediaAdapter(native=frozenset()).prepare([att])
    assert out[0].fallback_text and "the rent is due Friday" in out[0].fallback_text
    assert out[0].data == att.data  # original payload kept
    assert att.fallback_text is None  # input object never mutated


async def test_prepare_pdf_mode_none_leaves_a_note():
    out = await MediaAdapter(native=frozenset(), pdf_mode="none").prepare(
        [_pdf_attachment("x")]
    )
    assert "fallback is disabled" in out[0].fallback_text


async def test_prepare_missing_pymupdf_becomes_note_not_error(monkeypatch):
    def boom():
        raise ToolError(PDF_INSTALL_HINT)

    monkeypatch.setattr(modality_mod, "_import_pymupdf", boom)
    out = await MediaAdapter(native=frozenset()).prepare([_pdf_attachment("x")])
    assert "PDF text unavailable" in out[0].fallback_text
    assert "lingcore[pdf]" in out[0].fallback_text


async def test_prepare_extraction_failure_becomes_note(monkeypatch):
    def boom(data, max_chars):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(modality_mod, "extract_pdf_markdown", boom)
    out = await MediaAdapter(native=frozenset()).prepare([_pdf_attachment("x")])
    assert "PDF text unavailable" in out[0].fallback_text
    assert "engine exploded" in out[0].fallback_text


async def test_prepare_caps_fallback_text_before_model_copy(monkeypatch):
    def long_text(data, max_chars):
        return "x" * (FALLBACK_TEXT_MAX_CHARS + 10)

    monkeypatch.setattr(modality_mod, "extract_pdf_markdown", long_text)
    out = await MediaAdapter(native=frozenset()).prepare([_pdf_attachment("x")])
    assert len(out[0].fallback_text) == FALLBACK_TEXT_MAX_CHARS
    assert out[0].fallback_text.endswith("[fallback text truncated]")


async def test_prepare_image_without_vision_notes_it():
    out = await MediaAdapter(native=frozenset()).prepare([_image_attachment()])
    assert "cannot view images" in out[0].fallback_text


async def test_prepare_image_described_by_vision_client():
    vision = FakeLLMClient([ScriptedTurn(text="a red square on white")])
    adapter = MediaAdapter(
        native=frozenset(), vision=vision, vision_prompt="What is shown?"
    )
    att = _image_attachment()
    out = await adapter.prepare([att])
    assert "a red square on white" in out[0].fallback_text
    assert "fallback vision model" in out[0].fallback_text
    # The describe request carried the prompt and the image itself, natively.
    request = vision.calls[0][0]
    assert request.role == "user"
    assert request.content == "What is shown?"
    assert request.attachments[0].data == att.data
    assert isinstance(request.to_openai()["content"], list)


async def test_prepare_truncates_long_descriptions():
    vision = FakeLLMClient([ScriptedTurn(text="x" * 100)])
    adapter = MediaAdapter(native=frozenset(), vision=vision, vision_max_chars=10)
    out = await adapter.prepare([_image_attachment()])
    description = out[0].fallback_text.splitlines()[-1]
    assert description == "x" * 10 + "…"


async def test_prepare_empty_description_becomes_note():
    vision = FakeLLMClient([ScriptedTurn(text="")])
    out = await MediaAdapter(native=frozenset(), vision=vision).prepare(
        [_image_attachment()]
    )
    assert "returned no text" in out[0].fallback_text


async def test_prepare_vision_failure_becomes_note_not_error():
    class BoomVision:
        async def stream(self, messages, tools=None):
            yield LLMChunk(text_delta="par")
            raise RuntimeError("vision endpoint down")

    out = await MediaAdapter(native=frozenset(), vision=BoomVision()).prepare(
        [_image_attachment()]
    )
    assert "image description unavailable" in out[0].fallback_text
    assert "vision endpoint down" in out[0].fallback_text


async def test_prepare_skips_native_kinds():
    adapter = MediaAdapter(native=frozenset({"image", "file"}))
    atts = [_image_attachment(), _pdf_attachment("x")]
    assert await adapter.prepare(atts) is atts  # untouched, same list object


async def test_prepare_mixed_native_and_fallback():
    adapter = MediaAdapter(native=frozenset({"image"}))
    img, pdf = _image_attachment(), _pdf_attachment("budget table")
    out = await adapter.prepare([img, pdf])
    assert out[0] is img  # natively supported: passed through
    assert "budget table" in out[1].fallback_text


async def test_prepare_is_idempotent():
    adapter = MediaAdapter(native=frozenset())
    first = await adapter.prepare([_pdf_attachment("once")])
    second = await adapter.prepare(first)
    assert second[0] is first[0]  # already prepared: no second conversion


async def test_prepare_skips_text_and_binary_kinds():
    # The adapter only converts image/file. text/binary carry (or will carry)
    # their own fallbacks from ingest, so prepare leaves them alone even when
    # fallback_text is still None — never routing them through PDF/vision paths.
    txt = Attachment(kind="text", media_type="text/plain", data=_b64(b"hello"))
    binary = Attachment(
        kind="binary", media_type="application/octet-stream", data=_b64(b"\x00\x01")
    )
    out = await MediaAdapter(native=frozenset()).prepare([txt, binary])
    assert out[0] is txt and out[1] is binary
    assert out[0].fallback_text is None and out[1].fallback_text is None
