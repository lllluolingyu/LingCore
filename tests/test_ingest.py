"""Tests for workspace attachment ingest (copy + text/binary fallbacks)."""

from __future__ import annotations

import base64
from pathlib import Path

from lingcore.ingest import ingest_attachments
from lingcore.media_types import TEXT_INLINE_MAX_CHARS
from lingcore.message import Attachment


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _text(content: bytes, name: str = "a.txt") -> Attachment:
    return Attachment(
        kind="text", media_type="text/plain", data=_b64(content), name=name
    )


def _image(name: str = "p.png", body: bytes = b"rest") -> Attachment:
    return Attachment(
        kind="image",
        media_type="image/png",
        data=_b64(b"\x89PNG\r\n\x1a\n" + body),
        name=name,
    )


def test_copies_into_attachments_dir_and_notes_path(tmp_path):
    out, notes = ingest_attachments([_image()], tmp_path)
    saved = tmp_path / "attachments" / "p.png"
    assert saved.is_file()
    assert saved.read_bytes() == b"\x89PNG\r\n\x1a\nrest"
    assert any("attachments/p.png" in n for n in notes)
    assert any("image/png" in n for n in notes)


def test_image_passes_through_unchanged(tmp_path):
    att = _image()
    out, _ = ingest_attachments([att], tmp_path)
    # Image fallback is MediaAdapter's job, not ingest's: identity preserved.
    assert out[0] is att
    assert out[0].fallback_text is None


def test_text_content_is_inlined_as_fallback(tmp_path):
    att = _text(b"line one\nline two\n")
    out, _ = ingest_attachments([att], tmp_path)
    assert out[0].fallback_text == "line one\nline two\n"
    assert (tmp_path / "attachments" / "a.txt").is_file()


def test_long_text_is_truncated_with_path_marker(tmp_path):
    big = b"x" * (TEXT_INLINE_MAX_CHARS + 500)
    out, _ = ingest_attachments([_text(big, name="big.txt")], tmp_path)
    fallback = out[0].fallback_text
    assert len(fallback) <= TEXT_INLINE_MAX_CHARS
    assert "attachments/big.txt" in fallback
    assert "truncated" in fallback


def test_binary_gets_workspace_pointer_note(tmp_path):
    att = Attachment(
        kind="binary",
        media_type="application/octet-stream",
        data=_b64(b"\x00\x01\x02\x03"),
        name="x.bin",
    )
    out, _ = ingest_attachments([att], tmp_path)
    assert "attachments/x.bin" in out[0].fallback_text
    assert "workspace tools" in out[0].fallback_text


def test_same_name_same_bytes_is_reused(tmp_path):
    ingest_attachments([_image()], tmp_path)
    ingest_attachments([_image()], tmp_path)
    files = sorted(p.name for p in (tmp_path / "attachments").glob("p*.png"))
    assert files == ["p.png"]  # byte-identical: reused, no -2 copy


def test_same_name_different_bytes_is_suffixed(tmp_path):
    ingest_attachments([_image(body=b"one")], tmp_path)
    out, notes = ingest_attachments([_image(body=b"two")], tmp_path)
    assert (tmp_path / "attachments" / "p-2.png").is_file()
    assert any("p-2.png" in n for n in notes)


def test_writes_gitignore_to_shield_user_repo(tmp_path):
    ingest_attachments([_image()], tmp_path)
    gi = tmp_path / "attachments" / ".gitignore"
    assert gi.is_file() and gi.read_text().strip() == "*"


def test_copy_failure_degrades_to_note_and_still_inlines_text(tmp_path, monkeypatch):
    # A failed disk write must not raise; ingest notes it and the turn proceeds.
    def boom(self, data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", boom)
    out, notes = ingest_attachments([_text(b"hello", name="a.txt")], tmp_path)
    assert any("could not be saved" in n for n in notes)
    # Text inlining decodes the in-memory payload, so it survives a copy failure.
    assert out[0].fallback_text == "hello"


def test_empty_list_returns_unchanged(tmp_path):
    out, notes = ingest_attachments([], tmp_path)
    assert out == [] and notes == []
