"""Tests for workspace attachment ingest (copy + text/binary fallbacks)."""

from __future__ import annotations

import base64
from pathlib import Path

from lingcore.ingest import ingest_attachments
from lingcore.media_types import TEXT_INLINE_MAX_CHARS
from lingcore.message import Attachment
from lingcore.paths import ConfinedDirectory


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
    def boom(self, name, mode=0o644):
        raise OSError("disk full")

    monkeypatch.setattr(ConfinedDirectory, "open_exclusive", boom)
    out, notes = ingest_attachments([_text(b"hello", name="a.txt")], tmp_path)
    assert any("could not be saved" in n for n in notes)
    # Text inlining decodes the in-memory payload, so it survives a copy failure.
    assert out[0].fallback_text == "hello"


def test_empty_list_returns_unchanged(tmp_path):
    out, notes = ingest_attachments([], tmp_path)
    assert out == [] and notes == []


def test_symlinked_attachments_dir_does_not_escape(tmp_path):
    # A pre-existing `attachments` symlink pointing outside the workspace must
    # not redirect the write there; ingest confines it and notes the failure.
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "attachments").symlink_to(outside, target_is_directory=True)

    out, notes = ingest_attachments([_image()], workspace)

    # Nothing was written into the escape target.
    assert list(outside.iterdir()) == []
    assert any("escapes workspace" in n for n in notes)
    # Text/binary fallbacks still compute; the attachment object survives.
    assert len(out) == 1


def test_dangling_gitignore_symlink_does_not_escape(tmp_path):
    # A dangling `.gitignore` symlink passes an exists() check as absent; the
    # write must refuse to follow it instead of creating its outside target.
    workspace = tmp_path / "ws"
    (workspace / "attachments").mkdir(parents=True)
    escape_target = tmp_path / "outside-gitignore"
    (workspace / "attachments" / ".gitignore").symlink_to(escape_target)

    ingest_attachments([_image()], workspace)

    assert not escape_target.exists()  # never created through the link
    # The attachment itself still lands normally.
    assert (workspace / "attachments" / "p.png").is_file()


def test_dangling_attachment_symlink_does_not_escape(tmp_path):
    # A dangling symlink planted at the attachment's own filename must not
    # redirect the payload to its outside target.
    workspace = tmp_path / "ws"
    (workspace / "attachments").mkdir(parents=True)
    escape_target = tmp_path / "outside-payload"
    (workspace / "attachments" / "p.png").symlink_to(escape_target)

    out, notes = ingest_attachments([_image()], workspace)

    assert not escape_target.exists()
    # The planted name is treated as occupied; collision handling safely chooses
    # another regular-file entry under the held attachments directory.
    assert (workspace / "attachments" / "p-2.png").is_file()
    assert any("attachments/p-2.png" in n for n in notes)


def test_inside_workspace_dangling_symlink_is_not_written_through(tmp_path):
    # A dangling symlink whose target is *inside* the workspace passes the
    # confinement check (and exists() reports the name as free) — the exclusive
    # create must still refuse to follow it, or the payload would materialize
    # at the link's target instead of under attachments/.
    workspace = tmp_path / "ws"
    (workspace / "attachments").mkdir(parents=True)
    target = workspace / "planted.txt"  # does not exist
    (workspace / "attachments" / "a.txt").symlink_to(target)

    out, notes = ingest_attachments([_text(b"payload")], workspace)

    assert not target.exists()  # never created through the link
    assert (workspace / "attachments" / "a-2.txt").read_bytes() == b"payload"
    assert any("attachments/a-2.txt" in n for n in notes)


def test_parent_directory_swap_cannot_redirect_payload(tmp_path, monkeypatch):
    """Replacing the validated parent just before create must fail closed."""
    workspace = tmp_path / "ws"
    attachments = workspace / "attachments"
    attachments.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    held = workspace / "attachments-held"

    real_open = ConfinedDirectory.open_exclusive
    swapped = False

    def swap_then_open(self, name, mode=0o644):
        nonlocal swapped
        if name == "p.png" and not swapped:
            attachments.rename(held)
            attachments.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(self, name, mode)

    monkeypatch.setattr(ConfinedDirectory, "open_exclusive", swap_then_open)

    _, notes = ingest_attachments([_image()], workspace)

    assert not (outside / "p.png").exists()
    assert not (held / "p.png").exists()
    assert any("confined directory changed" in n for n in notes)
