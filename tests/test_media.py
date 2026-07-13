from __future__ import annotations

import base64

import pytest

from lingcore.errors import ToolError
from lingcore.media import (
    IMAGE_MAX_BYTES,
    attachment_from_path,
    attachment_from_wire,
    classify_bytes,
    supported_media_type,
)


def test_supported_media_type_maps_jpg_to_jpeg():
    assert supported_media_type("x.jpg") == ("image", "image/jpeg")
    assert supported_media_type("x.jpeg") == ("image", "image/jpeg")
    assert supported_media_type("x.pdf") == ("file", "application/pdf")


def test_attachment_from_path_success(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    att = attachment_from_path(p)
    assert att.kind == "image"
    assert att.media_type == "image/png"
    assert att.name == "pic.png"
    assert base64.b64decode(att.data) == p.read_bytes()


def test_attachment_from_path_classifies_unknown_extension_as_text(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"plain data")
    att = attachment_from_path(p)
    assert att.kind == "text"
    assert att.media_type == "text/plain"
    assert base64.b64decode(att.data) == b"plain data"


def test_attachment_from_path_classifies_ext_content_mismatch_as_binary(tmp_path):
    # .jpg extension but PNG magic: the two disagree, so it is not a native
    # image; the \x89 byte also makes it non-UTF-8, so it lands as binary.
    p = tmp_path / "pic.jpg"
    p.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    att = attachment_from_path(p)
    assert att.kind == "binary"
    assert att.media_type == "application/octet-stream"


def test_attachment_from_path_rejects_large_file_before_read(tmp_path, monkeypatch):
    p = tmp_path / "pic.png"
    with p.open("wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.truncate(IMAGE_MAX_BYTES + 1)

    def boom(self):
        raise AssertionError("read_bytes should not run for oversized media")

    monkeypatch.setattr(type(p), "read_bytes", boom)
    with pytest.raises(ToolError, match="file too large"):
        attachment_from_path(p)


def test_attachment_from_wire_validates_base64_and_size():
    data = base64.b64encode(b"%PDF-1.4\n").decode("ascii")
    raw = {"name": "../x.pdf", "media_type": "application/pdf", "data": data}
    att = attachment_from_wire(raw)
    assert att.name == "x.pdf"
    assert att.kind == "file"
    assert att.data == data

    with pytest.raises(ToolError, match="invalid attachment base64"):
        attachment_from_wire({"media_type": "image/png", "data": "not base64!!"})


def test_attachment_from_wire_text_type():
    data = base64.b64encode(b"hello world").decode("ascii")
    att = attachment_from_wire(
        {"name": "n.txt", "media_type": "text/plain", "data": data}
    )
    assert att.kind == "text"
    assert base64.b64decode(att.data) == b"hello world"


def test_attachment_from_wire_classifies_by_content_not_claim():
    # A non-native media type claim is ignored; the decoded bytes decide.
    data = base64.b64encode(b"\x00\x01\x02ML").decode("ascii")
    att = attachment_from_wire(
        {"name": "x.dat", "media_type": "application/zip", "data": data}
    )
    assert att.kind == "binary"


# --------------------------------------------------------------------------- #
# classify_bytes                                                               #
# --------------------------------------------------------------------------- #


def test_classify_native_image():
    assert classify_bytes(b"\x89PNG\r\n\x1a\nrest", "x.png") == ("image", "image/png")


def test_classify_utf8_text():
    kind, media_type = classify_bytes(b"print('hi')\n", "a.py")
    assert kind == "text"
    assert media_type.startswith("text/")  # mimetypes maps .py to a text/* type
    assert classify_bytes(b"{}", "a.json") == ("text", "application/json")


def test_classify_text_named_like_an_image_stays_text_plain():
    # A text file misnamed notes.png must not claim image/png.
    assert classify_bytes(b"hello", "notes.png") == ("text", "text/plain")


def test_classify_empty_file_is_text():
    assert classify_bytes(b"", "x") == ("text", "text/plain")


def test_classify_nul_bytes_are_binary():
    assert classify_bytes(b"ab\x00cd", "x.dat") == ("binary", "application/octet-stream")


def test_classify_non_utf8_is_binary():
    assert classify_bytes(b"\xff\xfe\x01\x02", "x")[0] == "binary"


def test_text_attachment_rejects_invalid_utf8():
    # A payload declared kind="text" but not valid UTF-8 must be rejected at
    # validation, not inlined into the prompt as replacement characters.
    from lingcore.message import Attachment

    bad = base64.b64encode(b"\xff\xfe\x01\x02").decode("ascii")
    with pytest.raises(ValueError, match="UTF-8"):
        Attachment(kind="text", media_type="text/plain", data=bad, name="x.txt")
