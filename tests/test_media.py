from __future__ import annotations

import base64

import pytest

from lingcore.errors import ToolError
from lingcore.media import (
    IMAGE_MAX_BYTES,
    attachment_from_path,
    attachment_from_wire,
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


def test_attachment_from_path_rejects_unknown_extension(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"data")
    with pytest.raises(ToolError, match="unsupported media type"):
        attachment_from_path(p)


def test_attachment_from_path_rejects_extension_content_mismatch(tmp_path):
    p = tmp_path / "pic.jpg"
    p.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    with pytest.raises(ToolError, match="unsupported media type"):
        attachment_from_path(p)


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
