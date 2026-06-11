from __future__ import annotations

import base64

import pytest

from lingcore.errors import ToolError
from lingcore.media import attachment_from_path, attachment_from_wire, supported_media_type


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


def test_attachment_from_wire_validates_base64_and_size():
    raw = {"name": "../x.pdf", "media_type": "application/pdf", "data": "cGRm"}
    att = attachment_from_wire(raw)
    assert att.name == "x.pdf"
    assert att.kind == "file"
    assert att.data == "cGRm"

    with pytest.raises(ToolError, match="invalid attachment base64"):
        attachment_from_wire({"media_type": "image/png", "data": "not base64!!"})
