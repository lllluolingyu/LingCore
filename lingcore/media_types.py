"""Shared media attachment validation rules.

This module deliberately avoids importing the rest of LingCore so the canonical
message model can validate attachments at construction time without creating
dependency cycles.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

AttachmentKind = Literal["image", "file"]

IMAGE_MAX_BYTES = 5 * 1024 * 1024
FILE_MAX_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENT_NAME_CHARS = 120
MAX_ATTACHMENTS = 8
TOTAL_ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024

_EXTENSIONS: dict[str, tuple[AttachmentKind, str]] = {
    ".png": ("image", "image/png"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".gif": ("image", "image/gif"),
    ".webp": ("image", "image/webp"),
    ".pdf": ("file", "application/pdf"),
}

_MAGIC: tuple[tuple[bytes, AttachmentKind, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image", "image/png"),
    (b"\xff\xd8\xff", "image", "image/jpeg"),
    (b"GIF87a", "image", "image/gif"),
    (b"GIF89a", "image", "image/gif"),
    (b"%PDF-", "file", "application/pdf"),
)

_SUPPORTED_TYPES = frozenset(_EXTENSIONS.values())


def supported_media_type(path_or_name: str | Path) -> tuple[AttachmentKind, str] | None:
    """Return the supported attachment kind/media type for a path extension."""
    suffix = Path(path_or_name).suffix.lower()
    return _EXTENSIONS.get(suffix)


def supported_extensions() -> set[str]:
    return set(_EXTENSIONS)


def supported_media_types() -> frozenset[tuple[AttachmentKind, str]]:
    return _SUPPORTED_TYPES


def kind_for_media_type(media_type: str) -> AttachmentKind | None:
    for kind, known in _SUPPORTED_TYPES:
        if known == media_type:
            return kind
    return None


def max_bytes_for(kind: AttachmentKind) -> int:
    return IMAGE_MAX_BYTES if kind == "image" else FILE_MAX_BYTES


def sanitize_name(name: str | None, fallback: str = "attachment") -> str:
    """Return a short display filename without path separators."""
    raw = (name or "").replace("\\", "/").split("/")[-1].strip()
    raw = raw or fallback
    raw = "".join(
        ch if ch.isprintable() and ch not in "\r\n\t" else "_" for ch in raw
    )
    raw = raw[:MAX_ATTACHMENT_NAME_CHARS].strip(" .")
    return raw or fallback


def _magic_type(data: bytes) -> tuple[AttachmentKind, str] | None:
    for prefix, kind, media_type in _MAGIC:
        if data.startswith(prefix):
            return kind, media_type
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image", "image/webp"
    return None


def detect_media(
    data: bytes, path_or_name: str | Path = ""
) -> tuple[AttachmentKind, str] | None:
    """Detect supported media, requiring extension and magic bytes to agree."""
    by_ext = supported_media_type(path_or_name)
    by_magic = _magic_type(data)
    if by_ext is not None:
        return by_ext if by_ext == by_magic else None
    return by_magic


def media_bytes_match(data: bytes, media_type: str) -> bool:
    detected = _magic_type(data)
    return detected is not None and detected[1] == media_type


def validate_base64_payload(
    data: str, *, media_type: str, max_bytes: int
) -> tuple[str, int]:
    """Validate and normalize a base64 payload, returning normalized data + size."""
    raw = data.strip()
    if raw.startswith("data:"):
        try:
            header, raw = raw.split(",", 1)
        except ValueError:
            raise ValueError("invalid attachment data URI") from None
        uri_media_type = header[5:].split(";", 1)[0]
        if ";base64" not in header:
            raise ValueError("invalid attachment data URI")
        if uri_media_type and uri_media_type != media_type:
            raise ValueError(
                f"attachment data URI media type {uri_media_type!r} "
                f"does not match {media_type!r}"
            )
    # Reject on encoded length first: base64 decodes 4 chars -> 3 bytes, so an
    # over-cap payload is refused without materializing the decoded bytes.
    if (len(raw) // 4) * 3 - 2 > max_bytes:
        raise ValueError(
            f"attachment too large (>{max_bytes} bytes decoded)"
        )
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception:
        raise ValueError("invalid attachment base64 data") from None
    size = len(decoded)
    if size > max_bytes:
        raise ValueError(f"attachment too large ({size} bytes; limit {max_bytes})")
    if not media_bytes_match(decoded, media_type):
        raise ValueError(f"attachment data does not match media type {media_type!r}")
    return base64.b64encode(decoded).decode("ascii"), size


def decoded_payload_size(data: str) -> int:
    return len(base64.b64decode(data, validate=True))
