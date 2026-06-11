"""Media attachment helpers for multimodal model input."""

from __future__ import annotations

import base64
from pathlib import Path

from lingcore.errors import ToolError
from lingcore.message import Attachment, AttachmentKind

IMAGE_MAX_BYTES = 5 * 1024 * 1024
FILE_MAX_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENT_NAME_CHARS = 120

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
    (b"RIFF", "image", "image/webp"),
    (b"%PDF-", "file", "application/pdf"),
)


def supported_media_type(path_or_name: str | Path) -> tuple[AttachmentKind, str] | None:
    """Return the supported attachment kind/media type for a path extension."""
    suffix = Path(path_or_name).suffix.lower()
    return _EXTENSIONS.get(suffix)


def supported_extensions() -> set[str]:
    return set(_EXTENSIONS)


def detect_media(data: bytes, path_or_name: str | Path = "") -> tuple[AttachmentKind, str] | None:
    """Detect supported media by extension first, then simple magic bytes."""
    by_ext = supported_media_type(path_or_name)
    if by_ext is not None:
        return by_ext
    for prefix, kind, media_type in _MAGIC:
        if data.startswith(prefix):
            if media_type == "image/webp" and len(data) >= 12 and data[8:12] != b"WEBP":
                continue
            return kind, media_type
    return None


def is_probably_binary(data: bytes) -> bool:
    """Cheap binary guard for text-oriented tools."""
    return b"\x00" in data


def max_bytes_for(kind: AttachmentKind) -> int:
    return IMAGE_MAX_BYTES if kind == "image" else FILE_MAX_BYTES


def sanitize_name(name: str | None, fallback: str = "attachment") -> str:
    """Return a short display filename without path separators."""
    raw = (name or "").replace("\\", "/").split("/")[-1].strip()
    raw = raw or fallback
    raw = "".join(ch if ch.isprintable() and ch not in "\r\n\t" else "_" for ch in raw)
    raw = raw[:MAX_ATTACHMENT_NAME_CHARS].strip(" .")
    return raw or fallback


def validate_base64_payload(data: str, *, max_bytes: int) -> tuple[str, int]:
    """Validate and normalize a base64 payload, returning normalized data + size."""
    raw = data.strip()
    if raw.startswith("data:"):
        try:
            raw = raw.split(",", 1)[1]
        except IndexError:
            raise ToolError("invalid attachment data URI") from None
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception:
        raise ToolError("invalid attachment base64 data") from None
    size = len(decoded)
    if size > max_bytes:
        raise ToolError(f"attachment too large ({size} bytes; limit {max_bytes})")
    return base64.b64encode(decoded).decode("ascii"), size


def attachment_from_bytes(
    data: bytes,
    *,
    name: str,
    media_type: str | None = None,
    kind: AttachmentKind | None = None,
    max_bytes: int | None = None,
) -> Attachment:
    detected = detect_media(data, name)
    if media_type is None or kind is None:
        if detected is None:
            raise ToolError(f"unsupported media type: {name!r}")
        detected_kind, detected_media_type = detected
        kind = kind or detected_kind
        media_type = media_type or detected_media_type
    if (kind, media_type) not in set(_EXTENSIONS.values()):
        raise ToolError(f"unsupported media type: {media_type}")
    limit = max_bytes if max_bytes is not None else max_bytes_for(kind)
    if len(data) > limit:
        raise ToolError(f"file too large ({len(data)} bytes; limit {limit})")
    return Attachment(
        kind=kind,
        media_type=media_type,
        data=base64.b64encode(data).decode("ascii"),
        name=sanitize_name(name),
    )


def attachment_from_path(path: Path, *, max_bytes: int | None = None) -> Attachment:
    if not path.is_file():
        raise ToolError(f"not a file: {str(path)!r}")
    data = path.read_bytes()
    return attachment_from_bytes(data, name=path.name, max_bytes=max_bytes)


def attachment_from_wire(raw: object) -> Attachment:
    """Validate an attachment object received from a frontend."""
    if not isinstance(raw, dict):
        raise ToolError("attachment must be an object")
    name = sanitize_name(str(raw.get("name") or "attachment"))
    media_type = str(raw.get("media_type") or raw.get("mime_type") or "")
    data = str(raw.get("data") or "")
    matching = [item for item in _EXTENSIONS.values() if item[1] == media_type]
    if not matching:
        raise ToolError(f"unsupported media type: {media_type!r}")
    kind = matching[0][0]
    normalized, _ = validate_base64_payload(data, max_bytes=max_bytes_for(kind))
    return Attachment(kind=kind, media_type=media_type, data=normalized, name=name)
