"""Media attachment helpers for multimodal model input."""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import ValidationError

from lingcore.errors import ToolError
from lingcore.media_types import (
    FILE_MAX_BYTES,
    IMAGE_MAX_BYTES,
    MAX_ATTACHMENT_NAME_CHARS,
    AttachmentKind,
    detect_media,
    max_bytes_for,
    media_bytes_match,
    sanitize_name,
    supported_extensions,
    supported_media_type,
    supported_media_types,
    validate_base64_payload as _validate_base64_payload,
)
from lingcore.message import Attachment

_HEAD_BYTES = 16
__all__ = [
    "FILE_MAX_BYTES",
    "IMAGE_MAX_BYTES",
    "MAX_ATTACHMENT_NAME_CHARS",
    "attachment_from_bytes",
    "attachment_from_path",
    "attachment_from_wire",
    "detect_media",
    "is_probably_binary",
    "max_bytes_for",
    "sanitize_name",
    "supported_extensions",
    "supported_media_type",
    "validate_base64_payload",
]


def is_probably_binary(data: bytes) -> bool:
    """Cheap binary guard for text-oriented tools."""
    return b"\x00" in data


def _tool_error(exc: ValueError | ValidationError) -> ToolError:
    if isinstance(exc, ValidationError):
        msg = exc.errors()[0].get("msg", str(exc))
        msg = str(msg).removeprefix("Value error, ")
    else:
        msg = str(exc)
    return ToolError(msg)


def validate_base64_payload(
    data: str, *, max_bytes: int, media_type: str
) -> tuple[str, int]:
    """Validate and normalize a base64 payload, returning normalized data + size.

    ``media_type`` is required on purpose: every payload entering the system
    is checked against its declared type's magic bytes — there is no
    content-unchecked path.
    """
    try:
        return _validate_base64_payload(
            data, media_type=media_type, max_bytes=max_bytes
        )
    except ValueError as e:
        raise _tool_error(e) from None


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
    if (kind, media_type) not in supported_media_types():
        raise ToolError(f"unsupported media type: {media_type}")
    if not media_bytes_match(data, media_type):
        raise ToolError(f"file content does not match media type: {media_type}")
    limit = max_bytes if max_bytes is not None else max_bytes_for(kind)
    if len(data) > limit:
        raise ToolError(f"file too large ({len(data)} bytes; limit {limit})")
    try:
        return Attachment(
            kind=kind,
            media_type=media_type,
            data=base64.b64encode(data).decode("ascii"),
            name=sanitize_name(name),
        )
    except ValidationError as e:
        raise _tool_error(e) from None


def attachment_from_path(path: Path, *, max_bytes: int | None = None) -> Attachment:
    if not path.is_file():
        raise ToolError(f"not a file: {str(path)!r}")
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(_HEAD_BYTES)
    detected = detect_media(head, path)
    if detected is None:
        raise ToolError(f"unsupported media type: {path.name!r}")
    limit = max_bytes
    if limit is None:
        limit = max_bytes_for(detected[0])
    if size > limit:
        raise ToolError(f"file too large ({size} bytes; limit {limit})")
    data = path.read_bytes()
    return attachment_from_bytes(data, name=path.name, max_bytes=max_bytes)


def attachment_from_wire(raw: object) -> Attachment:
    """Validate an attachment object received from a frontend."""
    if not isinstance(raw, dict):
        raise ToolError("attachment must be an object")
    name = sanitize_name(str(raw.get("name") or "attachment"))
    media_type = str(raw.get("media_type") or raw.get("mime_type") or "")
    data = str(raw.get("data") or "")
    matching = [item for item in supported_media_types() if item[1] == media_type]
    if not matching:
        raise ToolError(f"unsupported media type: {media_type!r}")
    kind = matching[0][0]
    try:
        normalized, _ = _validate_base64_payload(
            data, media_type=media_type, max_bytes=max_bytes_for(kind)
        )
        return Attachment(
            kind=kind, media_type=media_type, data=normalized, name=name
        )
    except (ValueError, ValidationError) as e:
        raise _tool_error(e) from None
