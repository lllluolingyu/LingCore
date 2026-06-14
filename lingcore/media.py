"""Media attachment helpers for multimodal model input."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from pydantic import ValidationError

from lingcore.errors import ToolError
from lingcore.media_types import (
    FILE_MAX_BYTES,
    IMAGE_MAX_BYTES,
    MAX_ATTACHMENT_NAME_CHARS,
    AttachmentKind,
    decode_base64_payload,
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
# application/* types that are really UTF-8 text — accepted as a ``text`` kind
# so their content inlines rather than being treated as opaque binary.
_TEXTUAL_APPLICATION_TYPES = frozenset({
    "application/json",
    "application/xml",
    "application/javascript",
    "application/ecmascript",
    "application/yaml",
    "application/x-yaml",
    "application/toml",
    "application/x-sh",
    "application/x-shellscript",
})
__all__ = [
    "FILE_MAX_BYTES",
    "IMAGE_MAX_BYTES",
    "MAX_ATTACHMENT_NAME_CHARS",
    "attachment_from_bytes",
    "attachment_from_path",
    "attachment_from_wire",
    "classify_bytes",
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


def classify_bytes(
    data: bytes, path_or_name: str | Path = ""
) -> tuple[AttachmentKind, str]:
    """Classify arbitrary bytes into an attachment ``(kind, media_type)``.

    The ladder, most-specific first:

    1. A native image/PDF (extension *and* magic bytes agree) -> that kind.
    2. NUL-free and UTF-8-decodable -> ``text``; the media type is a
       ``mimetypes`` guess, accepted only when it is itself textual (so a text
       file misnamed ``notes.png`` can't claim ``image/png``), else
       ``text/plain``.
    3. Anything else -> ``binary`` (``application/octet-stream``).

    Never raises and never returns ``None`` — every byte string classifies,
    which is what lets *any* file attach.
    """
    detected = detect_media(data, path_or_name)
    if detected is not None:
        return detected
    if b"\x00" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            guess, _ = mimetypes.guess_type(Path(path_or_name).name or "f")
            if guess and (
                guess.startswith("text/") or guess in _TEXTUAL_APPLICATION_TYPES
            ):
                return "text", guess
            return "text", "text/plain"
    return "binary", "application/octet-stream"


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
    if kind is None or media_type is None:
        kind, media_type = classify_bytes(data, name)
    if kind in ("image", "file"):
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
    # Pre-read size gate: a head-detected image gets the 5 MB image cap;
    # everything else (PDF, text, binary) shares FILE_MAX_BYTES. A caller
    # override always wins. Any file is accepted now — the kind is decided
    # after the full read, in attachment_from_bytes.
    limit = max_bytes
    if limit is None:
        limit = max_bytes_for(detected[0]) if detected is not None else FILE_MAX_BYTES
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
    try:
        if matching:
            kind = matching[0][0]
            normalized, _ = _validate_base64_payload(
                data, media_type=media_type, max_bytes=max_bytes_for(kind)
            )
            return Attachment(
                kind=kind, media_type=media_type, data=normalized, name=name
            )
        # Non-native (or unspecified) type: decode and let the content decide
        # the kind — the client's media_type claim never overrides the bytes.
        _, decoded = decode_base64_payload(data, max_bytes=FILE_MAX_BYTES)
        return attachment_from_bytes(decoded, name=name)
    except (ValueError, ValidationError) as e:
        raise _tool_error(e) from None
