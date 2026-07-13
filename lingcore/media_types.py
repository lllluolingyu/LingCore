"""Shared media attachment validation rules.

This module deliberately avoids importing the rest of LingCore so the canonical
message model can validate attachments at construction time without creating
dependency cycles.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Literal

# Four attachment kinds. ``image``/``file`` may be delivered as native content
# parts; ``text`` always inlines its decoded content as prompt text; ``binary``
# is a copied-into-the-workspace file the model reaches only through tools.
AttachmentKind = Literal["image", "file", "text", "binary"]
# The kinds a chat-completions request can carry as a *native* content part.
# ``llm.modalities`` is typed to this subset (declaring ``text``/``binary`` as
# native is nonsensical, so config rejects it loudly).
NativeModality = Literal["image", "file"]

IMAGE_MAX_BYTES = 5 * 1024 * 1024
FILE_MAX_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENT_NAME_CHARS = 120
MAX_ATTACHMENTS = 8
TOTAL_ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024
# Hard ceiling on an attachment's text stand-in (``Attachment.fallback_text``).
# Producers cap below this (DEFAULT_PDF_MAX_CHARS = 256 KiB of chars); the
# model validator *truncates* to it rather than rejecting, so a stored row can
# never become unloadable over this field.
FALLBACK_TEXT_MAX_CHARS = 300_000
# Max characters of a text attachment inlined into the prompt at ingest. Sized
# so an inlined file (~chars/4 tokens) fits the default ``memory.max_tokens``
# budget with room for history; raise ``memory.max_tokens`` for heavy text work.
TEXT_INLINE_MAX_CHARS = 32_768

# A conservative ``type/subtype`` shape for the free-form media types text and
# binary attachments carry. Beyond well-formedness it bars whitespace, newlines,
# and brackets, so a media_type interpolated into a one-line fallback header
# can't reshape it.
_MEDIA_TYPE_RE = re.compile(r"[\w.+-]+/[\w.+-]+")


def is_valid_media_type(media_type: str) -> bool:
    return bool(_MEDIA_TYPE_RE.fullmatch(media_type))

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


def decode_base64_payload(
    data: str, *, max_bytes: int, declared_media_type: str | None = None
) -> tuple[str, bytes]:
    """Strip an optional ``data:`` URI, size-check, and decode a base64 payload.

    Returns ``(normalized_base64, decoded_bytes)``. No magic-byte check — that
    is the caller's job (image/file verify it; text/binary don't). The decoded
    bytes are handed back so callers needing them (a NUL scan, a content
    classifier) don't decode twice.
    """
    raw = data.strip()
    if raw.startswith("data:"):
        try:
            header, raw = raw.split(",", 1)
        except ValueError:
            raise ValueError("invalid attachment data URI") from None
        if ";base64" not in header:
            raise ValueError("invalid attachment data URI")
        uri_media_type = header[5:].split(";", 1)[0]
        if (
            declared_media_type
            and uri_media_type
            and uri_media_type != declared_media_type
        ):
            raise ValueError(
                f"attachment data URI media type {uri_media_type!r} "
                f"does not match {declared_media_type!r}"
            )
    # Reject on encoded length first: base64 decodes 4 chars -> 3 bytes, so an
    # over-cap payload is refused without materializing the decoded bytes.
    if (len(raw) // 4) * 3 - 2 > max_bytes:
        raise ValueError(f"attachment too large (>{max_bytes} bytes decoded)")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception:
        raise ValueError("invalid attachment base64 data") from None
    if len(decoded) > max_bytes:
        raise ValueError(f"attachment too large ({len(decoded)} bytes; limit {max_bytes})")
    return base64.b64encode(decoded).decode("ascii"), decoded


def validate_base64_payload(
    data: str, *, media_type: str, max_bytes: int
) -> tuple[str, int]:
    """Validate+normalize an image/file payload, asserting its magic bytes."""
    normalized, decoded = decode_base64_payload(
        data, max_bytes=max_bytes, declared_media_type=media_type
    )
    if not media_bytes_match(decoded, media_type):
        raise ValueError(f"attachment data does not match media type {media_type!r}")
    return normalized, len(decoded)


def validate_text_payload(data: str, *, max_bytes: int) -> tuple[str, int]:
    """Validate+normalize a text payload: decodable, capped, NUL-free, UTF-8.

    A NUL byte means the payload is not the UTF-8 text it claims to be — the
    cheap mislabel guard ``is_probably_binary`` uses. Beyond that, the bytes
    must actually decode as UTF-8: a ``text`` attachment's content is inlined
    verbatim into the prompt, so a mislabeled binary payload must be rejected
    here rather than silently degrading to replacement characters downstream.
    """
    normalized, decoded = decode_base64_payload(data, max_bytes=max_bytes)
    if b"\x00" in decoded:
        raise ValueError("text attachment contains NUL bytes (not UTF-8 text)")
    try:
        decoded.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("text attachment is not valid UTF-8") from None
    return normalized, len(decoded)


def validate_binary_payload(data: str, *, max_bytes: int) -> tuple[str, int]:
    """Validate+normalize a binary payload: decodable and within the cap only."""
    normalized, decoded = decode_base64_payload(data, max_bytes=max_bytes)
    return normalized, len(decoded)


def decoded_payload_size(data: str) -> int:
    return len(base64.b64decode(data, validate=True))
