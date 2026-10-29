"""Bounded opt-in content; no reasoning, provider blocks, or secret values."""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_EXCLUDED = {
    "reasoning",
    "reasoning_content",
    "reasoning_delta",
    "anthropic_content",
    "thinking",
    "signature",
    "headers",
    "environment",
    "credentials",
}
_SECRET = re.compile(
    r"(?i)(api[_-]?key|password|secret|access[_-]?token|authorization)"
)
_URL = re.compile(r"https?://[^\s\"<>]+")


def redact_known_secrets(text: str, secrets: tuple[str, ...]) -> str:
    """Short credentials match entire values, never arbitrary substrings.

    Credential fields and authorization syntax are redacted separately below.
    A one-character credential must not corrupt IDs, model names or counts.
    """
    if text in secrets:
        return "[REDACTED]"
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")
    return text


def sanitized_content(
    value: Any, secrets: tuple[str, ...], max_bytes: int
) -> tuple[bytes, bool]:
    def clean_text(text: str) -> str:
        text = redact_known_secrets(text, secrets)

        def url(match: re.Match[str]) -> str:
            try:
                parts = urlsplit(match.group())
                host = parts.hostname or "redacted"
                if ":" in host:
                    host = f"[{host}]"
                if parts.port:
                    host += f":{parts.port}"
                return urlunsplit(
                    (
                        parts.scheme,
                        host,
                        parts.path,
                        urlencode(
                            [(k, "[REDACTED]") for k, _ in parse_qsl(parts.query)]
                        ),
                        "",
                    )
                )
            except ValueError:
                return "[REDACTED URL]"

        text = _URL.sub(url, text)
        text = re.sub(r"(?i)\bbearer\s+[\w.\-]+", "Bearer [REDACTED]", text)
        text = re.sub(
            r"(?i)((?:api[_-]?key|password|secret|token)\s*[=:]\s*)[^\s,;]+",
            r"\1[REDACTED]",
            text,
        )
        # Host home paths are unnecessary for trajectory inspection.
        return re.sub(r"/(?:home|Users)/[^/\s\"]+", "/[private-home]", text)

    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 24:
            return "[TRUNCATED]"
        if isinstance(item, str):
            return clean_text(item)
        if isinstance(item, dict):
            return {
                str(k): "[REDACTED]" if _SECRET.search(str(k)) else clean(v, depth + 1)
                for k, v in item.items()
                if str(k).lower() not in _EXCLUDED
            }
        if isinstance(item, (list, tuple)):
            return [clean(v, depth + 1) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return "[NON-FINITE NUMBER]"
        if item is None or isinstance(item, (int, float, bool)):
            return item
        return "[UNSUPPORTED VALUE]"

    data = json.dumps(clean(value), ensure_ascii=True, allow_nan=False).encode()
    if len(data) <= max_bytes:
        return data, False
    preview = data.decode("ascii")[: max(0, (max_bytes - 80) // 6)]
    return json.dumps({"truncated": True, "preview": preview}).encode(), True


class CommittedContentBuffer:
    """A bounded JSON list of sanitized items, including every wrapper byte.

    Truncate individual values or omit later items; retain the already admitted
    evidence. A whole-list preview would discard all earlier tool arguments.
    """

    def __init__(self, max_bytes: int, secrets: tuple[str, ...]) -> None:
        self.max_bytes = max_bytes
        self.secrets = secrets
        self.clear()

    def clear(self) -> None:
        self._items: list[bytes] = []
        self.byte_count = 2  # opening and closing JSON brackets
        self.truncated = False
        self.omitted_items = 0
        self._prompt_digest: bytes | None = None

    @property
    def data(self) -> bytes:
        return b"[" + b",".join(self._items) + b"]"

    def append(self, kind: str, value: Any) -> None:
        data, truncated = sanitized_content(value, self.secrets, self.max_bytes)
        if kind == "prompt" and not truncated:
            digest = sha256(data).digest()  # only sanitized content is hashed
            if digest == self._prompt_digest:
                return
            self._prompt_digest = digest
        prefix = (
            b'{"kind":' + json.dumps(kind, ensure_ascii=True).encode() + b',"value":'
        )
        suffix = b',"truncated":false}'
        available = (
            self.max_bytes
            - self.byte_count
            - bool(self._items)
            - len(prefix)
            - len(suffix)
        )
        if available < 64:
            self.omitted_items += 1
            self.truncated = True
            return
        if len(data) > available:
            data, truncated = sanitized_content(value, self.secrets, available)
        if truncated:
            suffix = b',"truncated":true}'
        item = prefix + data + suffix
        if self.byte_count + bool(self._items) + len(item) > self.max_bytes:
            self.omitted_items += 1
            self.truncated = True
            return
        self.byte_count += bool(self._items) + len(item)
        self._items.append(item)
        self.truncated |= truncated
