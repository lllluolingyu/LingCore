"""Terminal attachments helpers."""

from __future__ import annotations

import os
from pathlib import Path

from lingcore.io._cli.definitions import _ATTACH_RE
from lingcore.media import attachment_from_path
from lingcore.media_types import (
    MAX_ATTACHMENTS,
    TOTAL_ATTACHMENT_MAX_BYTES,
    decoded_payload_size,
)
from lingcore.message import Attachment, UserInput


def _looks_like_path(raw: str) -> bool:
    """Heuristic for whether an unmatched ``@token`` plausibly meant a file.

    Used only to decide whether to warn: a bare ``@mention`` shouldn't nag,
    but a mistyped ``@notes.txt`` should not vanish silently.
    """
    return "/" in raw or raw.startswith("~") or bool(Path(raw).suffix)


def _parse_attachments(
    line: str, base: Path | None = None
) -> tuple[UserInput, list[str]]:
    """Parse ``@path`` CLI attachments from one line.

    A token attaches iff it resolves to an existing file (any type); otherwise
    the ``@token`` stays in the text verbatim. Returns the parsed input plus
    warning lines for tokens that looked like a path but couldn't attach (no
    such file, too large, over the per-message limit) — the typed line is never
    lost, and caps can never make ``UserInput`` construction raise.
    """
    base = base or Path.cwd()
    attachments: list[Attachment] = []
    warnings: list[str] = []
    out: list[str] = []
    pos = 0
    total = 0
    for match in _ATTACH_RE.finditer(line):
        token = match.group(1)
        raw_path = (
            token[1:-1] if token.startswith('"') and token.endswith('"') else token
        )
        path = Path(os.path.expanduser(raw_path))
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            if _looks_like_path(raw_path):
                warnings.append(f"@{raw_path}: no such file; sent as text")
            continue
        if len(attachments) >= MAX_ATTACHMENTS:
            warnings.append(
                f"@{raw_path}: attachment limit ({MAX_ATTACHMENTS}) reached; "
                "sent as text"
            )
            continue
        try:
            attachment = attachment_from_path(path)
        except Exception as e:
            warnings.append(
                f"@{raw_path}: {e}; sent as text — copy it into the workspace "
                "and ask the agent to read it with tools"
            )
            continue
        size = decoded_payload_size(attachment.data)
        if total + size > TOTAL_ATTACHMENT_MAX_BYTES:
            warnings.append(
                f"@{raw_path}: would exceed the total attachment size limit; "
                "sent as text"
            )
            continue
        attachments.append(attachment)
        total += size
        out.append(line[pos : match.start()])
        out.append(raw_path)
        pos = match.end()
    out.append(line[pos:])
    return UserInput(text="".join(out), attachments=attachments), warnings
