"""Offload oversized tool output to a workspace file ("filesystem as context").

Every tool result is appended to the conversation and so joins the prompt
prefix of all later requests. Heavy results therefore both enlarge each
request's uncached tail and accelerate the window growth that forces
cache-busting eviction. When a result exceeds a threshold this stages the full
text to a file under the workspace and returns a lean preview plus the path, so
the model can pull the rest in slices with ``read_file`` instead of carrying it
all inline. Filenames are content-hashed, so the same output always maps to the
same file (deterministic, idempotent — no timestamps).

``offload_text`` **never raises**: a failed write (read-only workspace, etc.)
degrades to inline truncation, and offload can be disabled per tool (threshold
``<= 0``) without losing the truncation safety cap — consistent with
``ingest.py``'s never-raise contract.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import TYPE_CHECKING

from lingcore.paths import confined_directory

if TYPE_CHECKING:
    from lingcore.tools import ToolContext

# All LingCore runtime artifacts live under this top-level workspace dir.
# ``fs.search`` skips it so the agent never greps its own offloaded output.
RUNTIME_DIRNAME = ".lingcore"
_OFFLOAD_SUBDIR = "tool-output"
DEFAULT_OFFLOAD_OVER_CHARS = 8_000
DEFAULT_HEAD_CHARS = 2_000


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text) - max_chars} more chars)"


def offload_text(
    ctx: "ToolContext",
    *,
    source: str,
    text: str,
    threshold: int,
    head_chars: int = DEFAULT_HEAD_CHARS,
    fallback_max_chars: int = 16_000,
) -> str:
    """Return ``text`` inline when small; otherwise stage it and return a
    preview + pointer. Never raises.

    - ``threshold <= 0``  → offload disabled: keep the legacy truncation cap.
    - ``len(text) <= threshold`` → inline verbatim (small enough to keep).
    - otherwise → write the full text under ``<workspace>/.lingcore/tool-output/``
      and return its head plus a one-line pointer; a write failure degrades to
      truncation.
    """
    if threshold <= 0:
        return _truncate(text, fallback_max_chars)
    if len(text) <= threshold:
        return text
    payload = text.encode("utf-8", "replace")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    directory_rel = f"{RUNTIME_DIRNAME}/{_OFFLOAD_SUBDIR}"
    filename = f"{source}-{digest}.txt"
    rel = f"{directory_rel}/{filename}"
    try:
        # Hold the validated output directory open across the write and final
        # rename. A concurrent parent swap therefore fails closed instead of
        # redirecting this internal artifact outside the workspace, while the
        # unique temporary entry keeps the published content-hash file atomic.
        with confined_directory(
            ctx.workspace, directory_rel, create=True
        ) as directory:
            if not directory.same_bytes(filename, payload):
                part = f".{filename}.{secrets.token_hex(8)}.part"
                try:
                    with directory.open_exclusive(part) as fh:
                        fh.write(payload)
                    directory.replace(part, filename)
                finally:
                    directory.unlink(part, missing_ok=True)
    except Exception:
        return _truncate(text, fallback_max_chars)
    n_lines = text.count("\n") + 1
    n_bytes = len(payload)
    return text[:head_chars] + (
        f"\n… [full output: {n_lines} lines, {n_bytes} bytes → {rel}; "
        "read it with read_file offset/limit]"
    )
