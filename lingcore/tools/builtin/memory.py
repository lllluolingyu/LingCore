"""memory — persistent, profile-scoped agent memory.

Stores key/value entries as ``## key`` sections in a Markdown file that lives
in the profile directory (not the workspace).  The file is read automatically
by LayeredComposer on every compose() call; this tool handles explicit writes.

Security constraints enforced here:
- Relative paths are confined to profile_dir (no ``../`` escape).
- Absolute paths require ``allow_absolute_path: true`` in options.
- Writing into an installed package directory is blocked.
- ``max_bytes`` is enforced on the *final* file content after each write.
- Duplicate keys are rejected on ``remember``; missing keys on ``modify``/``forget``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from lingcore.errors import ConfigError, ToolError
from lingcore.message import Message
from lingcore.tools import ToolContext, tool

# The installed package root — writing memory there is forbidden.
_PACKAGE_DIR = Path(__file__).parent.parent.parent.resolve()

_DEFAULT_MAX_BYTES = 65_536
_DEFAULT_COMPACT_AT_RATIO = 0.8
_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)

# ctx.options key under which from_profile injects the duck-typed summarizer used
# for auto-compaction (mirrors skill.py's SKILL_STATE_KEY). Absent ⇒ disabled.
MEMORY_SUMMARIZER_KEY = "_memory_summarizer"

_MEMORY_COMPACT_SYSTEM = (
    "You condense an AI agent's long-term memory file. Merge duplicate or "
    "overlapping entries, drop superseded or stale notes, and keep every durable "
    "fact, decision, and user preference. Preserve the Markdown structure: each "
    "entry is a '## key' heading followed by its body. Output only the condensed "
    "memory file — no preamble or commentary."
)


# --------------------------------------------------------------------------- #
# Path resolution                                                              #
# --------------------------------------------------------------------------- #

def _resolve_memory_path(ctx: ToolContext) -> Path:
    opts = ctx.options.get("memory", {})
    raw = Path(opts.get("path", "memory.md"))

    if raw.is_absolute():
        if not opts.get("allow_absolute_path", False):
            raise ConfigError(
                "memory path is absolute; set allow_absolute_path: true to permit this"
            )
        return raw.resolve()

    if ctx.profile_dir is None:
        raise ToolError("memory tool requires a profile directory (profile_dir not set)")

    resolved = (ctx.profile_dir / raw).resolve()
    if not resolved.is_relative_to(ctx.profile_dir.resolve()):
        raise ConfigError(f"memory path escapes profile directory: {raw}")

    # Block writes into the installed package tree.
    try:
        resolved.relative_to(_PACKAGE_DIR)
        raise ToolError(
            "cannot write memory into the installed package directory; "
            "set an explicit absolute path with allow_absolute_path: true, "
            f"e.g. path: ${{HOME}}/.local/state/lingcore/memory.md"
        )
    except ValueError:
        pass  # not under _PACKAGE_DIR — good

    return resolved


# --------------------------------------------------------------------------- #
# Section helpers                                                              #
# --------------------------------------------------------------------------- #

def _parse(text: str) -> dict[str, str]:
    """Return ordered {key: body} mapping from flat ## sections."""
    entries: dict[str, str] = {}
    parts = re.split(r"^## .+$", text, flags=re.MULTILINE)
    keys = _HEADING.findall(text)
    for key, body in zip(keys, parts[1:]):
        entries[key] = body.strip()
    return entries


def _serialise(entries: dict[str, str]) -> str:
    if not entries:
        return ""
    return "\n\n".join(f"## {k}\n{v}" for k, v in entries.items()) + "\n"


async def _compact_memory(summarizer: Any, content: str, max_bytes: int) -> str | None:
    """Condense ``memory.md`` via the duck-typed summarizer.

    Returns the condensed file (re-normalized to canonical ``## key`` form and
    within ``max_bytes``) or ``None`` on any failure — a missing/broken
    summarizer, an unparseable reply, or one that didn't shrink enough — so the
    caller can fall back to the hard-cap guard. Never raises (invariant: a tool
    failure becomes a ToolResult, and compaction must not turn a write into a
    crash). Reached through the same ``stream`` seam as the rest of the runtime,
    so this module imports nothing from ``llm.py``.
    """
    request = [
        Message.system(_MEMORY_COMPACT_SYSTEM),
        Message.user(
            f"Condense this memory file to well under {max_bytes} bytes, keeping "
            f"the '## key' + body format:\n\n{content}"
        ),
    ]
    try:
        parts: list[str] = []
        async for chunk in summarizer.stream(request, tools=None):
            if getattr(chunk, "text_delta", ""):
                parts.append(chunk.text_delta)
        condensed = "".join(parts).strip()
    except Exception:
        return None
    entries = _parse(condensed)
    if not entries:
        return None  # no ## sections parsed → unusable, keep the original
    normalized = _serialise(entries)  # canonical form; drops any stray preamble
    if len(normalized.encode()) > max_bytes:
        return None  # didn't shrink enough
    return normalized


# --------------------------------------------------------------------------- #
# Tool                                                                         #
# --------------------------------------------------------------------------- #

class MemoryArgs(BaseModel):
    action: Literal["remember", "forget", "modify", "read"] = Field(
        description="Operation to perform on the memory file."
    )
    key: str | None = Field(
        default=None,
        description="Section heading that identifies the memory entry.",
    )
    content: str | None = Field(
        default=None,
        description="Text to store (required for remember / modify).",
    )


@tool(description=(
    "Read or update the agent's persistent memory file. "
    "Use `remember` to store a new entry, `modify` to update an existing one, "
    "`forget` to remove one, and `read` to inspect the full file."
))
async def memory(args: MemoryArgs, ctx: ToolContext) -> str:
    path = _resolve_memory_path(ctx)
    opts = ctx.options.get("memory", {})
    max_bytes: int = int(opts.get("max_bytes", _DEFAULT_MAX_BYTES))

    existing = path.read_text("utf-8") if path.is_file() else ""
    entries = _parse(existing)

    if args.action == "read":
        return existing if existing.strip() else "(memory is empty)"

    # Validate key presence for write operations.
    if args.key is None:
        raise ToolError(f"action {args.action!r} requires a key")

    if args.action == "remember":
        if args.key in entries:
            raise ToolError(
                f"key {args.key!r} already exists; use modify to update it"
            )
        if args.content is None:
            raise ToolError("remember requires content")
        entries[args.key] = args.content

    elif args.action == "modify":
        if args.key not in entries:
            raise ToolError(
                f"key {args.key!r} not found; use remember to create it"
            )
        if args.content is None:
            raise ToolError("modify requires content")
        entries[args.key] = args.content

    elif args.action == "forget":
        if args.key not in entries:
            raise ToolError(f"key {args.key!r} not found")
        del entries[args.key]

    new_content = _serialise(entries)
    note = ""

    # Auto-compaction: when the file crosses the soft length limit, condense it
    # with the injected summarizer instead of hard-failing. memory.md changes
    # only on write, so this is the precise trigger point. Gated on BOTH the
    # opt-in flag and an injected summarizer, so a directly-constructed context
    # (tests, alternate entrypoints) can't compact unless auto_compact is set.
    # The max_bytes guard below still catches a disabled/unavailable summarizer
    # or one that didn't shrink the file enough.
    summarizer = ctx.options.get(MEMORY_SUMMARIZER_KEY)
    compact_at = int(
        max_bytes * float(opts.get("compact_at_ratio", _DEFAULT_COMPACT_AT_RATIO))
    )
    if (
        opts.get("auto_compact", False)
        and summarizer is not None
        and len(new_content.encode()) > compact_at
    ):
        condensed = await _compact_memory(summarizer, new_content, max_bytes)
        if condensed is not None:
            before_n = len(new_content.encode())
            new_content = condensed
            note = (
                f" [memory auto-compacted {before_n}→{len(new_content.encode())} bytes]"
            )

    # Enforce max_bytes on the *final* file content (fallback guard).
    if len(new_content.encode()) > max_bytes:
        raise ToolError(
            f"memory file would exceed max_bytes ({max_bytes}); "
            "use forget to free space first"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    return f"{args.action}: {args.key!r} ok{note}"
