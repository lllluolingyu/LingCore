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
from typing import Literal

from pydantic import BaseModel, Field

from lingcore.errors import ConfigError, ToolError
from lingcore.tools import ToolContext, tool

# The installed package root — writing memory there is forbidden.
_PACKAGE_DIR = Path(__file__).parent.parent.parent.resolve()

_DEFAULT_MAX_BYTES = 65_536
_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)


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

    # Enforce max_bytes on the *final* file content.
    if len(new_content.encode()) > max_bytes:
        raise ToolError(
            f"memory file would exceed max_bytes ({max_bytes}); "
            "use forget to free space first"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    return f"{args.action}: {args.key!r} ok"
