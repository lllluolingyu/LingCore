"""knowledge — search a corpus of workspace files.

A hybrid retrieval tool.  The ``grep`` backend (built here) does a full-text
substring/regex scan over configured source globs, confined to the workspace.
The ``index``/``hybrid`` backends (embeddings + SQLite cosine store) are gated
behind a numpy availability check and raise a clear ConfigError until built.

All paths are confined to the workspace via the same containment check the fs
tools use, so knowledge can never read outside the workspace.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from lingcore.errors import ConfigError, ToolError
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._offload import RUNTIME_DIRNAME

_MAX_READ_BYTES = 256 * 1024
_MAX_HITS = 50
_GREP_LINE_CHARS = 200
_DEFAULT_SOURCES = ["**/*"]


def _confined(base: Path, p: Path) -> bool:
    return p == base or p.is_relative_to(base)


def _iter_source_files(base: Path, sources: list[str]):
    for pattern in sources:
        pat = Path(pattern)
        if pat.is_absolute() or any(part == ".." for part in pat.parts):
            raise ToolError(f"knowledge source escapes workspace: {pattern!r}")
        for p in sorted(base.glob(pattern)):
            try:
                full = p.resolve()
            except OSError:
                continue
            if not _confined(base, full) or not full.is_file():
                continue
            if RUNTIME_DIRNAME in full.relative_to(base).parts:
                continue  # skip LingCore's own runtime artifacts
            yield full


def _grep(
    base: Path,
    sources: list[str],
    query: str,
    *,
    max_hits: int = _MAX_HITS,
    max_line_chars: int = _GREP_LINE_CHARS,
) -> list[str]:
    hits: list[str] = []
    try:
        rx = re.compile(query)
    except re.error:
        rx = None  # fall back to substring matching

    for full in _iter_source_files(base, sources):
        try:
            if full.stat().st_size > _MAX_READ_BYTES:
                continue
            text = full.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = full.relative_to(base)
        for lineno, line in enumerate(text.splitlines(), start=1):
            matched = (rx.search(line) if rx else False) or (query in line)
            if matched:
                hits.append(f"{rel}:{lineno}: {line.strip()[:max_line_chars]}")
                if len(hits) >= max_hits:
                    hits.append(f"... (truncated at {max_hits} hits)")
                    return hits
    return hits


class KnowledgeArgs(BaseModel):
    action: Literal["query", "index", "status"] = Field(
        description="Operation: query (search), index (build), or status."
    )
    query: str | None = Field(
        default=None, description="Search query (required for query action)."
    )
    paths: list[str] | None = Field(
        default=None, description="Optional subset of paths to (re)index."
    )


@tool(description=(
    "Search the knowledge corpus of workspace files. Use `query` to find "
    "relevant excerpts (returns file:line matches), `index` to build the "
    "semantic index, and `status` to report index state."
))
async def knowledge(args: KnowledgeArgs, ctx: ToolContext) -> str:
    opts = ctx.options.get("knowledge", {})
    backend = opts.get("backend", "grep")
    sources = opts.get("sources", _DEFAULT_SOURCES)
    base = ctx.workspace.resolve()

    if args.action == "status":
        n = sum(1 for _ in _iter_source_files(base, sources))
        return f"backend={backend}, source files={n}"

    if args.action == "index":
        if backend == "grep":
            return "grep backend needs no index; query directly"
        _require_numpy()
        raise ConfigError(f"knowledge backend {backend!r} index build is not implemented yet")

    if args.action == "query":
        if not args.query:
            raise ToolError("query action requires a query string")
        if backend in ("index", "hybrid"):
            _require_numpy()
            raise ConfigError(
                f"knowledge backend {backend!r} is not implemented yet; use backend: grep"
            )
        hits = _grep(
            base,
            sources,
            args.query,
            max_hits=int(opts.get("max_hits", _MAX_HITS)),
            max_line_chars=int(opts.get("max_line_chars", _GREP_LINE_CHARS)),
        )
        return "\n".join(hits) if hits else "(no matches)"

    raise ToolError(f"unknown knowledge action: {args.action!r}")


def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise ConfigError(
            "knowledge index backend requires numpy; run `uv add numpy`"
        ) from None
