"""knowledge — grep, semantic, and hybrid retrieval over workspace files.

``grep`` remains the zero-cost, offline default.  The ``index`` and ``hybrid``
backends use a local incremental SQLite index; semantic vectors are created
only after the profile explicitly sets ``embedding.enabled: true``.  The model
provider is injected through ``ToolContext`` (or lazily built from config), so
the index and tool never depend on one vendor's SDK.

The index stores deterministic text chunks with source path, page, and line
metadata.  Content hashes avoid re-embedding unchanged chunks, full refreshes
remove deleted sources, and queries exclude changed/deleted documents until the
operator indexes again rather than returning stale citations.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Literal

from pydantic import BaseModel, Field

from lingcore.errors import ToolError
from lingcore.tools import ToolContext, tool
from lingcore.tools.builtin._knowledge.common import (
    _DEFAULT_GREP_TIME_BUDGET_MS,
    _GREP_LINE_CHARS,
    _MAX_HITS,
    _backend,
    _int_option,
    _knowledge_options,
    _sources,
)
from lingcore.tools.builtin._knowledge.common import (
    EMBEDDING_PROVIDER_KEY as EMBEDDING_PROVIDER_KEY,
)
from lingcore.tools.builtin._knowledge.common import (
    RERANKING_PROVIDER_KEY as RERANKING_PROVIDER_KEY,
)
from lingcore.tools.builtin._knowledge.grep import _grep
from lingcore.tools.builtin._knowledge.indexing import _index_sources
from lingcore.tools.builtin._knowledge.retrieval import (
    _query_index,
    _status,
)
from lingcore.tools.builtin._knowledge.sources import _selector_list


class KnowledgeArgs(BaseModel):
    action: Literal["query", "index", "status"] = Field(
        description="Operation: query (search), index (build/update), or status."
    )
    query: str | None = Field(
        default=None, description="Search query (required for query action)."
    )
    paths: list[str] | None = Field(
        default=None,
        description=(
            "Optional workspace-relative files, directories, or globs to update "
            "during an index action."
        ),
    )


@tool(
    description=(
        "Search the knowledge corpus of workspace files. Use `query` to find "
        "relevant, source-cited excerpts; `index` to incrementally build/update an "
        "opt-in semantic index; and `status` to inspect index freshness."
    )
)
async def knowledge(args: KnowledgeArgs, ctx: ToolContext) -> str:
    options = _knowledge_options(ctx)
    backend = _backend(options)
    sources = _sources(options)
    base = ctx.workspace.resolve()

    if args.action == "status":
        return _status(ctx, options, sources, backend)

    if args.action == "index":
        if backend == "grep":
            return "grep backend needs no index; query directly"
        return await _index_sources(ctx, options, sources, _selector_list(args.paths))

    if args.action == "query":
        if not args.query or not args.query.strip():
            raise ToolError("query action requires a query string")
        query = args.query.strip()
        if backend == "grep":
            cancel_event = threading.Event()
            try:
                hits = await asyncio.to_thread(
                    _grep,
                    base,
                    sources,
                    query,
                    max_hits=_int_option(
                        options, "max_hits", _MAX_HITS, minimum=1, maximum=100
                    ),
                    max_line_chars=_int_option(
                        options,
                        "max_line_chars",
                        _GREP_LINE_CHARS,
                        minimum=20,
                        maximum=10_000,
                    ),
                    time_budget_ms=_int_option(
                        options,
                        "time_budget_ms",
                        _DEFAULT_GREP_TIME_BUDGET_MS,
                        minimum=1,
                        maximum=120_000,
                    ),
                    cancel_event=cancel_event,
                )
            except BaseException:
                # The worker checks this between lines/matches; cancellation
                # cannot interrupt a single regex call, but the per-match
                # timeout above is always bounded by the remaining budget.
                cancel_event.set()
                raise
            return "\n".join(hits) if hits else "(no matches)"
        return await _query_index(ctx, options, sources, query, backend)

    raise ToolError(f"unknown knowledge action: {args.action!r}")
