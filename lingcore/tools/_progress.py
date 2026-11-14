"""Task-local routing for transient tool progress; never persistent state."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from lingcore.events import ToolProgress

progress_sink: ContextVar[Callable[[ToolProgress], None] | None] = ContextVar(
    "tool_progress_sink", default=None
)
invocation: ContextVar[tuple[object, str] | None] = ContextVar(
    "tool_progress_invocation", default=None
)


def report(context: object, payload: dict[str, Any]) -> None:
    sink, call = progress_sink.get(), invocation.get()
    if sink is None or call is None or call[0] is not context:
        return
    # Bound and snapshot plugin-owned data. Progress is best-effort and must
    # never fail a tool or retain a mutable reference owned by the producer.
    try:
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded) <= 16_000:
            sink(ToolProgress(call[1], json.loads(encoded)))
    except (TypeError, ValueError, RecursionError):
        pass
