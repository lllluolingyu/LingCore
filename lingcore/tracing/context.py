"""Explicit span identities with task-local propagation as a convenience."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

from lingcore.tracing.models import SpanContext

current_span: ContextVar[SpanContext | None] = ContextVar("lingcore_span", default=None)
request_purpose: ContextVar[str] = ContextVar(
    "lingcore_request_purpose", default="main"
)


def new_context(
    parent: SpanContext | None = None, **identity: str | None
) -> SpanContext:
    values = parent.model_dump() if parent else {}
    values.update(
        trace_id=parent.trace_id if parent else uuid4().hex,
        span_id=uuid4().hex[:16],
        parent_span_id=parent.span_id if parent else None,
    )
    values.update(identity)
    return SpanContext.model_validate(values)


@contextmanager
def bind(context: SpanContext | None) -> Iterator[None]:
    token = current_span.set(context)
    try:
        yield
    finally:
        current_span.reset(token)


@contextmanager
def purpose(value: str) -> Iterator[None]:
    token = request_purpose.set(value)
    try:
        yield
    finally:
        request_purpose.reset(token)
