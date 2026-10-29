"""Nonblocking recording and exactly-once operation scopes."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from uuid import uuid4

from lingcore.errors import BudgetExceeded
from lingcore.tracing.content import redact_known_secrets
from lingcore.tracing.context import bind, current_span, new_context
from lingcore.tracing.models import (
    Scalar,
    SpanContext,
    SpanStatus,
    TraceHealth,
    TraceRecord,
    allow_attributes,
)


class TraceSink(Protocol):
    def try_emit(self, record: TraceRecord) -> bool: ...
    def try_artifact(self, artifact_id: str, data: bytes) -> bool: ...
    @property
    def health(self) -> TraceHealth: ...
    async def flush(self, timeout_seconds: float = 2) -> TraceHealth: ...
    async def aclose(self, timeout_seconds: float = 2) -> TraceHealth: ...


class TraceRecorder:
    def __init__(
        self,
        sink: TraceSink | None = None,
        *,
        run_id: str | None = None,
        content: str = "metadata",
        secrets: tuple[str, ...] = (),
        artifact_max_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.sink = sink
        self.run_id = run_id or str(uuid4())
        self.content = content
        self.secrets = secrets
        self.artifact_max_bytes = artifact_max_bytes
        self._start = time.monotonic_ns()
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.sink is not None

    @property
    def health(self) -> TraceHealth:
        return self.sink.health if self.sink else TraceHealth()

    def emit(
        self,
        name: str,
        kind: Literal["span.start", "span.end", "event"],
        context: SpanContext,
        attributes: Mapping[str, Scalar],
        status: SpanStatus | None = None,
    ) -> None:
        if self.sink is None:
            return
        sanitized: dict[str, Scalar] = dict(attributes)
        for key, value in sanitized.items():
            if isinstance(value, str):
                sanitized[key] = redact_known_secrets(value, self.secrets)[:256]
        clean = allow_attributes(name, sanitized)
        with self._lock:
            self._seq += 1
            record = TraceRecord(
                **context.model_dump(),
                event_id=str(uuid4()),
                run_id=self.run_id,
                seq=self._seq,
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                elapsed_ns=max(0, time.monotonic_ns() - self._start),
                kind=kind,
                name=name,
                status=status,
                attributes=clean,
            )
            self.sink.try_emit(record)

    def event(
        self,
        name: str,
        *,
        context: SpanContext | None = None,
        attributes: Mapping[str, Scalar] | None = None,
    ) -> None:
        self.emit(
            name,
            "event",
            context or current_span.get() or new_context(),
            attributes or {},
        )

    def span(
        self,
        name: str,
        *,
        parent: SpanContext | None = None,
        attributes: Mapping[str, Scalar] | None = None,
        context: SpanContext | None = None,
    ) -> SpanScope:
        return SpanScope(
            self,
            name,
            context or new_context(parent or current_span.get()),
            attributes or {},
        )

    def capture(
        self, kind: str, value: Any, *, context: SpanContext | None = None
    ) -> None:
        if self.sink is None or self.content != "redacted":
            return
        from lingcore.tracing.content import sanitized_content

        data, truncated = sanitized_content(
            value, self.secrets, self.artifact_max_bytes
        )
        self._capture_sanitized(kind, data, context=context, truncated=truncated)

    def _capture_sanitized(
        self,
        kind: str,
        data: bytes,
        *,
        context: SpanContext | None = None,
        truncated: bool = False,
        omitted_items: int = 0,
    ) -> None:
        """Publish bytes already sanitized and bounded by the content buffer."""
        if self.sink is None or self.content != "redacted":
            return
        if len(data) > self.artifact_max_bytes:
            raise ValueError("sanitized artifact exceeds its byte allowance")
        artifact_id = uuid4().hex
        if self.sink.try_artifact(artifact_id, data):
            self.event(
                "content.capture",
                context=context,
                attributes={
                    "artifact_id": artifact_id,
                    "content.kind": kind,
                    "redacted": True,
                    "truncated": truncated,
                    "omitted_items": omitted_items,
                },
            )

    async def aclose(self, timeout_seconds: float = 2) -> TraceHealth:
        return await self.sink.aclose(timeout_seconds) if self.sink else self.health

    def abort(self) -> None:
        """Release a local writer when synchronous Agent assembly fails."""
        abort = getattr(self.sink, "abort", None)
        if abort is not None:
            abort()


class SpanScope:
    def __init__(
        self,
        recorder: TraceRecorder,
        name: str,
        context: SpanContext,
        attributes: Mapping[str, Scalar],
    ) -> None:
        self.recorder, self.name, self.context = recorder, name, context
        self.attributes = dict(attributes)
        self._start = time.monotonic_ns()
        self._closed = False
        self._binding: Any = None
        recorder.emit(name, "span.start", context, self.attributes)

    def bind(self) -> Any:
        return bind(self.context)

    def close(self, status: SpanStatus = "ok", **attributes: Scalar) -> None:
        if self._closed:
            return
        self._closed = True
        self.recorder.emit(
            self.name,
            "span.end",
            self.context,
            self.attributes
            | attributes
            | {"duration_ns": max(0, time.monotonic_ns() - self._start)},
            status,
        )

    def __enter__(self) -> SpanScope:
        self._binding = self.bind()
        self._binding.__enter__()
        return self

    def __exit__(self, typ: Any, exc: Any, tb: Any) -> None:
        try:
            status: SpanStatus = "ok"
            if typ is not None:
                status = (
                    "budget_exceeded"
                    if issubclass(typ, BudgetExceeded)
                    else "cancelled"
                    if issubclass(typ, (asyncio.CancelledError, GeneratorExit))
                    else "error"
                )
            self.close(status, **({"error.type": typ.__name__} if typ else {}))
        finally:
            self._binding.__exit__(typ, exc, tb)


class NoopRecorder(TraceRecorder):
    """Same API without a writer, filesystem access, or captured values."""
