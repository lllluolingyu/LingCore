"""Instrument the duck-typed client seam once, for every model purpose."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from lingcore.errors import INCOMPLETE_FINISH_REASONS, LLMStreamError
from lingcore.llm import LLMChunk
from lingcore.message import Message
from lingcore.tracing.context import request_purpose
from lingcore.tracing.models import Scalar, SpanStatus
from lingcore.tracing.recorder import TraceRecorder
from lingcore.usage import (
    RequestUsage,
    TokenUsage,
    billing_observer,
    request_usage_observer,
)

if TYPE_CHECKING:
    from lingcore.execution import ExecutionBudget


class StreamingClient(Protocol):
    def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]: ...


@asynccontextmanager
async def closing_stream(
    iterator: AsyncIterator[LLMChunk],
) -> AsyncIterator[AsyncIterator[LLMChunk]]:
    try:
        yield iterator
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


class RequestLedger:
    def __init__(self) -> None:
        self.records: list[RequestUsage] = []
        self.last_request_id: str | None = None
        self.retry_request_id: str | None = None


class InstrumentedClient:
    def __init__(
        self,
        client: StreamingClient,
        recorder: TraceRecorder,
        ledger: RequestLedger,
        *,
        model: str,
        provider: str,
        max_retries: int = 0,
        default_purpose: str | None = None,
        budget: ExecutionBudget | None = None,
    ) -> None:
        self.client, self.recorder, self.ledger = client, recorder, ledger
        self.model, self.provider, self.max_retries = model, provider, max_retries
        self.default_purpose, self.budget = default_purpose, budget

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]:
        if self.budget is not None:
            self.budget.reserve_request()
        request_id = str(uuid4())
        purpose = self.default_purpose or request_purpose.get()
        self.ledger.last_request_id = request_id
        attrs: dict[str, Scalar] = {
            "request_id": request_id,
            "purpose": purpose,
            "model": self.model,
            "provider": self.provider,
            "max_retries": self.max_retries,
            "transport_attempts": None,
        }
        if purpose == "main" and self.ledger.retry_request_id is not None:
            attrs["previous_request_id"] = self.ledger.retry_request_id
            self.ledger.retry_request_id = None
        span = self.recorder.span("llm.request", attributes=attrs)
        observed: TokenUsage | None = None
        billing: dict[str, int] = {}
        complete = True

        def observe(usage: TokenUsage) -> None:
            nonlocal observed
            observed = usage

        def raw(values: dict[str, int], available: bool) -> None:
            nonlocal complete
            billing.update(values)
            complete = available

        @contextmanager
        def bound() -> Iterator[None]:
            usage_token = request_usage_observer.set(observe)
            billing_token = billing_observer.set(raw)
            try:
                with span.bind():
                    yield
            finally:
                billing_observer.reset(billing_token)
                request_usage_observer.reset(usage_token)

        iterator: AsyncIterator[LLMChunk] | None = None
        start = time.monotonic_ns()
        status: SpanStatus = "ok"
        finish: str | None = None
        ttft: int | None = None
        first: int | None = None
        try:
            with bound():
                iterator = self.client.stream(messages, tools=tools)
            while True:
                try:
                    with bound():
                        chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                if first is None and (
                    chunk.text_delta or chunk.reasoning_delta or chunk.tool_calls
                ):
                    first = time.monotonic_ns() - start
                if ttft is None and chunk.text_delta:
                    ttft = time.monotonic_ns() - start
                finish = chunk.finish_reason or finish
                yield chunk
            if finish in INCOMPLETE_FINISH_REASONS:
                status = "incomplete"
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        except BaseException as exc:
            status = (
                "incomplete"
                if isinstance(exc, LLMStreamError)
                and exc.finish_reason in INCOMPLETE_FINISH_REASONS
                else "error"
            )
            if isinstance(exc, LLMStreamError):
                finish = exc.finish_reason or finish
            raise
        finally:
            propagating = sys.exc_info()[1] is not None
            try:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    with bound():
                        await close()
            except BaseException:
                status = "error" if not propagating else status
                if not propagating:
                    raise
            finally:
                values: dict[str, Any] = {}
                if observed is not None:
                    values = asdict(observed)
                usage = RequestUsage(
                    **({"model": self.model} | values | billing),
                    request_id=request_id,
                    purpose=purpose,
                    availability="missing"
                    if observed is None
                    else "reported"
                    if complete
                    else "partial",
                )
                self.ledger.records.append(usage)
                if self.budget is not None:
                    self.budget.record_usage(usage)
                self.recorder.event(
                    "request.usage", context=span.context, attributes=asdict(usage)
                )
                span.close(
                    status, finish_reason=finish, ttft_ns=ttft, first_delta_ns=first
                )
