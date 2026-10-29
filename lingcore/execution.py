"""Execution-wide admission and typed outcomes; headless driver below."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from lingcore.errors import BudgetExceeded
from lingcore.events import AgentEvent, Error, Final
from lingcore.message import UserInput
from lingcore.tracing import SpanContext, TraceHealth
from lingcore.tracing.context import bind
from lingcore.usage import RequestUsage

if TYPE_CHECKING:
    from lingcore.agent import Agent


class ExecutionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal[
        "completed",
        "failed",
        "timed_out",
        "budget_exceeded",
        "cancelled",
        "interrupted",
    ]
    cause: (
        Literal[
            "agent",
            "provider",
            "environment",
            "verifier",
            "configuration",
            "policy",
            "runner",
        ]
        | None
    ) = None
    limit_type: str | None = None
    provider_error_category: str | None = None


class ExecutionLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wall_seconds: float | None = Field(default=None, gt=0)
    model_requests: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    reported_tokens: int | None = Field(default=None, ge=0)
    usage_missing: Literal["ignore", "stop"] = "ignore"
    automatic_turns: int = Field(default=0, ge=0)


class ExecutionBudget:
    """Synchronous admission; independent of event rendering and trace health."""

    def __init__(self, limits: ExecutionLimits | None = None) -> None:
        self.limits = limits or ExecutionLimits()
        self.requests = 0
        self.tools = 0
        self.tokens = 0
        self.automatic_turns = 0
        self.usage_complete = True
        self.deadline: float | None = None
        self.exhausted: str | None = None
        self._seen_usage: set[str] = set()

    def start(self) -> None:
        if self.deadline is None and self.limits.wall_seconds is not None:
            self.deadline = time.monotonic() + self.limits.wall_seconds

    def _stop(self, limit: str) -> None:
        self.exhausted = limit
        raise BudgetExceeded(limit)

    def _check(self) -> None:
        if self.exhausted is not None:
            raise BudgetExceeded(self.exhausted)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._stop("wall_seconds")

    def reserve_request(self) -> None:
        self._check()
        if self.limits.usage_missing == "stop" and not self.usage_complete:
            self._stop("usage_unavailable")
        if (
            self.limits.model_requests is not None
            and self.requests >= self.limits.model_requests
        ):
            self._stop("model_requests")
        if (
            self.limits.reported_tokens is not None
            and self.tokens >= self.limits.reported_tokens
        ):
            self._stop("reported_tokens")
        self.requests += 1

    def reserve_tool(self) -> None:
        self._check()
        if self.limits.tool_calls is not None and self.tools >= self.limits.tool_calls:
            self._stop("tool_calls")
        self.tools += 1

    def reserve_automatic_turn(self) -> None:
        self._check()
        if self.automatic_turns >= self.limits.automatic_turns:
            self._stop("automatic_turns")
        self.automatic_turns += 1

    def record_usage(self, usage: RequestUsage) -> None:
        if usage.request_id in self._seen_usage:
            return
        self._seen_usage.add(usage.request_id)
        self.tokens += (usage.input_tokens or 0) + (usage.output_tokens or 0)
        self.usage_complete &= usage.availability == "reported"


class ScriptedFrontend:
    """Finite input and policy, driven by the ordinary run_session contract."""

    def __init__(
        self,
        agent: Agent,
        turns: Sequence[str | UserInput],
        *,
        allow_automatic_turns: bool = False,
    ) -> None:
        self.agent = agent
        self._turns = iter(turns)
        self.supports_background = allow_automatic_turns
        self.stopped = False
        self.timed_out = False
        self.terminals: list[Final | Error] = []

    async def read_input(self) -> str | UserInput | None:
        if self.stopped:
            return None
        outcome = self.agent.execution_outcome
        if outcome is not None and outcome.status != "completed":
            return None
        return next(self._turns, None)

    def render(self, event: AgentEvent) -> None:
        if isinstance(event, Final | Error):
            self.terminals.append(event)

    async def confirm(self, command: str) -> bool:
        self.agent.recorder.event(
            "policy.decision",
            attributes={"decision": "deny", "policy": "evaluation_deny"},
        )
        return False

    @contextmanager
    def interrupt_scope(self, stop: Callable[[], bool]) -> Iterator[None]:
        yield


@dataclass(frozen=True)
class HeadlessResult:
    outcome: ExecutionOutcome
    usage: tuple[RequestUsage, ...]
    trace: TraceHealth
    elapsed_ms: int
    cleanup_failed: bool = False


async def run_headless(
    agent: Agent,
    turns: Sequence[str | UserInput],
    *,
    allow_automatic_turns: bool = False,
    cleanup_seconds: float = 5,
    trace_context: SpanContext | None = None,
) -> HeadlessResult:
    """Driver-owned deadline, normal Stop handshake, bounded cleanup on all exits."""
    from lingcore.io.base import run_session

    frontend = ScriptedFrontend(
        agent, turns, allow_automatic_turns=allow_automatic_turns
    )

    # Preserve the Agent's approval instrumentation around the headless policy.
    async def confirm(prompt: str) -> bool:
        with agent.recorder.span(
            "approval.wait", attributes={"policy": "evaluation_deny"}
        ) as span:
            allowed = await frontend.confirm(prompt)
            span.close("denied", decision="deny")
            return allowed

    agent.tool_ctx.confirm = confirm
    budget = agent.execution_budget
    if budget is not None:
        budget.start()
    started = time.monotonic()
    root = (
        agent.recorder.span("evaluation.trial", context=trace_context)
        if trace_context is not None
        else None
    )
    with bind(root.context if root else None):
        session = asyncio.create_task(run_session(agent, frontend))

    async def deadline() -> None:
        assert budget is not None and budget.deadline is not None
        await asyncio.sleep(max(0, budget.deadline - time.monotonic()))
        if not session.done():
            frontend.timed_out = True
            frontend.stopped = True
            if not agent.cancel_turn():
                session.cancel()

    timer = (
        asyncio.create_task(deadline())
        if budget is not None and budget.deadline is not None
        else None
    )
    cleanup_failed = False
    interrupted = False
    failure = False
    try:
        await session
    except asyncio.CancelledError:
        current = asyncio.current_task()
        interrupted = not frontend.timed_out or bool(current and current.cancelling())
        if not session.done():
            session.cancel()
        await asyncio.gather(session, return_exceptions=True)
        if interrupted:
            raise
    except Exception:
        failure = True
    finally:
        if timer is not None:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
        try:
            if agent.turn_pending_finalization:
                agent.finalize_cancelled_turn("execution stopped")
                agent.drain_usage()
        except Exception:
            cleanup_failed = True
        outcome = agent.execution_outcome or ExecutionOutcome(
            status="failed", cause="runner"
        )
        if frontend.timed_out:
            outcome = ExecutionOutcome(
                status="timed_out", cause="agent", limit_type="wall_seconds"
            )
        elif interrupted:
            outcome = ExecutionOutcome(status="cancelled", cause="runner")
        elif failure:
            outcome = ExecutionOutcome(status="failed", cause="runner")
        agent.execution_outcome = outcome
        agent.recorder.event(
            "execution.outcome",
            context=root.context if root else None,
            attributes=outcome.model_dump(),
        )
        if root is not None:
            root.close(
                "ok"
                if outcome.status == "completed"
                else "cancelled"
                if outcome.status in {"cancelled", "timed_out"}
                else "error"
            )
        try:
            async with asyncio.timeout(cleanup_seconds):
                await agent.aclose()
        except Exception:
            cleanup_failed = True
    return HeadlessResult(
        outcome,
        tuple(agent.request_ledger.records),
        agent.recorder.health,
        round((time.monotonic() - started) * 1000),
        cleanup_failed,
    )
