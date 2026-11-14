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
    from lingcore.io.base import Frontend
    from lingcore.tools import ConfirmFn


class RunnerSignal(BaseException):
    """Execution control failure that ordinary tool handlers must not consume."""

    outcome: ExecutionOutcome | None = None


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
    agents: int | None = Field(default=None, ge=0)
    reported_tokens: int | None = Field(default=None, ge=0)
    usage_missing: Literal["ignore", "stop"] = "ignore"
    automatic_turns: int = Field(default=0, ge=0)


class ExecutionBudget:
    """Synchronous admission; independent of event rendering and trace health."""

    def __init__(
        self,
        limits: ExecutionLimits | None = None,
        *,
        parent: ExecutionBudget | None = None,
    ) -> None:
        self.limits = limits or ExecutionLimits()
        self._parent = parent
        self.requests = 0
        self.tools = 0
        self.agents = 0
        self.tokens = 0
        self.automatic_turns = 0
        self.usage_complete = True
        self._deadline: float | None = None
        self.exhausted: str | None = None
        self._seen_usage: set[str] = set()

    def start(self) -> None:
        for budget in self._chain():
            if budget._deadline is None and budget.limits.wall_seconds is not None:
                budget._deadline = time.monotonic() + budget.limits.wall_seconds

    def _chain(self) -> Iterator[ExecutionBudget]:
        budget: ExecutionBudget | None = self
        while budget is not None:
            yield budget
            budget = budget._parent

    def child(self, limits: ExecutionLimits | None = None) -> ExecutionBudget:
        return ExecutionBudget(limits, parent=self)

    @property
    def deadline(self) -> float | None:
        deadlines = [b._deadline for b in self._chain() if b._deadline is not None]
        return min(deadlines, default=None)

    @deadline.setter
    def deadline(self, value: float | None) -> None:
        # Preserve the existing driver/test override; an inherited deadline
        # still bounds this budget even when its local deadline is extended.
        self._deadline = value

    @property
    def remaining_tokens(self) -> int | None:
        """Reported-usage headroom, excluding all unreported in-flight spend."""
        remaining = [
            max(0, b.limits.reported_tokens - b.tokens)
            for b in self._chain()
            if b.limits.reported_tokens is not None
        ]
        return min(remaining, default=None)

    def _stop(self, limit: str) -> None:
        self.exhausted = limit
        raise BudgetExceeded(limit)

    def _check(self) -> None:
        if self.exhausted is not None:
            raise BudgetExceeded(self.exhausted)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._stop("wall_seconds")

    def _check_request(self) -> None:
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

    def check(self) -> None:
        """Propagate an already exhausted ancestor after a contained worker exits."""
        for budget in self._chain():
            budget._check()

    def _reserve(self, counter: str, limit: str) -> None:
        # No await between validation and mutation: a rejected child cannot
        # consume any ancestor's admission capacity on the owning event loop.
        chain = tuple(self._chain())
        for budget in chain:
            if counter == "requests":
                budget._check_request()
            else:
                budget._check()
                cap = getattr(budget.limits, limit)
                if cap is not None and getattr(budget, counter) >= cap:
                    budget._stop(limit)
        for budget in chain:
            setattr(budget, counter, getattr(budget, counter) + 1)

    def reserve_request(self) -> None:
        self._reserve("requests", "model_requests")

    def reserve_tool(self) -> None:
        self._reserve("tools", "tool_calls")

    def reserve_agent(self) -> None:
        self._reserve("agents", "agents")

    def reserve_automatic_turn(self) -> None:
        self._reserve("automatic_turns", "automatic_turns")

    def record_usage(self, usage: RequestUsage) -> None:
        for budget in self._chain():
            if usage.request_id not in budget._seen_usage:
                budget._seen_usage.add(usage.request_id)
                budget.tokens += (usage.input_tokens or 0) + (usage.output_tokens or 0)
                budget.usage_complete &= usage.availability == "reported"


class ScriptedFrontend:
    """Finite input and policy, driven by the ordinary run_session contract."""

    def __init__(
        self,
        agent: Agent,
        turns: Sequence[str | UserInput],
        *,
        allow_automatic_turns: bool = False,
        benchmark_environment: bool = False,
    ) -> None:
        self.agent = agent
        self.benchmark_environment = benchmark_environment
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
            attributes={
                "decision": "allow" if self.benchmark_environment else "deny",
                "policy": "benchmark_environment"
                if self.benchmark_environment
                else "evaluation_deny",
            },
        )
        return self.benchmark_environment

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
    turns: Sequence[str | UserInput] = (),
    *,
    allow_automatic_turns: bool = False,
    cleanup_seconds: float = 5,
    trace_context: SpanContext | None = None,
    runner_signal: asyncio.Future[RunnerSignal] | None = None,
    benchmark_environment: bool = False,
    frontend: Frontend | None = None,
    confirm: ConfirmFn | None = None,
) -> HeadlessResult:
    """Driver-owned deadline, normal Stop handshake, bounded cleanup on all exits.

    Supply either scripted ``turns`` or a ``frontend`` that can supply nudges.
    ``confirm`` replaces the frontend's policy without losing approval tracing.
    The driver closes the Agent on completion, failure, and cancellation.
    """
    from lingcore.io.base import run_session

    if frontend is not None and turns:
        raise ValueError("pass either turns or frontend, not both")
    target = (
        frontend
        if frontend is not None
        else ScriptedFrontend(
            agent,
            turns,
            allow_automatic_turns=allow_automatic_turns,
            benchmark_environment=benchmark_environment,
        )
    )

    # Preserve the Agent's approval instrumentation around the headless policy.
    policy = (
        "worker"
        if frontend is not None or confirm is not None
        else ("benchmark_environment" if benchmark_environment else "evaluation_deny")
    )
    confirmation = confirm if confirm is not None else target.confirm

    async def traced_confirm(prompt: str) -> bool:
        with agent.recorder.span(
            "approval.wait",
            attributes={"policy": policy},
        ) as span:
            allowed = await confirmation(prompt)
            span.close(
                "ok" if allowed else "denied", decision="allow" if allowed else "deny"
            )
            return allowed

    agent.tool_ctx.confirm = traced_confirm
    # The wrapper owns stop state even when a custom frontend is waiting for
    # input or supplies a post-result nudge.
    from lingcore._execution_frontend import HeadlessFrontend

    driver = HeadlessFrontend(agent, target)
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
        session = asyncio.create_task(run_session(agent, driver))

    async def deadline() -> None:
        assert budget is not None and budget.deadline is not None
        await asyncio.sleep(max(0, budget.deadline - time.monotonic()))
        if not session.done():
            driver.timed_out = True
            if not driver.stop():
                session.cancel()

    timer = (
        asyncio.create_task(deadline())
        if budget is not None and budget.deadline is not None
        else None
    )
    received_signal: RunnerSignal | None = None

    async def monitor_signal() -> None:
        nonlocal received_signal
        assert runner_signal is not None
        received_signal = await asyncio.shield(runner_signal)
        if not driver.stop():
            session.cancel()

    monitor = (
        asyncio.create_task(monitor_signal()) if runner_signal is not None else None
    )
    cleanup_failed = False
    interrupted = False
    failure = False
    try:
        await session
    except asyncio.CancelledError:
        current = asyncio.current_task()
        interrupted = (not driver.timed_out and received_signal is None) or bool(
            current and current.cancelling()
        )
        if not session.done():
            session.cancel()
        await asyncio.gather(session, return_exceptions=True)
        if interrupted:
            raise
    except (RunnerSignal, Exception):
        failure = True
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
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
        if driver.timed_out:
            outcome = ExecutionOutcome(
                status="timed_out", cause="agent", limit_type="wall_seconds"
            )
        elif interrupted:
            outcome = ExecutionOutcome(status="cancelled", cause="runner")
        elif received_signal is not None:
            outcome = received_signal.outcome or ExecutionOutcome(
                status="failed", cause="runner"
            )
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
