"""Public turn driver, tracing, and failure recovery."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING
from uuid import uuid4

from lingcore.background import BackgroundInput
from lingcore.errors import (
    BudgetExceeded,
)
from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    TurnStatsReported,
    UsageReported,
)
from lingcore.execution import ExecutionOutcome
from lingcore.message import UserInput
from lingcore.tracing.content import CommittedContentBuffer
from lingcore.tracing.context import bind, current_span, new_context
from lingcore.tracing.models import TraceLink
from lingcore.usage import TurnClock

if TYPE_CHECKING:
    from lingcore.agent import Agent


async def run(
    self: Agent, user_input: str | UserInput
) -> AsyncGenerator[AgentEvent, None]:
    """Drive one user turn to completion, yielding events as they happen."""
    if self._closed:
        yield Error("agent is closed")
        return
    # The checkpoint marks an active turn and the identity token owns it.
    # In particular, a cancelled driver task is already ``done`` but its
    # partial state is still live until ``finalize_cancelled_turn`` has
    # repaired both memory and durable history.
    if self._turn_checkpoint is not None:
        yield Error("another turn is already running")
        return
    try:
        if self._session_store is not None and self._session_lease is not None:
            self._session_store.heartbeat_session(self._session_lease)
        self._retry_pending_turn_cleanup()
    except Exception as exc:
        yield Error(
            "cannot start turn until durable rollback succeeds: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    # Snapshot into locals first. No await separates the lease check from
    # assignment, while a custom memory/state getter that raises cannot
    # leave a half-acquired lease behind.
    # The protocol does not rely on an implementation returning a copy:
    # snapshot explicitly so an append cannot alias and mutate rollback
    # state in a custom memory implementation.
    turn_checkpoint = list(self.memory.messages)
    turn_index_checkpoint = self._turn_index
    turn_skills_checkpoint = (
        list(self.skill_state.active) if self.skill_state is not None else None
    )
    turn_skill_approvals_checkpoint = (
        dict(self.skill_state.approved_high_risk)
        if self.skill_state is not None
        else None
    )
    turn_todos_checkpoint = (
        self.todo_state.items if self.todo_state is not None else None
    )
    lease = object()
    self._active_turn_task = asyncio.current_task()
    self._turn_lease = lease
    self._turn_started_at = time.monotonic()
    self._turn_checkpoint = turn_checkpoint
    self._turn_user_message = None
    self._turn_index_checkpoint = turn_index_checkpoint
    self._turn_skills_checkpoint = turn_skills_checkpoint
    self._turn_skill_approvals_checkpoint = turn_skill_approvals_checkpoint
    self._turn_todos_checkpoint = turn_todos_checkpoint
    self._turn_message_seq = None
    self._turn_cancel_requested = False
    self._turn_committed = None
    self._turn_clock = TurnClock()
    self._turn_stats_lease = lease
    self._turn_stats_committed = False
    self._turn_stats_seq = None
    self.usage_meter.open_tally()
    parent_context = current_span.get()
    turn_context = new_context(
        turn_id=str(uuid4()),
        session_id=self._session_id,
        trial_id=parent_context.trial_id if parent_context else None,
        execution_id=parent_context.execution_id if parent_context else None,
    )
    if parent_context is not None:
        turn_context = turn_context.model_copy(
            update={
                "links": (
                    TraceLink(
                        trace_id=parent_context.trace_id,
                        span_id=parent_context.span_id,
                        relation="trial",
                    ),
                )
            }
        )
    turn_scope = self.recorder.span(
        "agent.turn",
        context=turn_context,
        attributes={
            "input_origin": "monitor"
            if isinstance(user_input, BackgroundInput)
            else "user"
        },
    )
    self._turn_trace = turn_scope
    self._trace_status = "error"
    self._turn_captures = CommittedContentBuffer(
        self.recorder.artifact_max_bytes, self.recorder.secrets
    )
    self._turn_iterations = 0
    self.execution_outcome = None

    turn = self._run_turn(user_input, lease)
    try:
        # Closing the public stream also closes the implementation stream,
        # so abandoned frontends release any nested model/tool generators.
        async with aclosing(turn):
            while True:
                # An async generator may be advanced by a different task on
                # each ``anext`` call. Track the task driving this specific
                # step so cancel/finalize cannot target a completed former
                # consumer while another task is inside the turn.
                if self._turn_lease is not lease:
                    break
                self._active_turn_task = asyncio.current_task()
                try:
                    with bind(turn_scope.context):
                        event = await anext(turn)
                except StopAsyncIteration:
                    break
                # Capture terminal stats before yielding: the conversation
                # lease is already released, so a consumer can start a new
                # turn while handling the final usage events.
                stats = (
                    self._finish_turn_stats(lease)
                    if isinstance(event, Final | Error)
                    else None
                )
                # Usage precedes the event that followed its request, so
                # it always arrives before the turn's terminal event; the
                # turn's totals come last, immediately before its outcome.
                for usage in self.usage_meter.drain():
                    yield UsageReported(usage)
                if stats is not None:
                    yield TurnStatsReported(stats)
                yield event
    except BudgetExceeded as exc:
        self.execution_outcome = ExecutionOutcome(
            status="failed"
            if exc.limit == "usage_unavailable"
            else "timed_out"
            if exc.limit == "wall_seconds"
            else "budget_exceeded",
            cause="provider" if exc.limit == "usage_unavailable" else "agent",
            limit_type=None if exc.limit == "usage_unavailable" else exc.limit,
            provider_error_category="usage_unavailable"
            if exc.limit == "usage_unavailable"
            else None,
        )
        repaired = True
        try:
            self._rollback_turn(
                lease,
                "failed",
                trace_status="cancelled"
                if exc.limit == "wall_seconds"
                else "budget_exceeded",
            )
        except Exception:
            repaired = False
        stats = self._finish_turn_stats(lease, persist=repaired)
        for usage in self.usage_meter.drain():
            yield UsageReported(usage)
        if stats is not None:
            yield TurnStatsReported(stats)
        yield Error(str(exc))
    except asyncio.CancelledError:
        # Cancellation alone deliberately retains the checkpoint for the
        # frontend's cancel -> await -> finalize handshake. Mark the
        # generator stopped before propagating so a same-task driver (the
        # CLI under asyncio.run) can finalize from its cancellation handler.
        if self._turn_lease is lease:
            self._turn_cancel_requested = True
            self._active_turn_task = None
            if self._turn_committed is None:
                self.execution_outcome = ExecutionOutcome(
                    status="cancelled", cause="agent"
                )
        raise
    except GeneratorExit:
        owner = self._active_turn_task
        cancelled_owner = owner is not None and owner.cancelled()
        if self._turn_lease is lease and (
            self._turn_cancel_requested or cancelled_owner
        ):
            # Cancellation can land while the consumer awaits frontend
            # work in the async-for body, outside Agent.run. Async-generator
            # finalization must preserve the checkpoint for the documented
            # explicit finalize handshake instead of racing it.
            self._turn_cancel_requested = True
            self._active_turn_task = None
        else:
            # Unsolicited ``aclose()`` cannot yield an Error event, but it
            # must repair the abandoned turn. Durable cleanup failures are
            # pending and retried before a later turn.
            try:
                self._rollback_turn(lease, "abandoned")
            except Exception:
                self._finish_turn_stats(lease, persist=False)
            else:
                self._finish_turn_stats(lease)
        raise
    except Exception as exc:
        self.execution_outcome = ExecutionOutcome(status="failed", cause="runner")
        notices = self._hooks.drain_notices()
        message = f"agent turn failed: {type(exc).__name__}: {exc}"
        rolled_back = True
        try:
            self._rollback_turn(lease, "failed")
        except Exception as rollback_exc:
            rolled_back = False
            message += (
                f"; rollback failed: {type(rollback_exc).__name__}: {rollback_exc}"
            )
        stats = self._finish_turn_stats(lease, persist=rolled_back)
        for usage in self.usage_meter.drain():
            yield UsageReported(usage)
        for notice in notices:
            yield notice
        if stats is not None:
            yield TurnStatsReported(stats)
        yield Error(message)
    except BaseException:
        # Do not turn process-level exits into ordinary agent events, but
        # do release the lease before allowing them to propagate.
        self._rollback_turn(lease, "abandoned")
        raise
