"""Turn checkpoints, cancellation, finalization, and Agent cleanup."""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Literal

from lingcore.events import (
    Error,
    Final,
    TurnCancelled,
)
from lingcore.execution import ExecutionOutcome
from lingcore.plugins import (
    TurnEndEvent,
)
from lingcore.plugins.hooks import TurnRolledBackEvent
from lingcore.usage import TurnStats

if TYPE_CHECKING:
    from lingcore.agent import Agent


async def aclose(self: Agent) -> None:
    """Close this Agent's plugin instances once, in reverse enablement order.

    Frontends stop and finalize an active turn before closing its Agent.
    Closing never invokes turn_end; Stop remains a synchronous rollback.
    """
    self._closed = True
    self.background.close()
    try:
        await self._hooks.aclose()
    finally:
        try:
            await self.recorder.aclose()
        finally:
            try:
                if self._session_store is not None and self._session_lease is not None:
                    self._session_store.release_session(self._session_lease)
            finally:
                owned = getattr(self, "_owned_session_store", None)
                if owned is not None:
                    owned.close()
                    self._owned_session_store = None


def cancel_turn(self: Agent) -> bool:
    """Request cancellation of the task currently driving ``run``.

    Returns ``False`` when no turn is active. The caller must await the
    driver task and then call :meth:`finalize_cancelled_turn` to repair the
    working set and obtain the lifecycle event to render.
    """
    task = self._active_turn_task
    if task is None or task.done():
        return False
    cancelled = task.cancel()
    if cancelled:
        self._turn_cancel_requested = True
    return cancelled


def finalize_cancelled_turn(
    self: Agent, reason: str = "stopped by user"
) -> TurnCancelled | Error | Final:
    """Restore the pre-turn state while retaining the submitted user input.

    A durable-store cleanup failure is returned as an :class:`Error` and
    retried before the next turn; it never leaves the in-process lease held.
    When Stop landed only during observe-only ``turn_end`` hooks, the turn's
    outcome was already committed: nothing is rolled back and the committed
    :class:`Final` (or :class:`Error`) is returned for the frontend to render.
    """
    task = self._active_turn_task
    if task is not None and not task.done():
        raise RuntimeError(
            "cannot finalize a cancelled turn before its driver task stops"
        )
    if self._turn_checkpoint is None or not self._turn_cancel_requested:
        raise RuntimeError("no cancelled turn is ready to finalize")
    committed = self._turn_committed
    if self._turn_trace is not None:
        self.recorder.event(
            "turn.finalize",
            context=self._turn_trace.context,
            attributes={
                "committed": committed is not None,
                "outcome": "committed" if committed is not None else "cancelled",
            },
        )
    if committed is not None:
        self._hooks.drain_notices()
        self._clear_turn_state()
        self._finish_turn_stats()
        return committed
    try:
        self._rollback_turn()
    except Exception as exc:
        # In-memory state and the running lease are already repaired. A
        # failed durable truncate is recorded for retry before the next
        # turn; surface it as an event instead of crashing Stop handling.
        # The retry would delete stats anchored to the turn, so skip them.
        self._finish_turn_stats(persist=False)
        return Error(
            f"turn cancelled, but rollback is pending: {type(exc).__name__}: {exc}"
        )
    self._finish_turn_stats()
    return TurnCancelled(reason=reason)


def _rollback_turn(
    self: Agent,
    lease: object | None = None,
    reason: Literal["cancelled", "failed", "abandoned"] = "cancelled",
    trace_status: Literal["error", "cancelled", "budget_exceeded"] | None = None,
) -> None:
    """Restore the turn checkpoint and always release its lease.

    Plugins hear ``turn_rolled_back`` even when the durable truncate fails:
    the conversation the model sees no longer contains the turn either way.
    """
    if lease is not None and self._turn_lease is not lease:
        return
    try:
        if self._turn_checkpoint is not None:
            restored = list(self._turn_checkpoint)
            if self._turn_user_message is not None:
                restored.append(self._turn_user_message)
            self.memory.replace(restored)
        if self._turn_index_checkpoint is not None:
            self._turn_index = self._turn_index_checkpoint
        if self.skill_state is not None and self._turn_skills_checkpoint is not None:
            self.skill_state.active[:] = self._turn_skills_checkpoint
            self.skill_state.approved_high_risk = dict(
                self._turn_skill_approvals_checkpoint or {}
            )
        if self.todo_state is not None and self._turn_todos_checkpoint is not None:
            self.todo_state.items = self._turn_todos_checkpoint
        if (
            self._session_store is not None
            and self._session_id is not None
            and self._turn_message_seq is not None
        ):
            message_seq = self._turn_message_seq
            try:
                self._session_store.truncate_after(
                    self._session_id, message_seq, lease=self._session_lease
                )
            except BaseException:
                self._pending_turn_message_seq = message_seq
                raise
            else:
                self._pending_turn_message_seq = None
    finally:
        # A failed repair (for example, the same SQLite outage that ended
        # the turn) must not leave the in-process turn lease wedged.
        self._hooks.drain_notices()
        self._hooks.turn_rolled_back(TurnRolledBackEvent(reason, self._turn_started_at))
        self._trace_status = trace_status or (
            "cancelled" if reason != "failed" else "error"
        )
        self._turn_captures.clear()
        if self._turn_trace is not None:
            self.recorder.event(
                "turn.rollback",
                context=self._turn_trace.context,
                attributes={
                    "reason": reason,
                    "session.message_seq": self._turn_message_seq,
                },
            )
        self._clear_turn_state(lease)


def _turn_waiting(self: Agent) -> AbstractContextManager[None]:
    """Exclude the enclosed span from the turn's generation time."""
    clock = self._turn_clock
    return clock.waiting() if clock is not None else nullcontext()


def _finish_turn_stats(
    self: Agent, lease: object | None = None, *, persist: bool = True
) -> TurnStats | None:
    """Close the turn's clock and usage tally; persist and return its stats.

    Only a turn whose user message was committed has stats: blocked input
    never became a turn. They are anchored to that user row after any
    rollback has run, so a stopped or failed turn's billed requests stay
    on record. Persisting is display data, so a store failure is ignored.
    """
    # Stats outlive the conversation lease; a stale generator may close
    # after another turn has started and must not consume its tally.
    if lease is not None and self._turn_stats_lease is not lease:
        return None
    self._turn_stats_lease = None
    clock, self._turn_clock = self._turn_clock, None
    if clock is None:
        return None
    stats = clock.finish(self.usage_meter.close_tally())
    committed, self._turn_stats_committed = self._turn_stats_committed, False
    seq, self._turn_stats_seq = self._turn_stats_seq, None
    if not committed:
        return None
    if (
        persist
        and seq is not None
        and self._session_store is not None
        and self._session_id is not None
    ):
        try:
            self._session_store.save_turn_stats(
                self._session_id, seq, stats, lease=self._session_lease
            )
        except Exception:
            pass
    return stats


def _clear_turn_state(self: Agent, lease: object | None = None) -> None:
    """Release turn state only when ``lease`` still owns it."""
    if lease is not None and self._turn_lease is not lease:
        return
    if self._turn_trace is not None:
        self._turn_trace.close(
            self._trace_status,
            iterations=self._turn_iterations,
            outcome=self._trace_status,
            **{"session.message_seq": self._turn_message_seq},
        )
        self._turn_trace = None
    self._turn_captures.clear()
    self._active_turn_task = None
    self._turn_lease = None
    self._turn_checkpoint = None
    self._turn_user_message = None
    self._turn_index_checkpoint = None
    self._turn_skills_checkpoint = None
    self._turn_skill_approvals_checkpoint = None
    self._turn_todos_checkpoint = None
    self._turn_message_seq = None
    self._turn_cancel_requested = False
    self._turn_committed = None


def _retry_pending_turn_cleanup(self: Agent) -> None:
    """Reconcile a durable tail whose earlier rollback failed."""
    message_seq = self._pending_turn_message_seq
    if message_seq is None:
        return
    if self._session_store is None or self._session_id is None:
        raise RuntimeError("pending turn cleanup has no session store")
    self._session_store.truncate_after(
        self._session_id, message_seq, lease=self._session_lease
    )
    self._pending_turn_message_seq = None


async def _turn_end(
    self: Agent, event: TurnEndEvent, lease: object, terminal: Final | Error
) -> None:
    """Run observe-only turn_end hooks after the turn's outcome is committed.

    Stop during a slow observer must not roll back a reply the user has
    already seen streamed. The lease stays held for the usual
    cancel -> await -> finalize handshake, but finalization then returns
    ``terminal`` instead of restoring the checkpoint.
    """
    self._trace_status = (
        "ok"
        if isinstance(terminal, Final)
        else "incomplete"
        if terminal.finish_reason
        else "error"
    )
    if self.execution_outcome is None:
        self.execution_outcome = ExecutionOutcome(
            status="completed" if isinstance(terminal, Final) else "failed",
            cause=None
            if isinstance(terminal, Final)
            else "provider"
            if terminal.finish_reason
            else "agent",
            provider_error_category=terminal.finish_reason
            if isinstance(terminal, Error)
            else None,
        )
    if self._turn_trace is not None:
        self._capture_committed(
            "guarded_final",
            terminal.content
            if isinstance(terminal, Final)
            else terminal.partial_content,
        )
        self.recorder._capture_sanitized(
            "turn.committed",
            self._turn_captures.data,
            context=self._turn_trace.context,
            truncated=self._turn_captures.truncated,
            omitted_items=self._turn_captures.omitted_items,
        )
    try:
        await self._hooks.turn_end(event)
    except asyncio.CancelledError:
        if self._turn_lease is lease:
            self._turn_committed = terminal
        raise


def _capture_committed(self: Agent, kind: str, value: Any) -> None:
    """Buffer bounded evidence until the whole turn commits; rollback drops it."""
    if self.recorder.content != "redacted" or self._turn_trace is None:
        return
    self._turn_captures.append(kind, value)
