"""The agent runtime loop — the core of LingCore.

``Agent.run`` is the whole point of "swap config, not code": every agent type
(coding, role-play, teaching, psych) runs this same loop, differing only in
the profile that built it. The loop:

    assemble context -> stream a turn -> if tool calls: run them (in parallel)
    and feed results back -> repeat -> on a tool-free turn, post-guard + Final.

The loop talks only to an ``LLMClient``-shaped object (duck-typed ``stream``),
never to the OpenAI SDK directly.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal, Protocol

from lingcore._agent import dispatch, iteration, lifecycle, run
from lingcore._agent import profile as profile_builder
from lingcore.background import BackgroundInbox
from lingcore.composer import ComposeContext, PromptComposer, StaticComposer
from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    PluginNotice,
    TurnCancelled,
    UsageReported,
)
from lingcore.execution import ExecutionBudget, ExecutionOutcome
from lingcore.guardrails import Guardrail, NoopGuardrail
from lingcore.guardrails import build_guardrail as build_guardrail
from lingcore.llm import LLMChunk
from lingcore.media_types import (
    MAX_ATTACHMENTS,
    TOTAL_ATTACHMENT_MAX_BYTES,
    decoded_payload_size,
)
from lingcore.memory import ShortTermMemory, WindowMemory
from lingcore.message import Attachment, Message, ToolCall, ToolResult, UserInput
from lingcore.plugins import (
    TurnEndEvent,
)
from lingcore.plugins.commands import CommandCatalog
from lingcore.plugins.hooks import HookRunner
from lingcore.session_leases import SessionLease
from lingcore.todos import TodoItem, TodoState
from lingcore.tools import Tool, ToolContext, ToolRegistry
from lingcore.tracing import NoopRecorder, SpanScope, TraceRecorder
from lingcore.tracing.content import CommittedContentBuffer
from lingcore.tracing.requests import InstrumentedClient, RequestLedger
from lingcore.usage import TurnClock, TurnStats, UsageMeter

if TYPE_CHECKING:
    from pathlib import Path

    from lingcore.config import AgentProfile
    from lingcore.modality import MediaAdapter
    from lingcore.output_rewrite import OutputRewriter
    from lingcore.sessions import SessionStore
    from lingcore.skills import SkillState
    from lingcore.tools import ConfirmFn, EnvironmentBinding


# Backoff between stream re-requests (mid-stream failure recovery): capped
# exponential with full jitter. Stream-open backoff is the SDK's job; this
# only paces the loop's own retries, so it stays simple and unconditional.
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0


def _backoff_seconds(attempt: int) -> float:
    """Delay before stream-retry ``attempt`` (1-based)."""
    return random.uniform(
        0.0, min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
    )


def _cap_hoist_media(
    media: list[Attachment],
) -> tuple[list[Attachment], list[Attachment]]:
    """Split a round's tool media into (kept, dropped) under the Message caps.

    Each ToolResult enforces the per-list limits on its own attachments, but
    the hoist aggregates *all* results of the round into one Message — whose
    validator re-applies those limits to the aggregate. Capping here keeps a
    many-tool round from raising out of the loop (invariant 5); the dropped
    tail is reported in the hoist text so the model can re-request what it
    actually needs one file at a time.
    """
    kept: list[Attachment] = []
    dropped: list[Attachment] = []
    total = 0
    for attachment in media:
        try:
            size = decoded_payload_size(attachment.data)
        except Exception:
            dropped.append(attachment)
            continue
        if len(kept) >= MAX_ATTACHMENTS or total + size > TOTAL_ATTACHMENT_MAX_BYTES:
            dropped.append(attachment)
            continue
        kept.append(attachment)
        total += size
    return kept, dropped


class _LLMLike(Protocol):
    def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]: ...


class Agent:
    def __init__(
        self,
        llm: _LLMLike,
        tools: ToolRegistry,
        tool_ctx: ToolContext,
        composer: PromptComposer | str | None = None,
        memory: ShortTermMemory | None = None,
        guardrail: Guardrail | None = None,
        max_iters: int = 25,
        parallel_tools: bool = True,
        stream_retries: int = 3,
        session_id: str | None = None,
        skill_state: "SkillState | None" = None,
        media_adapter: "MediaAdapter | None" = None,
        initial_tools: "frozenset[str] | None" = None,
        session_store: "SessionStore | None" = None,
        system_prompt: str | None = None,
        usage_meter: UsageMeter | None = None,
        todo_state: TodoState | None = None,
        hooks: HookRunner | None = None,
        commands: CommandCatalog | None = None,
        recorder: TraceRecorder | None = None,
        execution_budget: ExecutionBudget | None = None,
        request_ledger: RequestLedger | None = None,
        output_rewriter: OutputRewriter | None = None,
    ) -> None:
        # v0.1 accepted a static prompt as the fourth positional argument or
        # as ``system_prompt=``. Keep both forms while routing new code through
        # the composer seam introduced in v0.2.
        if system_prompt is not None:
            if composer is not None:
                raise ValueError("pass either composer or system_prompt, not both")
            composer = system_prompt
        if isinstance(composer, str):
            composer = StaticComposer(composer)
        if composer is None:
            raise TypeError("Agent requires a composer or system_prompt")
        self.recorder = recorder if recorder is not None else NoopRecorder()
        self.execution_budget = execution_budget
        self.execution_outcome: ExecutionOutcome | None = None
        self.request_ledger = request_ledger or (
            llm.ledger if isinstance(llm, InstrumentedClient) else RequestLedger()
        )
        self.llm: _LLMLike = (
            llm
            if isinstance(llm, InstrumentedClient)
            or (not self.recorder.enabled and execution_budget is None)
            else InstrumentedClient(
                llm,
                self.recorder,
                self.request_ledger,
                model=getattr(llm, "model", "unknown"),
                provider="injected",
                budget=execution_budget,
            )
        )
        self.tools = tools
        # Tools active without any skill. None ⇒ all of the ceiling (today's
        # default). A narrower set gates the rest behind skill activation; the
        # ceiling (``tools``) still bounds what a skill can ever unlock.
        self.initial_tools: frozenset[str] = (
            frozenset(initial_tools)
            if initial_tools is not None
            else frozenset(tools.names())
        )
        self.tool_ctx = tool_ctx
        self.tool_ctx.recorder = self.recorder
        if tool_ctx.confirm is not None:
            original_confirm = tool_ctx.confirm

            async def traced_confirm(prompt: str) -> bool:
                with self.recorder.span(
                    "approval.wait", attributes={"policy": "frontend"}
                ) as approval:
                    allowed = await original_confirm(prompt)
                    approval.attributes["decision"] = "allow" if allowed else "deny"
                    if not allowed:
                        approval.close("denied")
                    return allowed

            tool_ctx.confirm = traced_confirm
        self._hooks = hooks or HookRunner([], tool_ctx)
        self.commands = commands if commands is not None else CommandCatalog()
        self._closed = False
        self.composer = composer
        self.memory: ShortTermMemory = memory or WindowMemory()
        self.guardrail = guardrail or NoopGuardrail()
        self.max_iters = max_iters
        self.parallel_tools = parallel_tools
        self.stream_retries = max(0, stream_retries)
        self._session_id = session_id
        # Shared skill runtime state (None when skills are not configured).
        self.skill_state = skill_state
        # Shared todo checklist (None unless the todo_write tool is enabled).
        self.todo_state = todo_state
        # Text-fallback adapter for attachment kinds the model lacks
        # (None — the all-native default — costs nothing).
        self.media_adapter = media_adapter
        self.output_rewriter = output_rewriter
        # Provider usage reported by this agent's clients; drained into
        # ``UsageReported`` events by ``run``. An injected client may record
        # into ``agent.usage_meter`` itself.
        self.usage_meter = usage_meter or UsageMeter()
        self._session_store = session_store
        self._session_lease: SessionLease | None = getattr(self.memory, "lease", None)
        self._owned_session_store: SessionStore | None = None
        if session_store is not None and (
            session_id is None
            or not hasattr(self.memory, "last_sequence")
            or not isinstance(self._session_lease, SessionLease)
            or self._session_lease.session_id != session_id
        ):
            raise ValueError(
                "session_store requires a session id and leased, sequence-reporting memory"
            )
        self._turn_index: int = 0
        # Explicit frontend cancellation is a two-step contract: cancel the
        # task driving ``run()``, then call ``finalize_cancelled_turn()`` after
        # that task has stopped. The checkpoint keeps the working conversation
        # valid even if cancellation interrupted a tool-call block.
        self._active_turn_task: asyncio.Task[Any] | None = None
        # Identity token for the generator that owns the mutable checkpoint.
        # A terminal event releases the public turn slot before its generator
        # is necessarily exhausted; a later ``aclose()`` from that stale
        # generator must never roll back a newer owner.
        self._turn_lease: object | None = None
        self._turn_checkpoint: list[Message] | None = None
        self._turn_user_message: Message | None = None
        self._turn_index_checkpoint: int | None = None
        self._turn_skills_checkpoint: list[str] | None = None
        self._turn_skill_approvals_checkpoint: dict[str, frozenset[str]] | None = None
        self._turn_todos_checkpoint: tuple[TodoItem, ...] | None = None
        self._turn_message_seq: int | None = None
        self._turn_cancel_requested = False
        # time.monotonic() when the current turn acquired its lease; plugins
        # use it to tell a rolled-back turn's side effects from earlier ones.
        self._turn_started_at = 0.0
        # Set when Stop lands during observe-only turn_end hooks: the outcome is
        # already committed, so finalization returns it instead of rolling back.
        self._turn_committed: Final | Error | None = None
        # If durable rollback fails after the in-memory lease is released, no
        # later turn may append behind that uncommitted branch. Retried at the
        # start of every subsequent run until the store is reconciled.
        self._pending_turn_message_seq: int | None = None
        # The turn's TurnStats in the making. Unlike the checkpoint fields these
        # outlive _clear_turn_state, which runs before the terminal event that
        # reports them; _finish_turn_stats consumes them exactly once.
        self._turn_clock: TurnClock | None = None
        self._turn_stats_lease: object | None = None
        self._turn_stats_committed = False  # the user message reached memory
        self._turn_stats_seq: int | None = None  # ... and this durable row
        self._turn_trace: SpanScope | None = None
        self._trace_status: Literal[
            "ok", "error", "cancelled", "denied", "incomplete", "budget_exceeded"
        ] = "error"
        self._turn_captures = CommittedContentBuffer(
            self.recorder.artifact_max_bytes, self.recorder.secrets
        )
        self._turn_iterations = 0

    # ------------------------------------------------------------------
    # Convenience accessor kept for tests that still read .system_prompt
    # ------------------------------------------------------------------
    @property
    def system_prompt(self) -> str:
        """Return the static text for simple (StaticComposer) agents."""
        if isinstance(self.composer, StaticComposer):
            return self.composer.text
        # LayeredComposer: no single static string; return empty sentinel.
        return ""

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Back-compatible mutation of the v0.1 static prompt attribute."""
        self.composer = StaticComposer(value)

    async def initial_system_prompt(self) -> str:
        """Compose the turn-0 prompt without starting hooks or a model request."""
        return await self.composer.compose(
            ComposeContext(
                user_message="",
                turn_index=0,
                session_id=None,
                active_skills=tuple(self.skill_state.active)
                if self.skill_state
                else (),
            )
        )

    def ceiling_tool_schemas(self) -> list[dict[str, Any]]:
        """Read-only schemas for every authorized binding, including gated tools."""
        return self.tools.schemas()

    @property
    def turn_pending_finalization(self) -> bool:
        """Whether a cancelled checkpoint is ready for frontend finalization.

        This intentionally exposes no mutable checkpoint state. A frontend may
        finalize only after cancellation was requested and the task driving the
        generator has stopped.
        """
        task = self._active_turn_task
        driver_stopped = task is None or task.done()
        return (
            self._turn_checkpoint is not None
            and self._turn_cancel_requested
            and driver_stopped
        )

    @property
    def background(self) -> BackgroundInbox:
        return self.tool_ctx.background

    @classmethod
    def from_profile(
        cls,
        profile: "AgentProfile",
        *,
        confirm: "ConfirmFn | None" = None,
        llm: _LLMLike | None = None,
        vision_llm: _LLMLike | None = None,
        rewrite_llm: _LLMLike | None = None,
        base_dir: "Path | None" = None,
        tool_options: "dict | None" = None,
        session_store: "SessionStore | None" = None,
        session_id: str | None = None,
        recorder: TraceRecorder | None = None,
        state_dir: "Path | None" = None,
        tool_registry: ToolRegistry | None = None,
        execution_budget: ExecutionBudget | None = None,
        environment_binding: EnvironmentBinding | None = None,
    ) -> "Agent":
        """Assemble a runnable Agent from a validated profile.

        ``session_store``/``session_id`` opt into persistent history: the
        memory is hydrated from the store (resuming ``session_id`` when given,
        else starting a fresh session) and every subsequent message is
        recorded. Opening the store is the composition root's job — see
        ``lingcore.sessions.open_store``. ``vision_llm`` overrides the
        ``media_fallback.image`` client (tests inject a fake here, exactly
        like ``llm``). ``rewrite_llm`` similarly overrides the experimental
        ``output_rewrite.llm`` client when rewriting is enabled.
        """
        return profile_builder.from_profile(
            cls,
            profile,
            confirm=confirm,
            llm=llm,
            vision_llm=vision_llm,
            rewrite_llm=rewrite_llm,
            base_dir=base_dir,
            tool_options=tool_options,
            session_store=session_store,
            session_id=session_id,
            recorder=recorder,
            state_dir=state_dir,
            tool_registry=tool_registry,
            execution_budget=execution_budget,
            environment_binding=environment_binding,
        )

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close this Agent's plugin instances once, in reverse enablement order.

        Frontends stop and finalize an active turn before closing its Agent.
        Closing never invokes turn_end; Stop remains a synchronous rollback.
        """
        return await lifecycle.aclose(self)

    def drain_usage(self) -> list[UsageReported]:
        """Return usage recorded but not yet emitted by ``run``.

        A frontend that stops a turn calls this after
        ``finalize_cancelled_turn()``: requests that completed before the
        cancellation landed were billed even though their events never
        reached the stream.
        """
        return [
            UsageReported(usage, purpose)
            for usage, purpose in self.usage_meter.drain_with_purpose()
        ]

    def drain_plugin_notices(self) -> list[PluginNotice]:
        """Take pending transient notices before finalizing a cancelled turn.

        Frontends render these before TurnCancelled, alongside drained usage.
        Rollback discards anything not drained so abandoned streams cannot leak
        a prior turn's notices into a later conversation.
        """
        return self._hooks.drain_notices()

    def cancel_turn(self) -> bool:
        """Request cancellation of the task currently driving ``run``.

        Returns ``False`` when no turn is active. The caller must await the
        driver task and then call :meth:`finalize_cancelled_turn` to repair the
        working set and obtain the lifecycle event to render.
        """
        return lifecycle.cancel_turn(self)

    def finalize_cancelled_turn(
        self, reason: str = "stopped by user"
    ) -> TurnCancelled | Error | Final:
        """Restore the pre-turn state while retaining the submitted user input.

        A durable-store cleanup failure is returned as an :class:`Error` and
        retried before the next turn; it never leaves the in-process lease held.
        When Stop landed only during observe-only ``turn_end`` hooks, the turn's
        outcome was already committed: nothing is rolled back and the committed
        :class:`Final` (or :class:`Error`) is returned for the frontend to render.
        """
        return lifecycle.finalize_cancelled_turn(self, reason)

    def _rollback_turn(
        self,
        lease: object | None = None,
        reason: Literal["cancelled", "failed", "abandoned"] = "cancelled",
        trace_status: Literal["error", "cancelled", "budget_exceeded"] | None = None,
    ) -> None:
        """Restore the turn checkpoint and always release its lease.

        Plugins hear ``turn_rolled_back`` even when the durable truncate fails:
        the conversation the model sees no longer contains the turn either way.
        """
        return lifecycle._rollback_turn(self, lease, reason, trace_status)

    def _turn_waiting(self) -> AbstractContextManager[None]:
        """Exclude the enclosed span from the turn's generation time."""
        return lifecycle._turn_waiting(self)

    def _finish_turn_stats(
        self, lease: object | None = None, *, persist: bool = True
    ) -> TurnStats | None:
        """Close the turn's clock and usage tally; persist and return its stats.

        Only a turn whose user message was committed has stats: blocked input
        never became a turn. They are anchored to that user row after any
        rollback has run, so a stopped or failed turn's billed requests stay
        on record. Persisting is display data, so a store failure is ignored.
        """
        return lifecycle._finish_turn_stats(self, lease, persist=persist)

    def _clear_turn_state(self, lease: object | None = None) -> None:
        """Release turn state only when ``lease`` still owns it."""
        return lifecycle._clear_turn_state(self, lease)

    def _retry_pending_turn_cleanup(self) -> None:
        """Reconcile a durable tail whose earlier rollback failed."""
        return lifecycle._retry_pending_turn_cleanup(self)

    def run(self, user_input: str | UserInput) -> AsyncGenerator[AgentEvent, None]:
        """Drive one user turn to completion, yielding events as they happen."""
        return run.run(self, user_input)

    def _run_turn(
        self, user_input: str | UserInput, lease: object
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the acquired turn; :meth:`run` owns failure containment."""
        return iteration._run_turn(self, user_input, lease)

    async def _turn_end(
        self, event: TurnEndEvent, lease: object, terminal: Final | Error
    ) -> None:
        """Run observe-only turn_end hooks after the turn's outcome is committed.

        Stop during a slow observer must not roll back a reply the user has
        already seen streamed. The lease stays held for the usual
        cancel -> await -> finalize handshake, but finalization then returns
        ``terminal`` instead of restoring the checkpoint.
        """
        return await lifecycle._turn_end(self, event, lease, terminal)

    def _capture_committed(self, kind: str, value: Any) -> None:
        """Buffer bounded evidence until the whole turn commits; rollback drops it."""
        return lifecycle._capture_committed(self, kind, value)

    def _effective_tool_schemas(self) -> list[dict[str, Any]]:
        """Tool schemas for the current step: the initially-enabled tools plus
        any tools granted by active skills (intersected with the profile
        allowlist by SkillState).

        Computed fresh each iteration; the base ToolRegistry is never mutated.
        Tools in the ceiling but neither initial nor granted by an active skill
        are NOT advertised — progressive disclosure (invariant 11).
        """
        return dispatch._effective_tool_schemas(self)

    async def _dispatch(self, calls: list[ToolCall]) -> list[ToolResult]:
        # Authorization is a property of the model request that produced this
        # batch. Snapshot it before any tool runs: activate_skill/deactivation in
        # one call must affect only the next loop iteration, never sibling calls
        # whose schemas were computed under the previous state.
        return await dispatch._dispatch(self, calls)

    def _authorized_tool_names(self) -> frozenset[str]:
        """Currently dispatchable tools, bounded by the profile registry."""
        return dispatch._authorized_tool_names(self)

    def _resolve_tool(
        self, name: str, *, authorized: frozenset[str] | None = None
    ) -> Tool:
        """Look up a tool by name, enforcing the permission model at dispatch.

        A tool is dispatchable only when it is currently *authorized*: in the
        initially-enabled set, or granted by an active skill (profile ∩
        requested). A ceiling tool that is neither — advertised or not — is
        refused with the same error an unknown tool would raise, so a model that
        calls a gated-but-not-active tool can't reach it.
        """
        return dispatch._resolve_tool(self, name, authorized=authorized)
