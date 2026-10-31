"""Model iteration and committed tool-result handling."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from lingcore.background import BackgroundInput
from lingcore.composer import ComposeContext
from lingcore.errors import (
    INCOMPLETE_FINISH_REASONS,
    LLMStreamError,
    incomplete_response_message,
)
from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    SkillActivated,
    StreamRetry,
    TextDelta,
    TodoUpdated,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.execution import ExecutionOutcome
from lingcore.ingest import ingest_attachments
from lingcore.message import Message, ToolCall, UserInput
from lingcore.plugins import (
    AttachmentView,
    TurnEndEvent,
    UserMessageEvent,
)
from lingcore.tracing.requests import closing_stream

if TYPE_CHECKING:
    from lingcore.agent import Agent


async def _run_turn(
    self: Agent, user_input: str | UserInput, lease: object
) -> AsyncGenerator[AgentEvent, None]:
    """Run the acquired turn; :meth:`Agent.run` owns failure containment."""
    from lingcore.agent import _backoff_seconds, _cap_hoist_media

    await self._hooks.start()
    if isinstance(user_input, BackgroundInput) and self.execution_budget is not None:
        self.execution_budget.reserve_automatic_turn()
    for notice in self._hooks.drain_notices():
        yield notice
    if isinstance(user_input, str):
        incoming = UserInput(text=user_input)
    else:
        incoming = user_input
    text = await self.guardrail.pre_input(incoming.text)
    # A slash-command expansion stores the command the user typed. That is
    # user-authored input too, so it passes the same guardrail before it
    # can reach SQLite, history or Edit.
    input_text = (
        text
        if incoming.display_text is None
        else await self.guardrail.pre_input(incoming.display_text)
    )
    attachments = incoming.attachments
    if attachments:
        # Copy every attachment into <workspace>/attachments/ and announce
        # where each landed, so workspace-confined tools can reach them.
        # Runs before prepare(): ingest sets text/binary fallbacks, the
        # adapter sets image/PDF ones (it skips text/binary). Blocking I/O
        # goes to a thread so the event loop stays free.
        attachments, notes = await asyncio.to_thread(
            ingest_attachments, attachments, self.tool_ctx.workspace
        )
        if notes:
            note_block = "\n".join(notes)
            text = f"{text}\n{note_block}" if text else note_block
        if self.media_adapter is not None:
            # Compute text fallbacks once, before the message is committed —
            # stream retries re-render but never re-pay a conversion.
            attachments = await self.media_adapter.prepare(attachments)
    # A hook may ask the user here; that wait is not generation time.
    with self._turn_waiting():
        decision = await self._hooks.user_message(
            UserMessageEvent(
                text=text,
                input_text=input_text,
                attachments=tuple(
                    AttachmentView.from_attachment(a) for a in attachments
                ),
            )
        )
    for notice in self._hooks.drain_notices():
        yield notice
    if decision is not None:
        if decision.action == "block":
            self.execution_outcome = ExecutionOutcome(status="failed", cause="policy")
            self._trace_status = "denied"
            self._clear_turn_state(lease)
            yield Error(f"input blocked by plugin {decision.plugin}: {decision.reason}")
            return
        if decision.context:
            text = f"{text}\n{decision.context}" if text else decision.context
    background = isinstance(incoming, BackgroundInput)
    user_message = Message.user(
        text,
        attachments=attachments,
        # Monitor output is not user-authored; keep it out of input_text.
        input_text=None if background else input_text,
    )
    if background:
        user_message.name = "monitor"
    self.memory.add(user_message)
    self._capture_committed(
        "message", {"role": "user", "content": user_message.content}
    )
    self._turn_user_message = user_message
    self._turn_stats_committed = True
    if self._session_store is not None and self._session_id is not None:
        # SessionMemory captures the sequence returned by the same SQLite
        # transaction that appended this exact row. A pre-append MAX(seq)
        # estimate could retain/delete the wrong row if another process
        # wrote to the session between those operations.
        message_seq = getattr(self.memory, "last_sequence", None)
        if isinstance(message_seq, int) and not isinstance(message_seq, bool):
            self._turn_message_seq = message_seq
            self._turn_stats_seq = message_seq
        else:  # guarded at construction; fail closed if a wrapper misbehaves
            raise RuntimeError(
                "session-backed memory did not report the committed user sequence"
            )

    compacted_this_turn = False
    for _ in range(self.max_iters):
        active = tuple(self.skill_state.active) if self.skill_state else ()
        # Re-compose every iteration so skill activation and memory writes
        # take effect on the next model request, not the next user turn.
        compose_ctx = ComposeContext(
            user_message=text,
            turn_index=self._turn_index,
            session_id=self._session_id,
            active_skills=active,
        )
        with self.recorder.span("prompt.compose") as composition:
            system_prompt = await self.composer.compose(compose_ctx)
            composition.attributes.update(
                characters=len(system_prompt),
                layers=len(getattr(self.composer, "layers", ())),
            )

        # The live todo list is carried by tool results, which compaction
        # summarizes away and eviction drops. Memory re-pins this note in
        # front of whatever survives either, so the model never loses the
        # list while TodoState still holds it. Refreshed every iteration
        # because the previous tool batch may have changed it.
        set_pinned = getattr(self.memory, "set_pinned_note", None)
        if callable(set_pinned) and self.todo_state is not None:
            set_pinned(self.todo_state.pinned_note())

        # Turn-boundary compaction: once per turn, before the first request,
        # so that request already sees the compacted context. Done here (not
        # mid tool-loop) so within-turn iterations stay append-only and
        # cache-stable. The system prompt is passed so the trigger measures
        # the same footprint the window does. Plain window memory is a
        # no-op; session snapshot failures are contained by run().
        if not compacted_this_turn:
            compacted_this_turn = True
            set_turn_index = getattr(self.memory, "set_compaction_turn_index", None)
            if callable(set_turn_index):
                set_turn_index(self._turn_index)
            with self.recorder.span("memory.compact") as compaction:
                compaction.attributes["input_messages"] = len(self.memory.messages)
                compacted = await self.memory.maybe_compact(system_prompt)
                compaction.attributes["output_messages"] = len(self.memory.messages)
            if compacted is not None:
                yield compacted

        self._turn_index += 1
        self._turn_iterations += 1

        messages = self.memory.render(system_prompt)
        schemas = self._effective_tool_schemas()
        self._capture_committed("prompt", {"system": system_prompt, "tools": schemas})

        # --- Stream one assistant turn, recovering from mid-stream loss ---
        # A turn that fails in flight was never committed to memory, so it
        # can be re-requested cleanly: discard the partial accumulation,
        # tell the frontend (StreamRetry voids any text it already
        # rendered), back off, and ask again with the same request. An
        # LLM failure never crashes the loop (invariant 5): terminal
        # failures end the turn with an Error event instead of raising.
        attempt = 0
        while True:
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            finish_reason: str | None = None
            finish_detail: str | None = None
            dropped_tool_calls = 0
            anthropic_content: list[dict[str, Any]] | None = None
            responses_content: list[dict[str, Any]] | None = None
            try:
                async with closing_stream(
                    self.llm.stream(messages, tools=schemas)
                ) as stream:
                    async for chunk in stream:
                        if chunk.reasoning_delta:
                            reasoning_parts.append(chunk.reasoning_delta)
                        if chunk.text_delta:
                            content_parts.append(chunk.text_delta)
                            yield TextDelta(chunk.text_delta)
                        if chunk.tool_calls:
                            tool_calls = chunk.tool_calls
                        if chunk.finish_reason:
                            finish_reason = chunk.finish_reason
                        if (
                            snapshot := getattr(chunk, "anthropic_content", None)
                        ) is not None:
                            anthropic_content = snapshot
                        if (
                            snapshot := getattr(chunk, "responses_content", None)
                        ) is not None:
                            responses_content = snapshot
                break
            except LLMStreamError as e:
                if e.finish_reason in INCOMPLETE_FINISH_REASONS:
                    # The provider hit a limit, rather than losing the
                    # connection. Preserve partial text and end explicitly;
                    # retrying the same request/budget cannot fix this.
                    finish_reason = e.finish_reason
                    finish_detail = str(e)
                    dropped_tool_calls = e.dropped_tool_calls
                    break
                attempt += 1
                if not e.retryable or attempt > self.stream_retries:
                    spent = (
                        f" after {self.stream_retries} retries"
                        if e.retryable and self.stream_retries
                        else ""
                    )
                    # This request produced no committed assistant message;
                    # keep the runtime counter aligned with durable replay.
                    self._turn_index -= 1
                    message = f"model request failed{spent}: {e}"
                    self.execution_outcome = ExecutionOutcome(
                        status="failed",
                        cause="provider",
                        provider_error_category=e.category,
                    )
                    await self._turn_end(
                        TurnEndEvent("", error=message, turn_index=self._turn_index),
                        lease,
                        Error(message),
                    )
                    for notice in self._hooks.drain_notices():
                        yield notice
                    self._clear_turn_state(lease)
                    yield Error(message)
                    return
                yield StreamRetry(
                    attempt=attempt,
                    max_attempts=self.stream_retries,
                    reason=str(e),
                    discarded_chars=sum(len(p) for p in content_parts),
                )
                previous_request = self.request_ledger.last_request_id
                self.request_ledger.retry_request_id = previous_request
                delay = _backoff_seconds(attempt)
                with (
                    self._turn_waiting(),
                    self.recorder.span(
                        "retry.backoff",
                        attributes={
                            "attempt": attempt,
                            "delay_seconds": delay,
                            "previous_request_id": previous_request,
                        },
                    ),
                ):
                    await asyncio.sleep(delay)
            except Exception as e:
                # A duck-typed backend may raise anything; without a
                # retryable classification, surface it and end the turn.
                self._turn_index -= 1
                message = f"model request failed: {type(e).__name__}: {e}"
                self.execution_outcome = ExecutionOutcome(
                    status="failed",
                    cause="provider",
                    provider_error_category="unclassified",
                )
                await self._turn_end(
                    TurnEndEvent("", error=message, turn_index=self._turn_index),
                    lease,
                    Error(message),
                )
                for notice in self._hooks.drain_notices():
                    yield notice
                self._clear_turn_state(lease)
                yield Error(message)
                return

        incomplete = finish_reason in INCOMPLETE_FINISH_REASONS
        content = "".join(content_parts)
        if incomplete:
            assert finish_reason is not None
            # Guard and persist the authoritative partial reply before the
            # terminal outcome commits, including the Stop/hook path.
            content = await self.guardrail.post_output(content)
            finish_detail = incomplete_response_message(
                finish_reason,
                detail=finish_detail,
                dropped_tool_calls=max(dropped_tool_calls, len(tool_calls)),
            )
        assistant = Message.assistant(
            content=content,
            # No call in a cut-off batch may execute, even one whose JSON
            # happened to finish before the rest of the response was lost.
            tool_calls=[] if incomplete else tool_calls,
            reasoning_content="".join(reasoning_parts) or None,
            anthropic_content=None if incomplete else anthropic_content,
            responses_content=None if incomplete else responses_content,
            finish_reason=finish_reason,
            finish_detail=finish_detail,
        )
        self.memory.add(assistant)
        self._capture_committed(
            "message",
            {
                "role": "assistant",
                "request_id": self.request_ledger.last_request_id,
                "finish_reason": finish_reason,
                "content": assistant.content,
                "tool_calls": [
                    c.model_dump(exclude={"invalid_arguments"})
                    for c in assistant.tool_calls
                ],
            },
        )

        if incomplete:
            assert finish_detail is not None
            message = finish_detail
            terminal = Error(
                message, finish_reason=finish_reason, partial_content=content
            )
            await self._turn_end(
                TurnEndEvent("", error=message, turn_index=self._turn_index),
                lease,
                terminal,
            )
            for notice in self._hooks.drain_notices():
                yield notice
            self._clear_turn_state(lease)
            yield terminal
            return

        if not assistant.tool_calls:
            final = await self.guardrail.post_output(assistant.content)
            await self._turn_end(
                TurnEndEvent(final, turn_index=self._turn_index),
                lease,
                Final(final),
            )
            for notice in self._hooks.drain_notices():
                yield notice
            self._clear_turn_state(lease)
            yield Final(final)
            return

        for call in assistant.tool_calls:
            yield ToolCallStarted(call)

        before_active = list(self.skill_state.active) if self.skill_state else []
        before = set(before_active)
        before_approvals = (
            dict(self.skill_state.approved_high_risk) if self.skill_state else {}
        )
        before_todos = self.todo_state.items if self.todo_state else None
        # Tool runs (and the approvals inside them) are the turn waiting
        # on the world, not generating.
        with self._turn_waiting():
            results = await self._dispatch(assistant.tool_calls)
        for notice in self._hooks.drain_notices():
            yield notice
        for result in results:
            self.memory.add(Message.from_tool_result(result))
            self._capture_committed(
                "message",
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "name": result.name,
                    "content": result.content
                    if result.ok
                    else "[tool error content excluded]",
                    "ok": result.ok,
                },
            )
            yield ToolResultEvent(result)
        media = [attachment for result in results for attachment in result.attachments]
        if media:
            kept, dropped = _cap_hoist_media(media)
            if kept and self.media_adapter is not None:
                # After the cap, so dropped attachments never cost a
                # conversion (PDF extraction / a vision describe call).
                kept = await self.media_adapter.prepare(kept)
            names = ", ".join(a.name or a.media_type for a in kept[:3])
            if len(kept) > 3:
                names += f", +{len(kept) - 3} more"
            note = f"[media from tool results: {names}]"
            if dropped:
                note += (
                    f" [{len(dropped)} attachment(s) dropped over the "
                    "per-message media limit; re-run the tool for the ones "
                    "you need, fewer at a time]"
                )
            self.memory.add(
                Message(
                    role="user",
                    content=note,
                    name="media",
                    attachments=kept,
                )
            )

        # Surface skill activation/deactivation changes as events.
        if self.skill_state is not None:
            after_active = list(self.skill_state.active)
            after = set(self.skill_state.active)
            activated = sorted(after - before)
            deactivated = sorted(before - after)
            state_changed = (
                after_active != before_active
                or self.skill_state.approved_high_risk != before_approvals
            )
            if (
                state_changed
                and self._session_store is not None
                and self._session_id is not None
            ):
                self._session_store.save_skill_state(
                    self._session_id,
                    lease=self._session_lease,
                    active=list(self.skill_state.active),
                    activated=activated,
                    deactivated=deactivated,
                    approved_high_risk={
                        name: sorted(
                            self.skill_state.approved_high_risk.get(name, frozenset())
                        )
                        for name in self.skill_state.active
                    },
                )
            for name in activated:
                yield SkillActivated(name=name, active=True)
            for name in deactivated:
                yield SkillActivated(name=name, active=False)

        if self.todo_state is not None and self.todo_state.items != before_todos:
            if self._session_store is not None and self._session_id is not None:
                self._session_store.save_todo_state(
                    self._session_id,
                    self.todo_state.items,
                    lease=self._session_lease,
                )
            yield TodoUpdated(todos=self.todo_state.items)

    message = f"reached max iterations ({self.max_iters}) without a final reply"
    await self._turn_end(
        TurnEndEvent("", error=message, turn_index=self._turn_index),
        lease,
        Error(message),
    )
    for notice in self._hooks.drain_notices():
        yield notice
    self._clear_turn_state(lease)
    yield Error(message)
