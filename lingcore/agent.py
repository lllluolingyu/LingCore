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
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol

from lingcore.composer import ComposeContext, PromptComposer, StaticComposer
from lingcore.errors import LLMStreamError
from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    SkillActivated,
    StreamRetry,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.guardrails import Guardrail, NoopGuardrail
from lingcore.ingest import ingest_attachments
from lingcore.llm import LLMChunk
from lingcore.media_types import (
    MAX_ATTACHMENTS,
    TOTAL_ATTACHMENT_MAX_BYTES,
    decoded_payload_size,
)
from lingcore.memory import ShortTermMemory, WindowMemory
from lingcore.message import Attachment, Message, ToolCall, ToolResult, UserInput
from lingcore.tools import ToolContext, ToolOutput, ToolRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from lingcore.config import AgentProfile
    from lingcore.modality import MediaAdapter
    from lingcore.sessions import SessionStore
    from lingcore.skills import Skill, SkillState
    from lingcore.tools import ConfirmFn


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


def _build_guardrail(policy: str) -> Guardrail:
    """Map a profile's guardrail policy name to an implementation."""
    if policy == "noop":
        return NoopGuardrail()
    from lingcore.errors import ConfigError

    raise ConfigError(f"unknown guardrail policy: {policy!r}")


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
        composer: PromptComposer,
        memory: ShortTermMemory | None = None,
        guardrail: Guardrail | None = None,
        max_iters: int = 25,
        parallel_tools: bool = True,
        stream_retries: int = 3,
        session_id: str | None = None,
        skill_state: "SkillState | None" = None,
        media_adapter: "MediaAdapter | None" = None,
        initial_tools: "frozenset[str] | None" = None,
    ) -> None:
        self.llm = llm
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
        self.composer = composer
        self.memory: ShortTermMemory = memory or WindowMemory()
        self.guardrail = guardrail or NoopGuardrail()
        self.max_iters = max_iters
        self.parallel_tools = parallel_tools
        self.stream_retries = max(0, stream_retries)
        self._session_id = session_id
        # Shared skill runtime state (None when skills are not configured).
        self.skill_state = skill_state
        # Text-fallback adapter for attachment kinds the model lacks
        # (None — the all-native default — costs nothing).
        self.media_adapter = media_adapter
        self._turn_index: int = 0

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

    @classmethod
    def from_profile(
        cls,
        profile: "AgentProfile",
        *,
        confirm: "ConfirmFn | None" = None,
        llm: _LLMLike | None = None,
        vision_llm: _LLMLike | None = None,
        base_dir: "Path | None" = None,
        tool_options: "dict | None" = None,
        session_store: "SessionStore | None" = None,
        session_id: str | None = None,
    ) -> "Agent":
        """Assemble a runnable Agent from a validated profile.

        ``session_store``/``session_id`` opt into persistent history: the
        memory is hydrated from the store (resuming ``session_id`` when given,
        else starting a fresh session) and every subsequent message is
        recorded. Opening the store is the composition root's job — see
        ``lingcore.sessions.open_store``. ``vision_llm`` overrides the
        ``media_fallback.image`` client (tests inject a fake here, exactly
        like ``llm``).
        """
        from pathlib import Path

        import lingcore.tools.builtin  # noqa: F401  (registration side effect)
        from lingcore.composer import LayeredComposer
        from lingcore.config import AgentProfile  # noqa: F401  (typing only)
        from lingcore.llm import LLMClient
        from lingcore.tools import REGISTRY, ToolContext

        client: _LLMLike = llm or LLMClient(
            model=profile.llm.model,
            api_key=profile.llm.resolve_api_key(
                getattr(profile, "_profile_env", {})
            ),
            base_url=profile.llm.base_url,
            sampling=profile.llm.sampling.as_kwargs(),
            max_retries=profile.llm.max_retries,
            timeout=profile.llm.timeout,
            modalities=profile.llm.modalities,
        )

        # --- Modality fallbacks (only when the model lacks a native kind) ----
        media_adapter: "MediaAdapter | None" = None
        native = frozenset(profile.llm.modalities)
        if native != frozenset({"image", "file"}):
            from lingcore.modality import MediaAdapter

            fb = profile.media_fallback
            vision = vision_llm
            if vision is None and "image" not in native and fb.image is not None:
                # The describe request always renders natively (no modalities
                # narrowing): config validation guarantees the vision model
                # accepts images.
                vision = LLMClient(
                    model=fb.image.model,
                    api_key=fb.image.resolve_api_key(
                        getattr(profile, "_profile_env", {})
                    ),
                    base_url=fb.image.base_url,
                    sampling=fb.image.sampling.as_kwargs(),
                    max_retries=fb.image.max_retries,
                    timeout=fb.image.timeout,
                )
            media_adapter = MediaAdapter(
                native,
                pdf_mode=fb.pdf,
                pdf_max_chars=fb.pdf_max_chars,
                vision=vision,
                vision_prompt=fb.image_prompt,
                vision_max_chars=fb.image_max_chars,
            )

        source_dir = getattr(profile, "_source_dir", None)
        sk_opts = dict(profile.tool_options).get("activate_skill", {})

        # --- Load skills + their shipped tool code BEFORE the tool subset ----
        # A skill may ship its own @tool code (skill.md ``module:``/``provides:``).
        # That code must be registered before ``REGISTRY.subset(profile.tools)``
        # so a profile can authorize a skill-shipped tool by listing it in
        # ``tools:`` exactly like a builtin. Registration is NOT authorization —
        # the subset/ceiling still governs reachability (invariant 13).
        loaded_skills: dict[str, "Skill"] = {}
        if profile.skills or "activate_skill" in profile.tools:
            from lingcore.errors import ConfigError
            from lingcore.skills import load_skill_tools, load_skills

            # Bundled skills first, profile-local second: load_skills is
            # last-write-wins on a name collision, so a profile-local skill
            # shadows a bundled one of the same name (invariant 13) rather than
            # the reverse.
            skill_dirs: list[Path] = [Path(__file__).parent / "skills"]
            if source_dir is not None:
                skill_dirs.append(source_dir / sk_opts.get("skills_dir", "skills"))
            loaded_skills = load_skills(skill_dirs)
            for name in profile.skills:
                if name not in loaded_skills:
                    raise ConfigError(
                        f"profile declares unknown skill {name!r}; "
                        f"available: {sorted(loaded_skills) or '(none)'}"
                    )
            # Import the shipped code only for skills this profile can engage:
            # statically declared, or providing a tool the profile authorizes.
            to_import = {
                name: sk
                for name, sk in loaded_skills.items()
                if sk.module is not None
                and (name in profile.skills or (set(sk.provides) & set(profile.tools)))
            }
            load_skill_tools(to_import)

        tools = REGISTRY.subset(profile.tools)
        # The initially-enabled subset (None ⇒ all of the ceiling). Skills unlock
        # the rest, if any, on activation.
        initial_tools_set = (
            frozenset(profile.initial_tools)
            if profile.initial_tools is not None
            else frozenset(profile.tools)
        )
        # A statically-engaged skill (profile ``skills:``) is always on: its
        # requested tools are granted from the start, exactly like an activated
        # dynamic skill's — intersected with the ceiling, never beyond it
        # (invariant 11). Naming the skill in the profile is the operator's
        # explicit consent, so the high-risk confirmation gate (which guards
        # *model-initiated* activation) does not apply here.
        for skill_name in profile.skills:
            sk = loaded_skills.get(skill_name)
            if sk is not None:
                initial_tools_set |= frozenset(sk.requested_tools) & frozenset(
                    profile.tools
                )
        workspace = profile.workspace_path(base_dir)
        workspace.mkdir(parents=True, exist_ok=True)

        # One effective tool_options dict drives BOTH the ToolContext the tools
        # see and the auto-compaction injection below — reading the override for
        # one and the profile for the other would let a caller's override be
        # silently inverted (profile off + override on wouldn't compact; the
        # reverse would compact anyway).
        effective_tool_options = (
            tool_options if tool_options is not None else dict(profile.tool_options)
        )
        tool_ctx = ToolContext(
            workspace=workspace,
            confirm=confirm,
            options=effective_tool_options,
            profile_dir=source_dir,
            environment=dict(getattr(profile, "_profile_env", {})),
        )
        # Persistent-memory auto-compaction: inject the summarizer (the main
        # client, duck-typed) so the memory tool can condense memory.md at its
        # length limit instead of hard-failing. Opt-in via tool_options.memory.
        mem_tool_opts = effective_tool_options.get("memory", {})
        if "memory" in profile.tools and mem_tool_opts.get("auto_compact", False):
            from lingcore.tools.builtin.memory import MEMORY_SUMMARIZER_KEY

            tool_ctx.options[MEMORY_SUMMARIZER_KEY] = client
        mem = WindowMemory(
            max_messages=profile.memory.max_messages,
            max_tokens=profile.memory.max_tokens,
            model=profile.llm.model,
            evict_to_ratio=profile.memory.evict_to_ratio,
        )
        memory: ShortTermMemory = mem
        # Compaction wraps the window when enabled: it summarizes old history
        # (via the main client, duck-typed) before the window's eviction fires.
        if profile.memory.compaction.enabled:
            from lingcore.memory import SummarizingMemory

            cc = profile.memory.compaction
            memory = SummarizingMemory(
                mem,
                summarizer=client,
                compact_at_ratio=cc.compact_at_ratio,
                keep_recent_ratio=cc.keep_recent_ratio,
                max_summary_chars=cc.max_summary_chars,
            )
        sid = session_id
        restored_turn_index = 0
        if session_store is not None:
            from lingcore.sessions import attach_session

            memory, sid, restored_turn_index = attach_session(
                memory, session_store, session_id
            )
        # Prompt-cache routing key: bind same-session requests to one warm node.
        # Set post-construction now that the session id is known; only on a real
        # LLMClient we built (a test-injected fake is left untouched).
        if profile.llm.send_prompt_cache_key and sid and isinstance(client, LLMClient):
            client._prompt_cache_key = sid
        guardrail = _build_guardrail(profile.guardrail.policy)

        # --- Dynamic skill state (only when the activate_skill tool is enabled) ---
        skill_state: "SkillState | None" = None
        if "activate_skill" in profile.tools:
            from lingcore.skills import SkillState
            from lingcore.tools.builtin.skill import SKILL_STATE_KEY

            ptools = frozenset(profile.tools)
            # Only offer skills this profile can actually use. A skill is
            # activatable only if the profile authorizes at least one of its
            # requested tools (or it ships none and is purely instructional).
            # Otherwise a code-shipping skill like ``canvas`` is advertised to a
            # profile that can grant none of its tools — effective_tools is ∅, so
            # activating it would only mislead the model.
            offerable = {
                name: sk
                for name, sk in loaded_skills.items()
                if not sk.requested_tools or (ptools & frozenset(sk.requested_tools))
            }
            high_risk = sk_opts.get("high_risk_tools")
            skill_state = SkillState(
                skills=offerable,
                profile_tools=ptools,
                allow_concurrent=bool(sk_opts.get("allow_concurrent", False)),
                high_risk_tools=frozenset(high_risk) if high_risk is not None
                else SkillState.high_risk_tools,
            )
            # Share the live state object with the activate_skill tool.
            tool_ctx.options[SKILL_STATE_KEY] = skill_state

        # --- Build composer ----------------------------------------------
        # LayeredComposer when .md layers, memory, or skills exist; else Static.
        composer: PromptComposer
        layers: list[str] = []
        includes: list[str] = []
        memory_path: Path | None = None
        if source_dir is not None:
            layer_names = ["world.md", "role.md", "workflow.md"]
            layers = [
                (source_dir / name).read_text("utf-8")
                for name in layer_names
                if (source_dir / name).is_file()
            ]
            includes = [
                (source_dir / inc).read_text("utf-8")
                for inc in profile.persona.include
                if (source_dir / inc).is_file()
            ]
            # Read memory settings from the *effective* options (caller overrides
            # included), so the injected memory file matches where the memory
            # tool actually writes. An absolute path is honored only when the
            # profile opted in — the same gate the tool enforces before writing.
            mem_opts = effective_tool_options.get("memory", {})
            if "memory" in profile.tools:
                raw_mem = Path(mem_opts.get("path", "memory.md"))
                if raw_mem.is_absolute():
                    if mem_opts.get("allow_absolute_path", False):
                        memory_path = raw_mem.resolve()
                else:
                    memory_path = (source_dir / raw_mem).resolve()

        # Statically-declared skills (profile ``skills:``) inject their
        # instructions as an always-on prompt layer — distinct from the dynamic
        # activate_skill path, which injects via active_skills in the loop.
        static_skill_layers = [
            loaded_skills[name].instructions
            for name in profile.skills
            if loaded_skills.get(name) and loaded_skills[name].instructions
        ]

        # Layer the prompt when there are real layer sources (files, includes,
        # static skills), a memory file, or dynamic skills. The persona's inline
        # system_prompt does NOT by itself trigger layering — a bare-prompt agent
        # stays a zero-overhead StaticComposer.
        needs_layering = (
            bool(layers or includes or static_skill_layers)
            or memory_path is not None
            or skill_state is not None
        )
        if needs_layering:
            # persona.system_prompt is the inline *fallback* persona: fold it in
            # as the base layer when no world/role/workflow file supplies one, so
            # a profile relying on the inline prompt doesn't silently lose it just
            # because memory or skills forced the layered path.
            base_layers = layers
            if not layers and profile.persona.system_prompt.strip():
                base_layers = [profile.persona.system_prompt]
            composer = LayeredComposer(
                layers=base_layers + includes + static_skill_layers,
                memory_path=memory_path,
                skill_instructions=skill_state.instruction_map() if skill_state else {},
            )
        else:
            composer = StaticComposer(profile.persona.system_prompt)

        agent = cls(
            llm=client,
            tools=tools,
            tool_ctx=tool_ctx,
            composer=composer,
            memory=memory,
            guardrail=guardrail,
            max_iters=profile.loop.max_iters,
            parallel_tools=profile.loop.parallel_tools,
            stream_retries=profile.llm.stream_retries,
            session_id=sid,
            skill_state=skill_state,
            media_adapter=media_adapter,
            initial_tools=initial_tools_set,
        )
        agent._turn_index = restored_turn_index
        return agent

    async def run(self, user_input: str | UserInput) -> AsyncIterator[AgentEvent]:
        """Drive one user turn to completion, yielding events as they happen."""
        if isinstance(user_input, str):
            incoming = UserInput(text=user_input)
        else:
            incoming = user_input
        text = await self.guardrail.pre_input(incoming.text)
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
        self.memory.add(Message.user(text, attachments=attachments))

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
            system_prompt = await self.composer.compose(compose_ctx)

            # Turn-boundary compaction: once per turn, before the first request,
            # so that request already sees the compacted context. Done here (not
            # mid tool-loop) so within-turn iterations stay append-only and
            # cache-stable. The system prompt is passed so the trigger measures
            # the same footprint the window does. No-op unless a
            # SummarizingMemory is configured; never raises.
            if not compacted_this_turn:
                compacted_this_turn = True
                compacted = await self.memory.maybe_compact(system_prompt)
                if compacted is not None:
                    yield compacted

            self._turn_index += 1

            messages = self.memory.render(system_prompt)
            schemas = self._effective_tool_schemas()

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
                tool_calls: list[ToolCall] = []
                try:
                    async for chunk in self.llm.stream(messages, tools=schemas):
                        if chunk.text_delta:
                            content_parts.append(chunk.text_delta)
                            yield TextDelta(chunk.text_delta)
                        if chunk.tool_calls:
                            tool_calls = chunk.tool_calls
                    break
                except LLMStreamError as e:
                    attempt += 1
                    if not e.retryable or attempt > self.stream_retries:
                        spent = (
                            f" after {self.stream_retries} retries"
                            if e.retryable and self.stream_retries
                            else ""
                        )
                        yield Error(f"model request failed{spent}: {e}")
                        return
                    yield StreamRetry(
                        attempt=attempt,
                        max_attempts=self.stream_retries,
                        reason=str(e),
                        discarded_chars=sum(len(p) for p in content_parts),
                    )
                    await asyncio.sleep(_backoff_seconds(attempt))
                except Exception as e:
                    # A duck-typed backend may raise anything; without a
                    # retryable classification, surface it and end the turn.
                    yield Error(f"model request failed: {type(e).__name__}: {e}")
                    return

            assistant = Message.assistant(
                content="".join(content_parts), tool_calls=tool_calls
            )
            self.memory.add(assistant)

            if not assistant.tool_calls:
                final = await self.guardrail.post_output(assistant.content)
                yield Final(final)
                return

            for call in assistant.tool_calls:
                yield ToolCallStarted(call)

            before = set(self.skill_state.active) if self.skill_state else set()
            results = await self._dispatch(assistant.tool_calls)
            for result in results:
                self.memory.add(Message.from_tool_result(result))
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
                self.memory.add(Message(
                    role="user",
                    content=note,
                    name="media",
                    attachments=kept,
                ))

            # Surface skill activation/deactivation changes as events.
            if self.skill_state is not None:
                after = set(self.skill_state.active)
                for name in sorted(after - before):
                    yield SkillActivated(name=name, active=True)
                for name in sorted(before - after):
                    yield SkillActivated(name=name, active=False)

        yield Error(f"reached max iterations ({self.max_iters}) without a final reply")

    def _effective_tool_schemas(self) -> list[dict[str, Any]]:
        """Tool schemas for the current step: the initially-enabled tools plus
        any tools granted by active skills (intersected with the profile
        allowlist by SkillState).

        Computed fresh each iteration; the base ToolRegistry is never mutated.
        Tools in the ceiling but neither initial nor granted by an active skill
        are NOT advertised — progressive disclosure (invariant 11).
        """
        base = [
            self.tools.get(name).json_schema()
            for name in self.tools.names()
            if name in self.initial_tools
        ]
        if self.skill_state is None:
            return base

        # Disclose available skills in activate_skill's description so the model
        # knows what it can invoke (progressive disclosure via the tool spec).
        if self.skill_state.skills:
            listing = "\n".join(
                f"  - {s.name}: {s.description}"
                for s in self.skill_state.skills.values()
            )
            for schema in base:
                fn = schema["function"]
                if fn["name"] == "activate_skill":
                    fn["description"] = fn["description"] + "\nAvailable skills:\n" + listing
                    break

        if not self.skill_state.active:
            return base
        seen = {s["function"]["name"] for s in base}
        extra_names = self.skill_state.active_effective_tools() - seen
        for name in sorted(extra_names):
            if name in self.tools.names():
                base.append(self.tools.get(name).json_schema())
        return base

    async def _dispatch(self, calls: list[ToolCall]) -> list[ToolResult]:
        # Authorization is a property of the model request that produced this
        # batch. Snapshot it before any tool runs: activate_skill/deactivation in
        # one call must affect only the next loop iteration, never sibling calls
        # whose schemas were computed under the previous state.
        authorized = self._authorized_tool_names()

        async def run_one(call: ToolCall) -> ToolResult:
            try:
                tool = self._resolve_tool(call.name, authorized=authorized)
                args = tool.validate_args(call.arguments)
                out = await tool.run(args, self.tool_ctx)
                if isinstance(out, ToolOutput):
                    return ToolResult(
                        call_id=call.id,
                        name=call.name,
                        content=out.text,
                        attachments=out.attachments,
                        ok=True,
                    )
                return ToolResult(call_id=call.id, name=call.name, content=out, ok=True)
            except Exception as e:
                from lingcore.errors import ToolError

                msg = str(e) if isinstance(e, ToolError) else f"internal error: {e!r}"
                return ToolResult(
                    call_id=call.id, name=call.name, content=f"ERROR: {msg}", ok=False
                )

        if self.parallel_tools and len(calls) > 1:
            return list(await asyncio.gather(*(run_one(c) for c in calls)))
        return [await run_one(c) for c in calls]

    def _authorized_tool_names(self) -> frozenset[str]:
        """Currently dispatchable tools, bounded by the profile registry."""
        authorized = self.initial_tools
        if self.skill_state is not None:
            authorized |= self.skill_state.active_effective_tools()
        return authorized & frozenset(self.tools.names())

    def _resolve_tool(
        self, name: str, *, authorized: frozenset[str] | None = None
    ):
        """Look up a tool by name, enforcing the permission model at dispatch.

        A tool is dispatchable only when it is currently *authorized*: in the
        initially-enabled set, or granted by an active skill (profile ∩
        requested). A ceiling tool that is neither — advertised or not — is
        refused with the same error an unknown tool would raise, so a model that
        calls a gated-but-not-active tool can't reach it.
        """
        allowed = authorized if authorized is not None else self._authorized_tool_names()
        if name in allowed:
            return self.tools.get(name)
        from lingcore.errors import ToolError

        raise ToolError(f"unknown tool: {name}")
