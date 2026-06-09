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
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol

from lingcore.composer import ComposeContext, PromptComposer, StaticComposer
from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    SkillActivated,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.guardrails import Guardrail, NoopGuardrail
from lingcore.llm import LLMChunk
from lingcore.memory import ShortTermMemory, WindowMemory
from lingcore.message import Message, ToolCall, ToolResult
from lingcore.tools import ToolContext, ToolRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from lingcore.config import AgentProfile
    from lingcore.skills import Skill, SkillState
    from lingcore.tools import ConfirmFn


def _build_guardrail(policy: str) -> Guardrail:
    """Map a profile's guardrail policy name to an implementation."""
    if policy == "noop":
        return NoopGuardrail()
    from lingcore.errors import ConfigError

    raise ConfigError(f"unknown guardrail policy: {policy!r}")


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
        session_id: str | None = None,
        skill_state: "SkillState | None" = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_ctx = tool_ctx
        self.composer = composer
        self.memory: ShortTermMemory = memory or WindowMemory()
        self.guardrail = guardrail or NoopGuardrail()
        self.max_iters = max_iters
        self.parallel_tools = parallel_tools
        self._session_id = session_id
        # Shared skill runtime state (None when skills are not configured).
        self.skill_state = skill_state
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
        base_dir: "Path | None" = None,
        tool_options: "dict | None" = None,
    ) -> "Agent":
        """Assemble a runnable Agent from a validated profile."""
        from pathlib import Path

        import lingcore.tools.builtin  # noqa: F401  (registration side effect)
        from lingcore.composer import LayeredComposer
        from lingcore.config import AgentProfile  # noqa: F401  (typing only)
        from lingcore.llm import LLMClient
        from lingcore.tools import REGISTRY, ToolContext

        client: _LLMLike = llm or LLMClient(
            model=profile.llm.model,
            api_key=profile.llm.resolve_api_key(),
            base_url=profile.llm.base_url,
            sampling=profile.llm.sampling.as_kwargs(),
            max_retries=profile.llm.max_retries,
            timeout=profile.llm.timeout,
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

            skill_dirs: list[Path] = []
            if source_dir is not None:
                skill_dirs.append(source_dir / sk_opts.get("skills_dir", "skills"))
            skill_dirs.append(Path(__file__).parent / "skills")
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
        workspace = profile.workspace_path(base_dir)
        workspace.mkdir(parents=True, exist_ok=True)

        tool_ctx = ToolContext(
            workspace=workspace,
            confirm=confirm,
            options=tool_options if tool_options is not None else dict(profile.tool_options),
            profile_dir=source_dir,
        )
        mem = WindowMemory(
            max_messages=profile.memory.max_messages,
            max_tokens=profile.memory.max_tokens,
            model=profile.llm.model,
        )
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
            mem_opts = dict(profile.tool_options).get("memory", {})
            if "memory" in profile.tools:
                raw_mem = Path(mem_opts.get("path", "memory.md"))
                if not raw_mem.is_absolute():
                    memory_path = (source_dir / raw_mem).resolve()

        # Statically-declared skills (profile ``skills:``) inject their
        # instructions as an always-on prompt layer — distinct from the dynamic
        # activate_skill path, which injects via active_skills in the loop.
        static_skill_layers = [
            loaded_skills[name].instructions
            for name in profile.skills
            if loaded_skills.get(name) and loaded_skills[name].instructions
        ]
        all_layers = layers + includes + static_skill_layers

        if all_layers or memory_path is not None or skill_state is not None:
            composer = LayeredComposer(
                layers=all_layers,
                memory_path=memory_path,
                skill_instructions=skill_state.instruction_map() if skill_state else {},
            )
        else:
            composer = StaticComposer(profile.persona.system_prompt)

        return cls(
            llm=client,
            tools=tools,
            tool_ctx=tool_ctx,
            composer=composer,
            memory=mem,
            guardrail=guardrail,
            max_iters=profile.loop.max_iters,
            parallel_tools=profile.loop.parallel_tools,
            skill_state=skill_state,
        )

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Drive one user turn to completion, yielding events as they happen."""
        text = await self.guardrail.pre_input(user_input)
        self.memory.add(Message.user(text))

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
            self._turn_index += 1

            messages = self.memory.render(system_prompt)
            schemas = self._effective_tool_schemas()
            content_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            async for chunk in self.llm.stream(messages, tools=schemas):
                if chunk.text_delta:
                    content_parts.append(chunk.text_delta)
                    yield TextDelta(chunk.text_delta)
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

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

            # Surface skill activation/deactivation changes as events.
            if self.skill_state is not None:
                after = set(self.skill_state.active)
                for name in sorted(after - before):
                    yield SkillActivated(name=name, active=True)
                for name in sorted(before - after):
                    yield SkillActivated(name=name, active=False)

        yield Error(f"reached max iterations ({self.max_iters}) without a final reply")

    def _effective_tool_schemas(self) -> list[dict[str, Any]]:
        """Tool schemas for the current step: base tools plus any tools granted
        by active skills (intersected with the profile allowlist by SkillState).

        Computed fresh each iteration; the base ToolRegistry is never mutated.
        """
        base = self.tools.schemas()
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
            try:
                from lingcore.tools import REGISTRY

                base.append(REGISTRY.get(name).json_schema())
            except Exception:
                continue
        return base

    async def _dispatch(self, calls: list[ToolCall]) -> list[ToolResult]:
        async def run_one(call: ToolCall) -> ToolResult:
            try:
                tool = self._resolve_tool(call.name)
                args = tool.validate_args(call.arguments)
                out = await tool.run(args, self.tool_ctx)
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

    def _resolve_tool(self, name: str):
        """Look up a tool by name: base subset first, then skill-granted tools.

        A skill-granted tool is only resolvable if it is in the *effective* set
        (profile ∩ active-skill requested), enforcing the permission ceiling at
        execution time — not just in the advertised schemas.
        """
        if name in self.tools.names():
            return self.tools.get(name)
        if self.skill_state is not None and name in self.skill_state.active_effective_tools():
            from lingcore.tools import REGISTRY

            return REGISTRY.get(name)
        # Not permitted: raise the same error the registry would.
        return self.tools.get(name)
