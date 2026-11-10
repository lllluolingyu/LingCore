"""Profile-driven Agent assembly and dependency wiring."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from lingcore.composer import PromptComposer, StaticComposer
from lingcore.execution import ExecutionBudget
from lingcore.memory import ShortTermMemory, WindowMemory
from lingcore.plugins.commands import CommandCatalog
from lingcore.plugins.hooks import HookRunner
from lingcore.session_leases import SessionLease
from lingcore.todos import TodoState
from lingcore.tools import ToolContext, ToolRegistry
from lingcore.tracing import TraceRecorder
from lingcore.tracing.requests import InstrumentedClient, RequestLedger
from lingcore.usage import UsageSink

if TYPE_CHECKING:
    from pathlib import Path

    from lingcore.agent import Agent, _LLMLike
    from lingcore.config import AgentProfile
    from lingcore.modality import MediaAdapter
    from lingcore.sessions import SessionStore
    from lingcore.skills import SkillState
    from lingcore.tools import ConfirmFn, EnvironmentBinding


def from_profile(
    cls: type[Agent],
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
    from pathlib import Path

    import lingcore.tools.builtin  # noqa: F401  (registration side effect)
    from lingcore.agent import UsageMeter, build_guardrail
    from lingcore.composer import LayeredComposer
    from lingcore.config import LLMCfg
    from lingcore.llm import LLMClient
    from lingcore.llm_anthropic import AnthropicLLMClient
    from lingcore.llm_responses import ResponsesLLMClient
    from lingcore.output_rewrite import OutputRewriter
    from lingcore.tools import REGISTRY

    # One meter per agent: the main client doubles as the compaction and
    # memory summarizer, so every request it makes is accounted here.
    usage_meter = UsageMeter()

    def make_client(cfg: LLMCfg, *, usage_sink: UsageSink | None = None) -> _LLMLike:
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "api_key": cfg.resolve_api_key(getattr(profile, "_profile_env", {})),
            "base_url": cfg.base_url,
            "sampling": cfg.sampling.as_kwargs(),
            "max_retries": cfg.max_retries,
            "timeout": cfg.timeout,
            "modalities": cfg.modalities,
            "stream_usage": cfg.stream_usage,
            "usage_sink": usage_sink or usage_meter.record,
            "preserve_reasoning": cfg.preserve_reasoning,
        }
        if cfg.backend == "anthropic":
            return AnthropicLLMClient(**kwargs, prompt_caching=cfg.prompt_caching)
        if cfg.backend == "openai-responses":
            return ResponsesLLMClient(**kwargs)
        return LLMClient(**kwargs)

    client = llm if llm is not None else make_client(profile.llm)
    if (
        llm is not None
        and hasattr(llm, "usage_sink")
        and getattr(llm, "usage_sink") is None
    ):
        setattr(llm, "usage_sink", usage_meter.record)
    if (
        vision_llm is not None
        and hasattr(vision_llm, "usage_sink")
        and getattr(vision_llm, "usage_sink") is None
    ):
        setattr(vision_llm, "usage_sink", usage_meter.record)

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
            vision = make_client(fb.image)
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

    # Discovery reads only data; loading executes only engaged code. The
    # plugin/skill catalog must exist before selecting the explicit ceiling.
    from lingcore.errors import ConfigError
    from lingcore.plugins.loader import LoadedPlugins, load_plugins

    if tool_registry is not None:
        if (
            profile.plugins
            or profile.skills
            or "activate_skill" in profile.tools
            or sk_opts
        ):
            raise ConfigError(
                "an injected registry cannot be combined with plugins or skills"
            )
        if set(tool_registry.names()) != set(profile.tools):
            raise ConfigError(
                "injected registry must match the profile tool ceiling exactly"
            )
        loaded_plugins = LoadedPlugins([], {}, frozenset(), [], CommandCatalog(), [])
    else:
        loaded_plugins = load_plugins(profile)
    loaded_skills = loaded_plugins.skills
    for name in profile.skills:
        if name not in loaded_skills:
            from lingcore.errors import ConfigError

            raise ConfigError(
                f"profile declares unknown skill {name!r}; "
                f"available: {sorted(loaded_skills) or '(none)'}"
            )

    from lingcore.skills import registered_module_tools

    # Process-global registration survives Agent shutdown. A later profile
    # still needs its own engagement; cached code must not supply consent.
    unengaged = (
        set(profile.tools) & registered_module_tools()
    ) - loaded_plugins.tool_names
    if unengaged and tool_registry is None:
        from lingcore.errors import ConfigError

        raise ConfigError(
            f"profile tools belong to modules not engaged by this profile: {sorted(unengaged)}"
        )
    tools = (
        tool_registry if tool_registry is not None else REGISTRY.subset(profile.tools)
    )
    # The initially-enabled subset (``initial_tools`` or the complement of
    # ``skill_gated_tools``; neither ⇒ all of the ceiling). Skills unlock the
    # rest, if any, on activation.
    initial_tools_set = profile.initial_tool_set()
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
    if state_dir is not None:
        from lingcore.profiles import initialize_state_directory

        state_dir = initialize_state_directory(state_dir)
    effective_state = state_dir or source_dir
    workspace = (
        state_dir / "workspace"
        if state_dir is not None and profile.workspace is None
        else profile.workspace_path(base_dir)
    )
    workspace.mkdir(parents=True, exist_ok=True)

    from lingcore.tracing.config import build_recorder

    recorder = (
        recorder
        if recorder is not None
        else build_recorder(profile, workspace, state_dir)
    )
    owns_store = False
    session_lease: SessionLease | None = None
    try:
        ledger = RequestLedger()
        raw_client = client

        def instrument(
            raw: _LLMLike, cfg: LLMCfg, purpose: str | None = None
        ) -> _LLMLike:
            if not recorder.enabled and execution_budget is None:
                return raw
            return InstrumentedClient(
                raw,
                recorder,
                ledger,
                model=cfg.model,
                provider=cfg.backend,
                max_retries=cfg.max_retries,
                default_purpose=purpose,
                budget=execution_budget,
            )

        client = instrument(raw_client, profile.llm)
        if media_adapter is not None and media_adapter.vision is not None:
            media_adapter.vision = instrument(
                media_adapter.vision,
                profile.media_fallback.image or profile.llm,
                "vision",
            )

        output_rewriter = None
        rewrite_cfg = profile.output_rewrite
        if rewrite_cfg.enabled:
            assert rewrite_cfg.llm is not None
            rewrite_sink = partial(usage_meter.record, purpose="output_rewrite")
            rewrite_client = (
                rewrite_llm
                if rewrite_llm is not None
                else make_client(rewrite_cfg.llm, usage_sink=rewrite_sink)
            )
            if (
                rewrite_llm is not None
                and hasattr(rewrite_llm, "usage_sink")
                and getattr(rewrite_llm, "usage_sink") is None
            ):
                setattr(rewrite_llm, "usage_sink", rewrite_sink)
            output_rewriter = OutputRewriter(
                instrument(rewrite_client, rewrite_cfg.llm, "output_rewrite"),
                rewrite_cfg,
            )

        # One effective tool_options dict drives BOTH the ToolContext the tools
        # see and the auto-compaction injection below — reading the override for
        # one and the profile for the other would let a caller's override be
        # silently inverted (profile off + override on wouldn't compact; the
        # reverse would compact anyway).
        effective_tool_options = (
            tool_options if tool_options is not None else dict(profile.tool_options)
        )
        if state_dir is not None:
            from lingcore.paths import PathEscapeError, resolve_confined

            for key, default in (
                ("memory", "memory.md"),
                ("calendar", "calendar.ics"),
                ("world", ".lingcore/world.md"),
            ):
                options = effective_tool_options.get(key, {})
                raw = Path(options.get("path", default))
                if raw.is_absolute() and not options.get("allow_absolute_path", False):
                    raise ConfigError(
                        f"absolute {key} path requires allow_absolute_path"
                    )
                try:
                    resolved = resolve_confined(state_dir, raw)
                except PathEscapeError:
                    raise ConfigError(
                        f"{key} path escapes the explicit state directory"
                    ) from None
                if not resolved.is_relative_to(state_dir):
                    raise ConfigError(
                        f"{key} path escapes the explicit state directory"
                    )
        owns_store = (
            state_dir is not None and session_store is None and profile.sessions.enabled
        )
        if owns_store:
            from lingcore.sessions import open_store

            session_store, _ = open_store(profile, state_dir=state_dir)
        if (
            state_dir is not None
            and session_store is not None
            and not session_store.db_path.resolve().is_relative_to(state_dir)
        ):
            raise ConfigError("session store escapes the explicit state directory")
        # Persistent-memory auto-compaction: inject the summarizer (the main
        # client, duck-typed) so the memory tool can condense memory.md at its
        # length limit instead of hard-failing. Opt-in via tool_options.memory.
        mem_tool_opts = effective_tool_options.get("memory", {})
        if "memory" in profile.tools and mem_tool_opts.get("auto_compact", False):
            from lingcore.tools.builtin.memory import MEMORY_SUMMARIZER_KEY

            effective_tool_options[MEMORY_SUMMARIZER_KEY] = instrument(
                raw_client, profile.llm, "memory_condensation"
            )
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
                summarizer=instrument(raw_client, profile.llm, "summary"),
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
            session_lease = memory.lease
        # Prompt-cache routing key: bind same-session requests to one warm node.
        # Set post-construction now that the session id is known; only on a real
        # LLMClient we built (a test-injected fake is left untouched).
        if (
            profile.llm.send_prompt_cache_key
            and sid
            and isinstance(raw_client, LLMClient)
        ):
            raw_client._prompt_cache_key = sid
        # Built whole once the session id is known: a tool that keeps per-chat
        # auxiliary state (e.g. external-agent conversation aliases) scopes it by
        # ``session_id``, so the context must carry the resumed/fresh id from
        # the start rather than have it patched in afterwards.
        tool_ctx = ToolContext(
            workspace=workspace,
            confirm=confirm,
            options=effective_tool_options,
            profile_dir=effective_state,
            environment=dict(getattr(profile, "_profile_env", {})),
            session_id=sid,
            environment_binding=environment_binding,
        )
        guardrail = build_guardrail(profile.guardrail.policy, profile.guardrail.options)

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
            # The name-based baseline (or the operator's override of it) plus
            # every authorized tool that declares itself ``high_risk`` — a
            # tool's own declaration is a floor the override cannot lower, so
            # skill-shipped code stays confirmation-gated without the core
            # permission model listing its names.
            high_risk = sk_opts.get("high_risk_tools")
            declared_high_risk = frozenset(
                name for name in tools.names() if tools.get(name).high_risk
            )
            skill_state = SkillState(
                skills=offerable,
                profile_tools=ptools,
                allow_concurrent=bool(sk_opts.get("allow_concurrent", False)),
                high_risk_tools=(
                    frozenset(high_risk)
                    if high_risk is not None
                    else SkillState.high_risk_tools
                )
                | declared_high_risk,
            )
            if session_store is not None and sid is not None:
                persisted = session_store.active_skill_state(sid)
                restored = [
                    name
                    for name in persisted.active
                    if name in skill_state.skills
                    and (
                        skill_state.effective_tools(skill_state.skills[name])
                        & skill_state.high_risk_tools
                    )
                    <= (persisted.approvals_for(name) & skill_state.high_risk_tools)
                ]
                if not skill_state.allow_concurrent and len(restored) > 1:
                    restored = restored[-1:]
                skill_state.active[:] = restored
                skill_state.approved_high_risk = {
                    name: (persisted.approvals_for(name) & skill_state.high_risk_tools)
                    for name in restored
                }
            # Share the live state object with the activate_skill tool.
            tool_ctx.options[SKILL_STATE_KEY] = skill_state

        # --- Todo checklist (only when the todo_write tool is enabled) -------
        todo_state: TodoState | None = None
        if "todo_write" in profile.tools:
            from lingcore.tools.builtin.todo import TODO_STATE_KEY

            todo_state = TodoState()
            if session_store is not None and sid is not None:
                todo_state.items = session_store.latest_todos(sid)
            tool_ctx.options[TODO_STATE_KEY] = todo_state

        # --- Build composer ----------------------------------------------
        # LayeredComposer when .md layers, memory, or skills exist; else Static.
        composer: PromptComposer
        layers: list[str] = []
        includes: list[str] = []
        memory_path: Path | None = None
        from lingcore.world import WORLD_STATE_KEY, WorldFile, build_world_file

        world_file: WorldFile | None = None
        authored_world = ""
        if "world" in profile.tools and effective_state is not None:
            source_world = source_dir / "world.md" if source_dir is not None else None
            if source_world is not None and source_world.is_file():
                authored_world = source_world.read_text("utf-8")
            world_file = build_world_file(
                effective_state,
                effective_tool_options.get("world", {}),
                fallback=authored_world,
            )
            if (
                source_world is not None
                and world_file.path.absolute() == source_world.absolute()
            ):
                raise ConfigError(
                    "world runtime path must differ from authored world.md"
                )
            tool_ctx.options[WORLD_STATE_KEY] = world_file
        if source_dir is not None:
            layer_names = ["role.md", "workflow.md"]
            if world_file is None:
                layer_names.insert(0, "world.md")
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
        if effective_state is not None:
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
                    memory_path = (effective_state / raw_mem).resolve()

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
            bool(
                layers
                or includes
                or static_skill_layers
                or loaded_plugins.prompt_layers
            )
            or memory_path is not None
            or world_file is not None
            or skill_state is not None
            or bool(profile.persona.project_instructions)
        )
        if needs_layering:
            # persona.system_prompt is the inline *fallback* persona: fold it in
            # as the base layer when no world/role/workflow file supplies one, so
            # a profile relying on the inline prompt doesn't silently lose it just
            # because memory or skills forced the layered path.
            base_layers = layers
            if (
                not layers
                and not authored_world.strip()
                and profile.persona.system_prompt.strip()
            ):
                base_layers = [profile.persona.system_prompt]
            composer = LayeredComposer(
                layers=base_layers
                + includes
                + static_skill_layers
                + loaded_plugins.prompt_layers,
                memory_path=memory_path,
                skill_instructions=skill_state.instruction_map() if skill_state else {},
                workspace=workspace,
                project_instructions=tuple(profile.persona.project_instructions),
                world=world_file,
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
            session_store=session_store,
            usage_meter=usage_meter,
            todo_state=todo_state,
            hooks=HookRunner(loaded_plugins.hook_factories, tool_ctx),
            commands=loaded_plugins.commands,
            recorder=recorder,
            request_ledger=ledger,
            execution_budget=execution_budget,
            output_rewriter=output_rewriter,
        )
        agent._turn_index = restored_turn_index
        agent._owned_session_store = session_store if owns_store else None
        if session_store is not None and sid is not None:
            meta = session_store.get(sid)
            if meta is not None and meta.fork is not None:
                recorder.event(
                    "session.link",
                    attributes={
                        "source_session_id": meta.fork.parent_session_id,
                        "through_seq": meta.fork.through_seq,
                        "relation": "fork",
                    },
                )
        return agent
    except BaseException:
        recorder.abort()
        try:
            if session_store is not None and session_lease is not None:
                session_store.release_session(session_lease)
        finally:
            if owns_store and session_store is not None:
                session_store.close()
        raise
