"""Plugin hooks at Agent authorization, persistence and cancellation boundaries."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

import lingcore.tools.builtin  # noqa: F401
from lingcore.agent import Agent
from lingcore.events import Error, Final, PluginNotice, ToolResultEvent, TurnCancelled
from lingcore.memory import WindowMemory
from lingcore.message import ToolCall, UserInput
from lingcore.plugins import (
    PluginHooks,
    ToolDecision,
    ToolResultPatch,
    UserMessageDecision,
)
from lingcore.plugins.hooks import HookFactory, HookRunner
from lingcore.sessions import SessionStore, attach_session
from lingcore.tools import REGISTRY, ToolContext, ToolRegistry, tool
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure


class EchoArgs(BaseModel):
    text: str


def build(
    tmp_path,
    classes,
    *,
    turns=None,
    confirm=None,
    store=None,
    parallel=False,
    initial=None,
    timeout=1,
    mode="block",
    registry=None,
):
    ctx = ToolContext(workspace=tmp_path, confirm=confirm)
    factories = [
        HookFactory(f"p{i}", tmp_path, cls, f"p{i}", mode, timeout)
        for i, cls in enumerate(classes)
    ]
    hooks = HookRunner(factories, ctx)
    reg = registry or ToolRegistry()
    if registry is None:

        @tool(name="echo", registry=reg)
        async def echo(args: EchoArgs, ctx: ToolContext) -> str:
            return args.text

    llm = FakeLLMClient(turns or [ScriptedTurn(text="done")])
    memory = WindowMemory()
    sid = None
    if store is not None:
        memory, sid, _ = attach_session(memory, store, None)
    agent = Agent(
        llm,
        reg,
        ctx,
        system_prompt="sys",
        memory=memory,
        session_store=store,
        session_id=sid,
        hooks=hooks,
        parallel_tools=parallel,
        initial_tools=initial,
    )
    return agent, llm


def tool_turn(name="echo", args=None, count=1):
    return ScriptedTurn(
        tool_calls=[
            ToolCall(
                id=f"c{i}",
                name=name,
                arguments=args if args is not None else {"text": "original"},
            )
            for i in range(count)
        ]
    )


async def drain(agent, text="hello"):
    return [event async for event in agent.run(text)]


async def test_user_block_leaves_no_memory_or_session(tmp_path):
    class Block(PluginHooks):
        async def user_message(self, e):
            return UserMessageDecision.block("private")

    with SessionStore(tmp_path / "sessions.db") as store:
        agent, llm = build(tmp_path, [Block], store=store)
        events = await drain(agent)
        assert agent.memory.messages == []
        assert store.list() == []
        assert llm.calls == []
        assert isinstance(events[-2], PluginNotice)
        assert events[-2].action == "blocked"
        assert isinstance(events[-1], Error)


async def test_context_and_display_text_survive_session_roundtrip(tmp_path):
    class Context(PluginHooks):
        async def user_message(self, e):
            assert e.text == "Review src/"
            return UserMessageDecision.add_context("policy context")

    with SessionStore(tmp_path / "sessions.db") as store:
        agent, llm = build(tmp_path, [Context], store=store)
        await drain(
            agent, UserInput(text="Review src/", display_text="/audit:review src/")
        )
        user = next(m for m in llm.calls[0] if m.role == "user")
        assert user.content == "Review src/\npolicy context"
        assert user.input_text == "/audit:review src/"
        restored = store.messages(agent._session_id)
        assert restored[0].input_text == "/audit:review src/"
        assert restored[0].content == user.content


async def test_deny_stops_before_chain_and_preserves_failed_result(tmp_path):
    order = []

    class Deny(PluginHooks):
        async def before_tool(self, e):
            order.append("deny")
            return ToolDecision.deny("restricted")

    class Later(PluginHooks):
        async def before_tool(self, e):
            order.append("later")

    agent, _ = build(
        tmp_path, [Deny, Later], turns=[tool_turn(), ScriptedTurn(text="done")]
    )
    events = await drain(agent)
    result = next(e.result for e in events if isinstance(e, ToolResultEvent))
    assert order == ["deny"]
    assert not result.ok
    assert "blocked by plugin p0" in result.content
    assert "restricted" in result.content
    assert any(isinstance(e, PluginNotice) and e.action == "denied" for e in events)


@pytest.mark.parametrize("approval", [True, False, None])
async def test_ask_approval_decline_and_missing_handler(tmp_path, approval):
    prompts = []

    async def confirm(prompt):
        prompts.append(prompt)
        return approval

    class Ask(PluginHooks):
        async def before_tool(self, e):
            return ToolDecision.ask("Proceed?")

    agent, _ = build(
        tmp_path,
        [Ask],
        confirm=confirm if approval is not None else None,
        turns=[tool_turn(), ScriptedTurn(text="done")],
    )
    events = await drain(agent)
    result = next(e.result for e in events if isinstance(e, ToolResultEvent))
    assert result.ok is (approval is True)
    assert prompts == ([] if approval is None else ["Proceed?"])


async def test_allow_still_reaches_shell_confirmation(tmp_path):
    prompts = []

    async def confirm(prompt):
        prompts.append(prompt)
        return False

    class Allow(PluginHooks):
        async def before_tool(self, e):
            return ToolDecision.allow()

    reg = ToolRegistry()
    reg.register(REGISTRY.get("run_shell"))
    agent, _ = build(
        tmp_path,
        [Allow],
        registry=reg,
        confirm=confirm,
        turns=[
            tool_turn("run_shell", {"command": "echo hello"}),
            ScriptedTurn(text="done"),
        ],
    )
    result = next(
        e.result for e in await drain(agent) if isinstance(e, ToolResultEvent)
    )
    assert prompts and not result.ok


async def test_only_valid_authorized_calls_reach_hooks(tmp_path):
    calls = []

    class Observe(PluginHooks):
        async def before_tool(self, e):
            calls.append(e.name)

    for initial, args in [(frozenset(), {"text": "ok"}), (None, {})]:
        agent, _ = build(
            tmp_path,
            [Observe],
            initial=initial,
            turns=[tool_turn(args=args), ScriptedTurn(text="done")],
        )
        result = next(
            e.result for e in await drain(agent) if isinstance(e, ToolResultEvent)
        )
        assert not result.ok
    assert calls == []


async def test_after_patch_chain_commits_only_final_content(tmp_path):
    class Replace(PluginHooks):
        async def after_tool(self, e, r):
            assert r.content == "original"
            return ToolResultPatch.replace("redacted")

    class Append(PluginHooks):
        async def after_tool(self, e, r):
            assert r.content == "redacted"
            return ToolResultPatch.append_note("audit note")

    agent, llm = build(
        tmp_path, [Replace, Append], turns=[tool_turn(), ScriptedTurn(text="done")]
    )
    events = await drain(agent)
    result = next(e.result for e in events if isinstance(e, ToolResultEvent))
    assert result.content == "redacted\naudit note"
    assert (result.call_id, result.name, result.ok) == ("c0", "echo", True)
    assert next(m.content for m in llm.calls[1] if m.role == "tool") == result.content
    assert (
        next(m.content for m in agent.memory.messages if m.role == "tool")
        == result.content
    )


@pytest.mark.parametrize("mode", ["block", "ignore"])
@pytest.mark.parametrize("timeout", [False, True])
async def test_hook_failures_contained_and_noticed(tmp_path, mode, timeout):
    class Fail(PluginHooks):
        async def before_tool(self, e):
            if timeout:
                await asyncio.Event().wait()
            raise ValueError("do not echo secrets")

    agent, _ = build(
        tmp_path,
        [Fail],
        mode=mode,
        timeout=0.01,
        turns=[tool_turn(), ScriptedTurn(text="done")],
    )
    events = await drain(agent)
    result = next(e.result for e in events if isinstance(e, ToolResultEvent))
    assert result.ok is (mode == "ignore")
    notice = next(e for e in events if isinstance(e, PluginNotice))
    assert notice.action == "failed"
    assert "do not echo secrets" not in notice.message
    assert isinstance(events[-1], Final)


async def test_start_failure_retried_and_close_reverse_idempotent(tmp_path):
    starts = []
    closed = []

    class First(PluginHooks):
        async def start(self):
            starts.append("first")

        async def aclose(self):
            closed.append("first")

    class Retry(PluginHooks):
        async def start(self):
            starts.append("retry")
            if starts.count("retry") == 1:
                raise ValueError("secret")

        async def aclose(self):
            closed.append("retry")

    agent, llm = build(tmp_path, [First, Retry])
    events = await drain(agent)
    assert isinstance(events[-1], Error) and not agent.memory.messages
    assert not llm.calls
    assert any(isinstance(e, PluginNotice) for e in events)
    assert isinstance((await drain(agent))[-1], Final)
    assert starts == ["first", "retry", "retry"]
    await agent.aclose()
    await agent.aclose()
    assert closed == ["retry", "first"]


async def test_stop_in_hook_finalizes_without_turn_end(tmp_path):
    reached = asyncio.Event()
    ended = []

    class Wait(PluginHooks):
        async def before_tool(self, e):
            reached.set()
            await asyncio.Event().wait()

        async def turn_end(self, e):
            ended.append(e)

    with SessionStore(tmp_path / "sessions.db") as store:
        agent, _ = build(tmp_path, [Wait], store=store, timeout=60, turns=[tool_turn()])
        task = asyncio.create_task(drain(agent))
        await reached.wait()
        assert agent.cancel_turn()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert isinstance(agent.finalize_cancelled_turn(), TurnCancelled)
        assert [m.role for m in agent.memory.messages] == ["user"]
        assert [m.role for m in store.messages(agent._session_id)] == ["user"]
        assert ended == []


async def test_parallel_batch_hooks_share_instance(tmp_path):
    entered = []
    ready = asyncio.Event()

    class Parallel(PluginHooks):
        async def before_tool(self, e):
            entered.append(e.call_id)
            if len(entered) == 2:
                ready.set()
            await ready.wait()

    agent, _ = build(
        tmp_path,
        [Parallel],
        parallel=True,
        turns=[tool_turn(count=2), ScriptedTurn(text="done")],
    )
    events = await drain(agent)
    assert entered == ["c0", "c1"]
    assert len([e for e in events if isinstance(e, ToolResultEvent)]) == 2


async def test_turn_end_failure_notice_precedes_model_error(tmp_path):
    class End(PluginHooks):
        async def turn_end(self, e):
            assert e.error is not None
            raise RuntimeError("secret")

    agent, _ = build(tmp_path, [End], turns=[StreamFailure(retryable=False)])
    events = await drain(agent)
    assert isinstance(events[-2], PluginNotice)
    assert events[-2].hook == "turn_end"
    assert isinstance(events[-1], Error)


async def test_parallel_hook_cancellation_reaps_sibling_before_finalization(tmp_path):
    sibling_entered = asyncio.Event()
    release_sibling = asyncio.Event()
    side_effects = []

    class Cancel(PluginHooks):
        async def before_tool(self, e):
            if e.call_id == "c0":
                await sibling_entered.wait()
                raise asyncio.CancelledError()
            sibling_entered.set()
            await release_sibling.wait()

    reg = ToolRegistry()

    @tool(name="echo", registry=reg)
    async def echo(args: EchoArgs, ctx: ToolContext) -> str:
        side_effects.append(args.text)
        return args.text

    agent, _ = build(
        tmp_path,
        [Cancel],
        registry=reg,
        parallel=True,
        timeout=60,
        turns=[tool_turn(count=2)],
    )
    with pytest.raises(asyncio.CancelledError):
        await drain(agent)
    assert isinstance(agent.finalize_cancelled_turn(), TurnCancelled)
    release_sibling.set()
    # A task that escaped the batch will reach its side effect on the next tick.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert side_effects == []


async def test_cancelled_turn_notices_drain_before_terminal_and_do_not_leak(tmp_path):
    entered = asyncio.Event()

    class Ask(PluginHooks):
        async def before_tool(self, e):
            return ToolDecision.ask("Proceed?")

    class Wait(PluginHooks):
        async def before_tool(self, e):
            entered.set()
            await asyncio.Event().wait()

    async def confirm(prompt):
        return True

    agent, _ = build(
        tmp_path,
        [Ask, Wait],
        confirm=confirm,
        timeout=60,
        turns=[tool_turn(), ScriptedTurn(text="next")],
    )
    task = asyncio.create_task(drain(agent))
    await entered.wait()
    agent.cancel_turn()
    with pytest.raises(asyncio.CancelledError):
        await task
    notices = agent.drain_plugin_notices()
    assert len(notices) == 1 and notices[0].action == "asked"
    assert isinstance(agent.finalize_cancelled_turn(), TurnCancelled)
    events = await drain(agent)
    assert isinstance(events[-1], Final)
    assert not any(isinstance(event, PluginNotice) for event in events)


async def test_from_profile_wires_prompt_commands_and_per_agent_instances(
    tmp_path, monkeypatch
):
    import yaml

    from lingcore.config import AgentProfile
    from lingcore.plugins.scaffold import scaffold_plugin

    monkeypatch.setattr(REGISTRY, "_tools", dict(REGISTRY._tools))
    root = scaffold_plugin(tmp_path, "per-agent")
    code = root / "plugin.py"
    code.write_text(
        code.read_text().replace(
            "return args.text", 'return str(ctx.plugins["per-agent"].ctx.session_id)'
        )
    )
    manifest_path = root / "plugin.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["prompt"] = "prompt.md"
    manifest_path.write_text(yaml.safe_dump(manifest))
    (root / "prompt.md").write_text("Always use the per-agent policy.")
    profile = AgentProfile(
        llm={"model": "fake"},
        workspace=str(tmp_path),
        plugins=["per-agent"],
        tools=["per_agent_echo"],
    )
    profile._source_dir = tmp_path
    agents = []
    for sid in ["one", "two"]:
        call = ToolCall(id="c", name="per_agent_echo", arguments={"text": "original"})
        llm = FakeLLMClient(
            [ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done")]
        )
        agent = Agent.from_profile(profile, llm=llm, session_id=sid)
        incoming = agent.commands.resolve("/per-agent:hello target", reserved=())
        assert incoming is not None
        events = await drain(agent, incoming)
        result = next(e.result for e in events if isinstance(e, ToolResultEvent))
        assert result.content == sid
        assert "Always use the per-agent policy." in llm.calls[0][0].content
        assert (
            next(m for m in agent.memory.messages if m.role == "user").input_text
            == "/per-agent:hello target"
        )
        agents.append(agent)
    assert (
        agents[0].tool_ctx.plugins["per-agent"]
        is not agents[1].tool_ctx.plugins["per-agent"]
    )
    await asyncio.gather(*(agent.aclose() for agent in agents))


async def test_enabled_plugin_never_widens_an_empty_tools_ceiling(
    tmp_path, monkeypatch
):
    from lingcore.config import AgentProfile
    from lingcore.plugins.scaffold import scaffold_plugin

    monkeypatch.setattr(REGISTRY, "_tools", dict(REGISTRY._tools))
    scaffold_plugin(tmp_path, "no-ceiling")
    profile = AgentProfile(
        llm={"model": "fake"}, workspace=str(tmp_path), plugins=["no-ceiling"], tools=[]
    )
    profile._source_dir = tmp_path
    llm = FakeLLMClient(
        [tool_turn("no_ceiling_echo", {"text": "forbidden"}), ScriptedTurn(text="done")]
    )
    async with Agent.from_profile(profile, llm=llm) as agent:
        events = await drain(agent)
        result = next(e.result for e in events if isinstance(e, ToolResultEvent))
        assert not result.ok and "unknown tool" in result.content
        assert llm.tool_schemas == [[], []]
        assert not any(isinstance(e, PluginNotice) for e in events)


async def test_stop_during_turn_end_keeps_committed_reply(tmp_path):
    reached = asyncio.Event()

    class SlowObserver(PluginHooks):
        async def turn_end(self, e):
            if not reached.is_set():
                reached.set()
                await asyncio.Event().wait()

    with SessionStore(tmp_path / "sessions.db") as store:
        agent, _ = build(tmp_path, [SlowObserver], store=store, timeout=60)
        task = asyncio.create_task(drain(agent))
        await reached.wait()
        assert agent.cancel_turn()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.turn_pending_finalization
        terminal = agent.finalize_cancelled_turn()
        assert terminal == Final("done")
        assert [m.role for m in agent.memory.messages] == ["user", "assistant"]
        assert [m.role for m in store.messages(agent._session_id)] == [
            "user",
            "assistant",
        ]
        assert not agent.turn_pending_finalization
        assert isinstance((await drain(agent))[-1], Final)


async def test_display_text_passes_the_guardrail_before_storage(tmp_path):
    class Redact:
        async def pre_input(self, text: str) -> str:
            return text.replace("hunter2", "[redacted]")

        async def post_output(self, text: str) -> str:
            return text

    agent, llm = build(tmp_path, [])
    agent.guardrail = Redact()
    await drain(
        agent,
        UserInput(text="Review hunter2 now", display_text="/review hunter2"),
    )
    stored = agent.memory.messages[0]
    assert stored.input_text == "/review [redacted]"
    assert "hunter2" not in stored.content
