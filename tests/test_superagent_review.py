from __future__ import annotations

import asyncio
import time

from lingcore.message import ToolCall
from lingcore.superagent.core.engine import Engine
from lingcore.superagent.core.maintenance import prune
from lingcore.superagent.runtime import Runtime
from lingcore.superagent.store import SuperagentStore
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.test_superagent_core import accept, engine
from tests.test_superagent_runtime import profile, trigger


def submitted(value):
    return ScriptedTurn(
        tool_calls=[
            ToolCall(
                id="done",
                name="submit_result",
                arguments={"status": "done", "result": value},
            )
        ]
    )


async def accepted(runtime):
    await runtime.ingest(trigger())
    evaluation = next(iter(runtime.store.read().evaluations.values()))
    runtime.command(
        lambda e: e.route(
            evaluation.id,
            [{"action": "accept", "trigger": "lark:1"}],
            frozenset({"read_file"}),
            time.time(),
        )
    )
    return next(e for e in runtime.store.read().effects.values() if e.kind == "launch")


async def test_router_rollback_cannot_cancel_or_detach_running_ticket(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    class Waiting(FakeLLMClient):
        async def stream(self, messages, tools=None):
            started.set()
            await release.wait()
            async for chunk in super().stream(messages, tools):
                yield chunk

    runtime = Runtime(
        profile(tmp_path), llm_factory=lambda *a: Waiting([submitted("done")])
    )
    try:
        effect = await accepted(runtime)
        runtime.router._turn_started_at = 20
        task = asyncio.create_task(runtime.launch(effect))
        await asyncio.wait_for(started.wait(), 2)
        runtime.supervisor.rollback(20)
        runtime.sessions.truncate_after(runtime.router._session_id, -1)
        await asyncio.sleep(0)
        ticket = next(iter(runtime.store.read().tickets.values()))
        assert not task.done() and ticket.assignments[-1].status == "running"
        release.set()
        await task
        ticket = next(iter(runtime.store.read().tickets.values()))
        assert ticket.assignments[-1].result == "ready"
        assert (
            runtime.supervisor.journal.get(ticket.assignments[-1].run_id).status
            == "completed"
        )
    finally:
        release.set()
        await runtime.close()


async def test_progress_cannot_replace_later_terminal_result(tmp_path):
    worker = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="p",
                        name="ticket_report",
                        arguments={"status": "progress", "summary": "early"},
                    )
                ]
            ),
            submitted("finished work"),
        ]
    )
    runtime = Runtime(profile(tmp_path), llm_factory=lambda *a: worker)
    try:
        await runtime.launch(await accepted(runtime))
        ticket = next(iter(runtime.store.read().tickets.values()))
        assert ticket.report == "finished work"
        assert ticket.assignments[-1].result == "ready"
    finally:
        await runtime.close()


async def test_followup_reaches_same_executor_before_completion(tmp_path):
    class Followup(FakeLLMClient):
        async def stream(self, messages, tools=None):
            if not self.calls:
                await runtime.ingest(
                    trigger().model_copy(
                        update={
                            "external_id": "2",
                            "text": "Also check the race",
                            "addressed": False,
                        }
                    )
                )
            async for chunk in super().stream(messages, tools):
                yield chunk

    worker = Followup([submitted("first"), submitted("followup complete")])
    runtime = Runtime(profile(tmp_path), llm_factory=lambda *a: worker)
    try:
        await runtime.launch(await accepted(runtime))
        ticket = next(iter(runtime.store.read().tickets.values()))
        assert len(worker.calls) == 2
        assert "Also check the race" in worker.calls[1][-1].content
        assert ticket.report == "followup complete" and len(ticket.assignments) == 1
        assert ticket.assignments[-1].input_cursor == 2
    finally:
        await runtime.close()


def test_ended_subscriptions_do_not_keep_fanning_out():
    e = engine()
    e.router_tools = frozenset({"read_file", "write_file"})
    ticket = accept(e, snapshot=e.router_tools)
    assignment = e.launch(ticket.id, ticket.assignments[-1].id, 1)
    e.report(ticket.id, assignment.id, "ready", "done", None, True, 2)
    effects = len(e.state.effects)
    e.ingest(
        ticket.mandate.trigger.model_copy(
            update={"external_id": "new", "addressed": False}
        ),
        3,
    )
    assert ticket.status == "ended" and len(ticket.inputs) == 1
    assert len(e.state.effects) == effects


def test_failed_routing_decision_stays_available_for_correction():
    e = engine()
    e.ingest(trigger(), 1)
    evaluation = next(iter(e.state.evaluations.values()))
    bad = e.route(
        evaluation.id, [{"action": "typo", "trigger": "lark:1"}], e.router_tools, 2
    )
    assert bad[0]["status"] == "error" and evaluation.status != "ended"
    good = e.route(
        evaluation.id, [{"action": "accept", "trigger": "lark:1"}], e.router_tools, 3
    )
    assert good[0]["status"] == "ok" and evaluation.status == "ended"


def test_store_skips_noop_writes_and_updates_only_changed_rows(tmp_path):
    template = engine()

    def factory(state):
        return Engine(
            state,
            template.config,
            ceiling=template.ceiling,
            roles=template.roles,
            router_tools=template.router_tools,
            constraints_digest=template.constraints,
        )

    store = SuperagentStore(tmp_path / "store.db")
    other = SuperagentStore(tmp_path / "store.db")
    try:
        for index in range(20):
            store.transact(factory, lambda e: e.emit("reply", {"text": str(index)}, 1))
        writes = []
        store.conn.set_trace_callback(
            lambda sql: (
                writes.append(sql)
                if sql.startswith(("INSERT", "UPDATE", "DELETE"))
                else None
            )
        )
        store.transact(factory, lambda e: e.tick(2))
        assert not writes
        effect = next(iter(store.read().effects))
        store.transact(factory, lambda e: e.finish_effect(effect, 3, None))
        assert len(writes) == 3  # state, that effect, and that outbox row
        other.transact(factory, lambda e: e.checkpoint("source", "external"))
        assert store.read().source_positions["source"] == "external"
        store.transact(factory, lambda e: e.state.effects.clear())
        assert store.conn.execute("SELECT count(*) FROM effects").fetchone()[0] == 0
        assert store.conn.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
    finally:
        other.close()
        store.close()


def test_retention_bounds_history_and_never_stores_rejected_message_bodies(tmp_path):
    e = engine()
    e.config.max_history = 100
    for index in range(200):
        e.ingest(
            trigger().model_copy(
                update={
                    "external_id": str(index),
                    "actor": "stranger",
                    "text": "private rejected body",
                }
            ),
            index,
        )
    assert not e.state.triggers and len(e.state.seen) == 100
    e.config.max_closed_tickets = 1
    e.router_tools = frozenset({"read_file", "write_file"})
    for index in [201, 202]:
        ticket = accept(e, index=index, snapshot=e.router_tools)
        assignment = e.launch(ticket.id, ticket.assignments[-1].id, index)
        e.report(ticket.id, assignment.id, "ready", "done", None, True, index + 1)
    for effect in e.state.effects.values():
        effect.status = "done"
    prune(e, 300)
    assert len(e.state.tickets) == 1


async def test_configured_executor_capacity_includes_reserved_followup_slot(tmp_path):
    p = profile(tmp_path)
    p.superagent.executor.max_workers = 5
    started, release = asyncio.Event(), asyncio.Event()
    count = 0

    class Waiting(FakeLLMClient):
        async def stream(self, messages, tools=None):
            nonlocal count
            count += 1
            if count == 6:
                started.set()
            await release.wait()
            async for chunk in super().stream(messages, tools):
                yield chunk

    runtime = Runtime(p, llm_factory=lambda *a: Waiting([submitted("done")]))
    try:
        for index in range(6):

            def add(e):
                ticket = accept(e, index=index + 10, snapshot=frozenset({"read_file"}))
                if index == 5:
                    ticket.assignments[-1].followup = True

            runtime.command(add)
        runtime.command(
            lambda e: [
                setattr(x, "status", "done")
                for x in e.state.effects.values()
                if x.kind != "launch"
            ]
        )
        await runtime.runner.once()
        await asyncio.wait_for(started.wait(), 3)
        assert count == 6
        release.set()
        await asyncio.gather(*runtime.runner.tasks.values())
    finally:
        release.set()
        await runtime.close()


async def test_plugin_ask_on_shell_is_denied_before_tool_execution(
    tmp_path, monkeypatch
):
    from lingcore.agent import Agent
    from lingcore.plugins import ToolDecision

    p = profile(tmp_path, ["run_shell"])
    p.tool_options = {"run_shell": {"sandbox": {"backend": "bubblewrap"}}}
    worker = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="shell", name="run_shell", arguments={"command": "true"}
                    )
                ]
            ),
            submitted("done"),
        ]
    )
    runtime = Runtime(p, llm_factory=lambda *a: worker)
    observed = []
    original = Agent.from_profile

    def build(*args, **kwargs):
        agent = original(*args, **kwargs)

        async def before(event):
            if event.name == "run_shell":
                allowed = await agent.tool_ctx.confirm(
                    "Plugin asks for separate consent"
                )
                observed.append(allowed)
                return ToolDecision.deny("plugin approval denied")
            return None

        monkeypatch.setattr(agent._hooks, "before_tool", before)
        return agent

    try:
        await runtime.ingest(trigger())
        evaluation = next(iter(runtime.store.read().evaluations.values()))
        runtime.command(
            lambda e: e.route(
                evaluation.id,
                [{"action": "accept", "trigger": "lark:1"}],
                frozenset(),
                time.time(),
            )
        )
        ticket = next(iter(runtime.store.read().tickets.values()))
        runtime.command(
            lambda e: e.approve(
                ticket.id,
                "approver",
                frozenset({"run_shell"}),
                ticket.mandate.envelope_digest,
                True,
                time.time(),
            )
        )
        monkeypatch.setattr(Agent, "from_profile", build)
        effect = next(
            e for e in runtime.store.read().effects.values() if e.kind == "launch"
        )
        await runtime.launch(effect)
        assert observed == [False]
    finally:
        await runtime.close()
