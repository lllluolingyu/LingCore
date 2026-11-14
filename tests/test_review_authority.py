from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lingcore.agent import Agent
from lingcore.delegation.models import RunRecord
from lingcore.errors import ToolError
from lingcore.superagent.authority import constraints
from lingcore.superagent.core.models import Effect
from lingcore.superagent.panel import Panel
from lingcore.superagent.runtime import Runtime
from tests.fakes import FakeLLMClient
from tests.support.workflows import result
from tests.test_delegation_kernel import profile as delegation_profile
from tests.test_superagent_core import accept, engine
from tests.test_superagent_review import accepted, submitted
from tests.test_superagent_runtime import Channel, profile, trigger


async def test_executor_metadata_does_not_read_global_registry_at_launch(
    tmp_path, monkeypatch
):
    from lingcore.delegation.models import WorkerReport
    from lingcore.tools import REGISTRY

    runtime = Runtime(profile(tmp_path), llm_factory=lambda *a: result())
    try:
        effect = await accepted(runtime)
        runtime.executor_registry.register(
            replace(runtime.executor_registry.get("read_file"), high_risk=True)
        )
        envelopes = []

        async def worker(handle, prompt, role, envelope, **kwargs):
            envelopes.append(envelope)
            return WorkerReport(
                worker_id="w-test",
                role=role,
                label="test",
                status="completed",
                result="done",
            )

        def forbidden(name):
            raise AssertionError("launch consulted global tool metadata")

        monkeypatch.setattr(runtime.supervisor, "worker", worker)
        monkeypatch.setattr(REGISTRY, "get", forbidden)
        await runtime.launch(effect)
        assert envelopes[0].high_risk == {"read_file"}
    finally:
        await runtime.close()


@pytest.mark.parametrize("name", ["run_shell", "write_file", "edit_file", "patch_file"])
async def test_small_mutating_delegate_still_requires_confirmation(tmp_path, name):
    agent = Agent.from_profile(
        delegation_profile(tmp_path, tools=["delegate", name]), llm=FakeLLMClient([])
    )
    async with agent:
        supervisor = agent.tool_ctx.delegation
        envelope = supervisor.envelope("general", frozenset({name}))
        assert name in envelope.high_risk
        with pytest.raises(ToolError, match="confirmation denied"):
            await supervisor.confirm_run(supervisor.new_run(), {"general": envelope}, 1)


async def test_risk_metadata_uses_scoped_registry_and_configured_skill_floor(tmp_path):
    p = delegation_profile(tmp_path, tools=["delegate", "read_file"])
    agent = Agent.from_profile(p, llm=FakeLLMClient([]))
    async with agent:
        supervisor = agent.tool_ctx.delegation
        agent.tools.register(replace(agent.tools.get("read_file"), high_risk=True))
        assert (
            "read_file"
            in supervisor.envelope("general", frozenset({"read_file"})).high_risk
        )
        agent.tools.register(replace(agent.tools.get("read_file"), high_risk=False))
        agent.tool_ctx.options["activate_skill"] = {"high_risk_tools": ["read_file"]}
        assert (
            "read_file"
            in supervisor.envelope("general", frozenset({"read_file"})).high_risk
        )


@pytest.mark.parametrize("stop", [False, True])
async def test_launch_cancellation_settles_old_effect_once(tmp_path, stop):
    started = asyncio.Event()

    class Waiting(FakeLLMClient):
        async def stream(self, messages, tools=None):
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover

    runtime = Runtime(profile(tmp_path), llm_factory=lambda *a: Waiting([]))
    try:
        effect = await accepted(runtime)
        task = asyncio.create_task(runtime.runner._run(effect))
        await asyncio.wait_for(started.wait(), 2)
        ticket = next(iter(runtime.store.read().tickets.values()))
        if stop:
            await runtime.ingest(trigger().model_copy(update={"external_id": "stop"}))
            evaluation = next(
                e
                for e in runtime.store.read().evaluations.values()
                if "lark:stop" in e.triggers
            )
            runtime.command(
                lambda e: e.route(
                    evaluation.id,
                    [
                        {
                            "action": "stop",
                            "trigger": "lark:stop",
                            "ticket_id": ticket.id,
                        }
                    ],
                    e.router_tools,
                    time.time(),
                )
            )
            await runtime.delivery(
                Effect(
                    id="stop",
                    kind="delivery",
                    payload={"ticket_id": ticket.id, "stop": True},
                    created_at=0,
                    due_at=0,
                )
            )
        else:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        state = runtime.store.read()
        assert state.effects[effect.id].status == "done"
        assignments = state.tickets[ticket.id].assignments
        if stop:
            assert (
                len(assignments) == 1 and assignments[0].result == "stopped by person"
            )
            assert (
                runtime.supervisor.journal.get(assignments[0].run_id).status
                == "cancelled"
            )
        else:
            assert len(assignments) == 2 and assignments[-1].recovery
    finally:
        await runtime.close()


async def test_terminal_launch_error_does_not_retry_the_effect(tmp_path, monkeypatch):
    runtime = Runtime(profile(tmp_path), llm_factory=lambda *a: result())
    try:
        effect = await accepted(runtime)

        async def failed(*a, **kw):
            raise ToolError("terminal worker failure")

        monkeypatch.setattr(runtime.supervisor, "worker", failed)
        await runtime.runner._run(effect)
        state = runtime.store.read()
        assert state.effects[effect.id].status == "done"
        assert next(iter(state.tickets.values())).assignments[-1].result == "failed"
        assert not runtime.status()["health"]["errors"]
    finally:
        await runtime.close()


async def test_restart_reuses_router_and_recovers_legacy_ticket_runs(tmp_path):
    p = profile(tmp_path)
    runtime = Runtime(p, llm_factory=lambda *a: result())
    effect = await accepted(runtime)
    old_session = runtime.router._session_id
    runtime.supervisor.journal.save(
        RunRecord(
            id="r-orphan",
            kind="ticket",
            status="running",
            parent_session_id="old-router",
        )
    )
    runtime.command(
        lambda e: e.bind_worker(
            effect.payload["ticket_id"],
            effect.payload["assignment_id"],
            "r-orphan",
            None,
            None,
        )
    )
    await runtime.close()
    runtime = Runtime(p, llm_factory=lambda *a: result())
    try:
        assert runtime.router._session_id == old_session
        assert runtime.supervisor.journal.get("r-orphan").status == "interrupted"
    finally:
        await runtime.close()


def test_approval_can_narrow_tools_and_respects_current_role():
    e = engine()
    e.ceiling |= {"edit_file"}
    e.roles["executor"] |= {"edit_file"}
    ticket = accept(e)
    e.approve(
        ticket.id,
        "approver",
        frozenset({"write_file"}),
        ticket.mandate.envelope_digest,
        True,
        1,
    )
    assert e.launch(ticket.id, ticket.assignments[-1].id, 2)
    assert e.grant(ticket) == {"read_file", "write_file"}
    e = engine()
    ticket = accept(e)
    e.roles["executor"] = frozenset({"read_file"})
    e.approve(
        ticket.id,
        "approver",
        frozenset({"write_file"}),
        ticket.mandate.envelope_digest,
        True,
        1,
    )
    assert not ticket.approval.tools
    assert e.launch(ticket.id, ticket.assignments[-1].id, 2)


async def test_panel_rechecks_current_constraints_and_environment_binding(tmp_path):
    p = profile(tmp_path, ["write_file"])
    p.superagent.panel_actor = "approver"
    runtime = Runtime(p, llm_factory=lambda *a: result(), channels={"lark": Channel()})
    try:
        await runtime.ingest(trigger())
        evaluation = next(iter(runtime.store.read().evaluations.values()))
        runtime.command(
            lambda e: e.route(
                evaluation.id,
                [{"action": "accept", "trigger": "lark:1"}],
                e.router_tools,
                time.time(),
            )
        )
        effect = next(
            e
            for e in runtime.store.read().effects.values()
            if e.payload.get("approval")
        )
        await runtime.reply(effect)
        card = runtime.store.read().effects[effect.id].payload["card"]
        before = constraints(p, runtime.router.tool_ctx)
        runtime.router.tool_ctx.environment_binding = SimpleNamespace(
            binding_id="changed"
        )
        assert constraints(p, runtime.router.tool_ctx) != before
        runtime.command(lambda e: None)
        with pytest.raises(ToolError, match="no longer admitted"):
            Panel(p).action(effect.payload["ticket_id"], "approve", token=card["token"])
    finally:
        await runtime.close()


async def test_final_report_delivers_once_after_worker_finishes(tmp_path):
    from lingcore.message import ToolCall
    from tests.fakes import ScriptedTurn

    worker = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="report",
                        name="ticket_report",
                        arguments={"status": "ready", "summary": "finished"},
                    )
                ]
            ),
            submitted("finished"),
        ]
    )
    runtime = Runtime(profile(tmp_path), llm_factory=lambda *a: worker)
    try:
        await runtime.launch(await accepted(runtime))
        effects = runtime.store.read().effects.values()
        assert not any(e.kind == "relay" for e in effects)
        assert len([e for e in effects if e.kind == "reply"]) == 1
    finally:
        await runtime.close()


async def test_delivered_card_without_conversation_is_not_sent_again(tmp_path):
    class MissingReceipt(Channel):
        async def send(self, *a, **kw):
            await super().send(*a, **kw)
            return ""

    channel = MissingReceipt()
    runtime = Runtime(
        profile(tmp_path, ["write_file"]),
        llm_factory=lambda *a: result(),
        channels={"lark": channel},
    )
    try:
        await runtime.ingest(trigger())
        evaluation = next(iter(runtime.store.read().evaluations.values()))
        runtime.command(
            lambda e: e.route(
                evaluation.id,
                [{"action": "accept", "trigger": "lark:1"}],
                e.router_tools,
                time.time(),
            )
        )
        effect = next(
            e
            for e in runtime.store.read().effects.values()
            if e.payload.get("approval")
        )
        await runtime.runner._run(effect)
        saved = runtime.store.read().effects[effect.id]
        assert saved.status == "done" and len(channel.sent) == 1
        assert saved.payload["card"]["conversation"] == ""
        assert saved.payload["card"]["delivery_warning"]
    finally:
        await runtime.close()
