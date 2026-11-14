from __future__ import annotations

import asyncio
import time

import pytest
import yaml

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, ToolError
from lingcore.message import ToolCall
from lingcore.superagent.core.models import Trigger
from lingcore.superagent.outbox import VerifiedAction
from lingcore.superagent.runtime import Runtime, validate_profile
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.workflows import result


def profile(tmp_path, executor_tools=None):
    (tmp_path / "roles").mkdir(exist_ok=True)
    for name in ["router", "executor", "observer"]:
        tools = (
            executor_tools
            if name == "executor" and executor_tools is not None
            else ["read_file"]
        )
        (tmp_path / "roles" / f"{name}.yaml").write_text(
            yaml.safe_dump({"tools": tools, "context": []})
        )
    p = AgentProfile.model_validate(
        {
            "name": "team-test",
            "llm": {"model": "fake"},
            "tools": ["read_file", "write_file", "run_shell"],
            "workspace": str(tmp_path / "ws"),
            "superagent": {
                "sources": {"lark": {"allowlist": ["person"]}},
                "policy": {"approvers": ["approver"]},
            },
        }
    )
    object.__setattr__(p, "_source_dir", tmp_path)
    return p


def trigger():
    return Trigger(
        source="lark",
        external_id="1",
        actor="person",
        thread="thread",
        address={"chat_id": "chat"},
        text="Read the file",
        addressed=True,
        timestamp=1,
    )


class Channel:
    def __init__(self):
        self.sent = []

    async def send(self, trigger, text, **kwargs):
        self.sent.append((text, kwargs))
        self.destination = trigger.address
        return (
            "approver-chat"
            if trigger.address.get("recipient")
            else trigger.address["chat_id"]
        )


async def test_router_executor_outbox_and_worker_lineage_end_to_end(tmp_path):
    router = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="r",
                        name="ticket_route",
                        arguments={
                            "eval_id": "eval-00000001",
                            "decisions": [{"action": "accept", "trigger": "lark:1"}],
                        },
                    )
                ]
            ),
            ScriptedTurn(text="Routed"),
        ]
    )
    channel = Channel()
    runtime = Runtime(
        profile(tmp_path),
        state_dir=tmp_path,
        llm_factory=lambda role, label: (
            router if role == "router" else result("read complete")
        ),
        channels={"lark": channel},
    )
    try:
        await runtime.ingest(trigger())
        for _ in range(300):
            await runtime.runner.once()
            await asyncio.sleep(0.01)
            if channel.sent:
                break
        assert channel.sent, runtime.store.read().model_dump()
        ticket = next(iter(runtime.store.read().tickets.values()))
        assert ticket.status == "ended" and ticket.assignments[-1].result == "ready"
        assert ticket.assignments[-1].session_id
        assert len(runtime.sessions.list(include_delegated=True)) == 2
        assert len(runtime.sessions.list()) == 1
        assert "ticket_report" not in {
            s["function"]["name"] for s in router.tool_schemas[0]
        }
    finally:
        await runtime.close()


def test_unattended_host_shell_is_refused_at_startup(tmp_path):
    with pytest.raises(ConfigError, match="host runner"):
        validate_profile(profile(tmp_path, ["run_shell"]))


async def test_cards_require_matching_source_conversation_actor_and_nonce(tmp_path):
    runtime = Runtime(
        profile(tmp_path, ["write_file"]),
        state_dir=tmp_path,
        llm_factory=lambda *a: result(),
        channels={"lark": Channel()},
    )
    try:
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
        effect = next(
            e for e in runtime.store.read().effects.values() if e.kind == "reply"
        )
        await runtime.reply(effect)
        assert runtime.channels["lark"].destination == {
            "chat_id": "approver",
            "recipient": "approver",
        }
        card = runtime.store.read().effects[effect.id].payload["card"]
        for actor, token, source, conversation in [
            ("person", card["token"], "lark", "chat"),
            ("approver", "fake", "lark", "chat"),
            ("approver", card["token"], "telegram", "chat"),
            ("approver", card["token"], "lark", "wrong"),
        ]:
            with pytest.raises(ToolError):
                await runtime.action(
                    VerifiedAction(actor, token, True, source, conversation)
                )
        await runtime.action(
            VerifiedAction("approver", card["token"], True, "lark", "approver-chat")
        )
        ticket = next(iter(runtime.store.read().tickets.values()))
        assert ticket.approval.actor == "approver"
    finally:
        await runtime.close()


async def test_only_one_service_owns_state(tmp_path):
    p = profile(tmp_path)
    runtime = Runtime(p, state_dir=tmp_path, llm_factory=lambda *a: result())
    try:
        with pytest.raises(ConfigError, match="already owns"):
            Runtime(p, state_dir=tmp_path, llm_factory=lambda *a: result())
    finally:
        await runtime.close()


async def test_unattended_confirmation_checks_tool_identity_not_prompt(tmp_path):
    from lingcore._agent.dispatch import dispatch_execution, dispatch_tool_name
    from lingcore.delegation.models import WorkerReport

    p = profile(tmp_path, ["run_shell"])
    p.tool_options = {"run_shell": {"sandbox": {"backend": "bubblewrap"}}}
    runtime = Runtime(p, state_dir=tmp_path, llm_factory=lambda *a: result())
    try:
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

        async def worker(*args, **kwargs):
            confirmation = kwargs["confirmation"]
            token = dispatch_tool_name.set("fetch_url")
            try:
                assert not await confirmation("run_shell harmless-looking text")
                dispatch_tool_name.set("run_shell")
                assert not await confirmation("plugin asks on shell")
                phase = dispatch_execution.set(True)
                assert await confirmation("any sandboxed command")
                dispatch_execution.reset(phase)
                dispatch_tool_name.set("activate_skill")
                assert not await confirmation("run_shell")
            finally:
                dispatch_tool_name.reset(token)
            return WorkerReport(
                worker_id="worker",
                role="executor",
                label="executor",
                status="completed",
                result="done",
            )

        runtime.supervisor.worker = worker
        effect = next(
            e for e in runtime.store.read().effects.values() if e.kind == "launch"
        )
        await runtime.launch(effect)
        assert runtime.store.read().tickets[ticket.id].assignments[-1].result == "ready"
    finally:
        await runtime.close()


async def test_process_hold_keeps_assignment_until_limit_then_handback():
    from types import SimpleNamespace as NS

    from lingcore.superagent.lifecycle import hold_processes

    runtime = NS(
        config=NS(executor=NS(background_limit_seconds=0.01)), final_reports={}
    )
    child = NS(
        tool_ctx=NS(plugins={"process": NS(processes=lambda: [NS(state="running")])})
    )
    await hold_processes(runtime, child, "ticket", "assignment")
    assert runtime.final_reports["assignment"]["status"] == "waiting"
    assert runtime.final_reports["assignment"]["question"]
