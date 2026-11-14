from __future__ import annotations

import asyncio
import json

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.delegation.schema import compile_schema
from lingcore.errors import ConfigError
from lingcore.events import DelegatedUsageReported, ToolResultEvent
from lingcore.message import ToolCall
from lingcore.sessions import SessionStore
from lingcore.usage import TokenUsage
from tests.fakes import FakeLLMClient, ScriptedTurn


def test_result_enum_keeps_declared_type_and_other_constraints():
    from pydantic import ValidationError

    adapter = compile_schema(
        {"type": "string", "maxLength": 2, "enum": ["ok", "long", 1]}
    )
    assert adapter.validate_python("ok", strict=True) == "ok"
    for invalid in ("long", 1, True, "no"):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid, strict=True)
    with pytest.raises(ValidationError):
        compile_schema({"enum": [1]}).validate_python(True, strict=True)


def profile(tmp_path, **updates):
    p = AgentProfile.model_validate(
        {
            "name": "test",
            "llm": {"model": "fake"},
            "workspace": str(tmp_path / "ws"),
            "tools": ["delegate", "read_file"],
            **updates,
        }
    )
    object.__setattr__(p, "_source_dir", tmp_path)
    return p


async def test_delegate_runs_ordinary_worker_and_accounts_separately(tmp_path):
    parent = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="d", name="delegate", arguments={"prompt": "read carefully"}
                    )
                ]
            ),
            ScriptedTurn(text="received"),
        ]
    )
    worker = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="s",
                        name="submit_result",
                        arguments={"status": "done", "result": "result"},
                    )
                ],
                usage=TokenUsage("worker", 10, 3),
            )
        ]
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        agent = Agent.from_profile(profile(tmp_path), llm=parent, session_store=store)
        agent.tool_ctx.delegation.llm_factory = lambda role, label: worker
        async with agent:
            events = [e async for e in agent.run("original request")]
            results = [e.result for e in events if isinstance(e, ToolResultEvent)]
            assert results[0].ok, results[0].content
            report = json.loads(results[0].content)
            assert report["status"] == "completed", report
            assert report["result"] == "result"
            assert any(isinstance(e, DelegatedUsageReported) for e in events)
            assert "original request" in worker.calls[0][-1].content
            assert not {"delegate", "workflow_run"} & {
                s["function"]["name"] for s in worker.tool_schemas[0]
            }
        assert len(store.list()) == 1 and len(store.list(include_delegated=True)) == 2


async def test_structured_worker_gets_one_nudge(tmp_path):
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "integer"}},
    }
    parent = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="d",
                        name="delegate",
                        arguments={"prompt": "answer", "output": schema},
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    worker = FakeLLMClient(
        [
            ScriptedTurn(text="forgot"),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="s",
                        name="submit_result",
                        arguments={"status": "done", "result": {"answer": 7}},
                    )
                ]
            ),
        ]
    )
    agent = Agent.from_profile(profile(tmp_path), llm=parent)
    agent.tool_ctx.delegation.llm_factory = lambda role, label: worker
    async with agent:
        events = [e async for e in agent.run("go")]
    report = json.loads(
        next(e.result.content for e in events if isinstance(e, ToolResultEvent))
    )
    assert report["result"] == {"answer": 7}, report
    assert len(worker.calls) == 2


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "bad"},
        {"type": "array", "items": {"type": "string"}, "maxItems": 5000},
        {"type": "string", "pattern": ".*"},
    ],
)
def test_schema_refuses_unbounded_or_executable_features(schema):
    with pytest.raises(ConfigError):
        compile_schema(schema)


async def test_nested_workers_share_caps_and_spend_without_writer_deadlock(tmp_path):
    from tests.support.workflows import build

    agent, supervisor = build(
        tmp_path,
        roles={
            "delegator": {"tools": ["delegate", "read_file"], "context": []},
            "pure": {"tools": [], "context": []},
        },
    )
    supervisor.profile.delegation.max_depth = 2
    outer = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="d",
                        name="delegate",
                        arguments={"role": "pure", "prompt": "inner"},
                    )
                ],
                usage=TokenUsage("fake", 10, 2),
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="s",
                        name="submit_result",
                        arguments={"status": "done", "result": "outer"},
                    )
                ],
                usage=TokenUsage("fake", 20, 3),
            ),
        ]
    )
    inner = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="s",
                        name="submit_result",
                        arguments={"status": "done", "result": "inner"},
                    )
                ],
                usage=TokenUsage("fake", 30, 4),
            )
        ]
    )
    supervisor.llm_factory = lambda role, label: outer if role == "delegator" else inner
    async with agent:
        handle = supervisor.new_run()
        envelope = supervisor.envelope("delegator", agent._authorized_tool_names())
        report = await asyncio.wait_for(
            supervisor.execute(
                handle,
                lambda: supervisor.worker(handle, "outer", "delegator", envelope),
            ),
            5,
        )
        assert report.status == "completed", report
        assert handle.record.agents == 2
        assert handle.record.usage.input_tokens == 60
        assert handle.record.usage.output_tokens == 9
        assert handle.record.usage.requests == 3
        assert len(agent.usage_meter.drain_delegated()) == 3
