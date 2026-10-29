from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.events import Final
from lingcore.message import Message, ToolCall
from lingcore.tracing.inspect import inspect_trace
from lingcore.usage import TokenUsage, usage_from_anthropic, usage_from_openai
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure


class Metered(FakeLLMClient):
    agent: Agent
    backend = "openai"

    async def stream(self, messages, tools=None):
        try:
            async for chunk in super().stream(messages, tools):
                yield chunk
        finally:
            if self.backend == "anthropic":
                usage = usage_from_anthropic(
                    "m",
                    {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 4,
                        "cache_creation_input_tokens": 3,
                        "cache_creation": {"ephemeral_5m_input_tokens": 3},
                    },
                )
            else:
                usage = usage_from_openai(
                    "m", {"prompt_tokens": 17, "completion_tokens": 2}
                )
            self.agent.usage_meter.record(usage)


def profile(tmp_path: Path, *, content="metadata", **updates):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    data = {
        "llm": {"model": "m", "stream_retries": 1},
        "workspace": str(tmp_path / "ws"),
        "tools": ["read_file", "write_file"],
        "tracing": {"enabled": True, "content": content},
        **updates,
    }
    result = AgentProfile.model_validate(data)
    object.__setattr__(result, "_source_dir", source)
    return result


def trace_directory(agent):
    return agent.recorder.sink.directory


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
async def test_accounting_all_model_roles(tmp_path, backend):
    raw = Metered([ScriptedTurn(text="done")] * 4)
    raw.backend = backend
    prof = profile(
        tmp_path,
        llm={"model": "m", "backend": backend, "modalities": []},
        memory={"compaction": {"enabled": True}},
        tools=["memory"],
        tool_options={"memory": {"auto_compact": True}},
        media_fallback={"image": {"model": "vision", "backend": backend}},
    )
    agent = Agent.from_profile(prof, llm=raw, vision_llm=raw)
    raw.agent = agent
    from lingcore.tools.builtin.memory import MEMORY_SUMMARIZER_KEY

    for client in [
        agent.llm,
        agent.memory._summarizer,
        agent.tool_ctx.options[MEMORY_SUMMARIZER_KEY],
        agent.media_adapter.vision,
    ]:
        [chunk async for chunk in client.stream([Message.user("input")])]
    assert len(agent.usage_meter.drain()) == 4
    assert {u.purpose for u in agent.request_ledger.records} == {
        "main",
        "summary",
        "memory_condensation",
        "vision",
    }
    assert all(u.availability == "reported" for u in agent.request_ledger.records)
    if backend == "anthropic":
        assert all(
            u.input_tokens == 17 and u.cache_creation_5m_tokens == 3
            for u in agent.request_ledger.records
        )
    await agent.aclose()
    evidence = inspect_trace(trace_directory(agent))
    assert evidence.complete
    assert len([r for r in evidence.records if r.name == "request.usage"]) == 4


async def test_retry_link_missing_usage_and_no_discarded_text(tmp_path, monkeypatch):
    monkeypatch.setattr("lingcore.agent._backoff_seconds", lambda attempt: 0)
    raw = FakeLLMClient(
        [StreamFailure(text="discarded-secret"), ScriptedTurn(text="accepted")]
    )
    agent = Agent.from_profile(profile(tmp_path, content="redacted"), llm=raw)
    events = [e async for e in agent.run("request")]
    assert isinstance(events[-1], Final)
    assert [u.availability for u in agent.request_ledger.records] == [
        "missing",
        "missing",
    ]
    await agent.aclose()
    result = inspect_trace(trace_directory(agent))
    starts = [
        r for r in result.records if r.name == "llm.request" and r.kind == "span.start"
    ]
    assert (
        starts[1].attributes["previous_request_id"]
        == starts[0].attributes["request_id"]
    )
    captured = "".join(p.read_text() for p in trace_directory(agent).rglob("*.json*"))
    assert "discarded-secret" not in captured and "accepted" in captured


async def test_parallel_tools_rejections_and_no_metadata_content(tmp_path):
    calls = [
        ToolCall(
            id="a",
            name="write_file",
            arguments={"path": "a", "content": "private-content"},
        ),
        ToolCall(
            id="b",
            name="write_file",
            arguments={"path": "b", "content": "private-content"},
        ),
        ToolCall(id="c", name="run_shell", arguments={"command": "sensitive"}),
        ToolCall(id="d", name="read_file", arguments={"not_a_path": "private-content"}),
    ]
    raw = Metered(
        [ScriptedTurn(tool_calls=calls), ScriptedTurn(text="private-content")]
    )
    agent = Agent.from_profile(profile(tmp_path), llm=raw)
    raw.agent = agent
    [e async for e in agent.run("private-content")]
    await agent.aclose()
    result = inspect_trace(trace_directory(agent))
    assert result.complete
    executes = [
        r for r in result.records if r.name == "tool.execute" and r.kind == "span.start"
    ]
    assert len(executes) == 2 and len({r.parent_span_id for r in executes}) == 1
    rejected = [
        r for r in result.records if r.name == "tool.dispatch" and r.status == "denied"
    ]
    assert {r.attributes["reason"] for r in rejected} == {
        "authorization",
        "invalid_arguments",
    }
    assert (
        "private-content" not in (trace_directory(agent) / "events.jsonl").read_text()
    )
    assert not (trace_directory(agent) / "artifacts").exists()


async def test_early_closed_request_counts_once(tmp_path):
    raw = Metered([ScriptedTurn(text="hello")])
    agent = Agent.from_profile(profile(tmp_path), llm=raw)
    raw.agent = agent
    stream = agent.llm.stream([Message.user("hi")])
    await anext(stream)
    await stream.aclose()
    await stream.aclose()
    assert len(agent.request_ledger.records) == 1
    assert len(agent.usage_meter.drain()) == 1
    await agent.aclose()


async def test_late_generator_cannot_close_new_turn(tmp_path):
    agent = Agent.from_profile(
        profile(tmp_path),
        llm=FakeLLMClient([ScriptedTurn(text="one"), ScriptedTurn(text="two")]),
    )
    first = agent.run("one")
    async for event in first:
        if isinstance(event, Final):
            break
    second = agent.run("two")
    await anext(second)
    current_turn = agent._turn_trace.context.turn_id
    await first.aclose()
    assert agent._turn_trace.context.turn_id == current_turn
    [e async for e in second]
    await agent.aclose()
    assert inspect_trace(trace_directory(agent)).complete


@pytest.mark.parametrize("destination", ["workspace/traces", ".", "../escape"])
def test_trace_overlap_and_escape_checked_at_assembly(tmp_path, destination):
    prof = profile(tmp_path)
    prof.workspace = str(tmp_path / "source/workspace")
    prof.tracing.path = destination
    with pytest.raises(ConfigError):
        Agent.from_profile(prof, llm=FakeLLMClient([]))


async def test_redacted_capture_matches_commit_and_guarded_final(tmp_path):
    prof = profile(tmp_path, content="redacted")
    object.__setattr__(
        prof, "_profile_env", {"SERVICE_API_KEY": "super-secret-credential"}
    )
    agent = Agent.from_profile(
        prof,
        llm=FakeLLMClient(
            [
                ScriptedTurn(
                    text="committed super-secret-credential", reasoning="hidden-thought"
                )
            ]
        ),
    )

    class Guard:
        async def pre_input(self, text):
            return text

        async def post_output(self, text):
            return "guarded"

    agent.guardrail = Guard()
    [e async for e in agent.run("hello")]
    await agent.aclose()
    artifacts = "".join(
        p.read_text() for p in (trace_directory(agent) / "artifacts").iterdir()
    )
    assert (
        "committed" in artifacts and "guarded" in artifacts and "REDACTED" in artifacts
    )
    assert (
        "super-secret-credential" not in artifacts and "hidden-thought" not in artifacts
    )


async def test_cancel_discards_content_and_retains_usage(tmp_path):
    ready = asyncio.Event()

    class Hanging:
        agent: Agent

        async def stream(self, messages, tools=None):
            self.agent.usage_meter.record(TokenUsage("m", 10, 2))
            from lingcore.llm import LLMChunk

            ready.set()
            yield LLMChunk(text_delta="uncommitted-private")
            await asyncio.Event().wait()

    raw = Hanging()
    agent = Agent.from_profile(profile(tmp_path, content="redacted"), llm=raw)
    raw.agent = agent

    async def drive():
        return [e async for e in agent.run("input")]

    task = asyncio.create_task(drive())
    await ready.wait()
    agent.cancel_turn()
    await asyncio.gather(task, return_exceptions=True)
    agent.finalize_cancelled_turn()
    await agent.aclose()
    result = inspect_trace(trace_directory(agent))
    assert result.complete
    assert len(agent.request_ledger.records) == 1
    assert not any(r.name == "content.capture" for r in result.records)


async def test_fork_links_history_without_copying_usage(tmp_path):
    from lingcore.sessions import SessionStore

    prof = profile(tmp_path)
    with SessionStore(tmp_path / "sessions.db") as store:
        first = Metered([ScriptedTurn(text="one")])
        agent = Agent.from_profile(prof, llm=first, session_store=store)
        first.agent = agent
        [e async for e in agent.run("one")]
        records = store.message_records(agent._session_id)
        original_id = agent._session_id
        fork = store.fork_session(original_id)
        await agent.aclose()
        evidence = inspect_trace(trace_directory(agent))
        turn = next(
            r
            for r in evidence.records
            if r.name == "agent.turn" and r.kind == "span.end"
        )
        assert turn.attributes["session.message_seq"] == records[0].seq
        second = Metered([ScriptedTurn(text="two")])
        child = Agent.from_profile(
            prof, llm=second, session_store=store, session_id=fork.id
        )
        second.agent = child
        [e async for e in child.run("two")]
        await child.aclose()
        evidence = inspect_trace(trace_directory(child))
        assert evidence.complete and len(child.request_ledger.records) == 1
        assert child.recorder.run_id != agent.recorder.run_id
        link = next(r for r in evidence.records if r.name == "session.link")
        assert link.attributes["source_session_id"] == original_id


async def test_stop_during_parallel_tools_closes_every_span(tmp_path):
    from pydantic import BaseModel

    from lingcore.tools import Tool, ToolRegistry

    entered = 0
    ready = asyncio.Event()
    reaped = []

    class Args(BaseModel):
        index: int

    async def wait(args, ctx):
        nonlocal entered
        entered += 1
        if entered == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            reaped.append(args.index)

    registry = ToolRegistry()
    registry.register(Tool("wait", "wait", Args, wait))
    prof = profile(tmp_path, tools=["wait"])
    raw = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id=str(i), name="wait", arguments={"index": i})
                    for i in (1, 2)
                ]
            )
        ]
    )
    agent = Agent.from_profile(prof, llm=raw, tool_registry=registry)

    async def drive():
        return [e async for e in agent.run("go")]

    task = asyncio.create_task(drive())
    await ready.wait()
    agent.cancel_turn()
    await asyncio.gather(task, return_exceptions=True)
    agent.finalize_cancelled_turn()
    await agent.aclose()
    evidence = inspect_trace(trace_directory(agent))
    assert evidence.complete and sorted(reaped) == [1, 2]
    ends = [
        r for r in evidence.records if r.name == "tool.execute" and r.kind == "span.end"
    ]
    assert len(ends) == 2 and all(r.status == "cancelled" for r in ends)


async def test_sync_client_failure_has_one_missing_usage_record(tmp_path):
    class Broken:
        def stream(self, messages, tools=None):
            raise RuntimeError("secret failure")

    agent = Agent.from_profile(profile(tmp_path), llm=Broken())
    [e async for e in agent.run("go")]
    await agent.aclose()
    evidence = inspect_trace(trace_directory(agent))
    assert evidence.complete and len(agent.request_ledger.records) == 1
    assert agent.request_ledger.records[0].availability == "missing"
    assert "secret failure" not in (trace_directory(agent) / "events.jsonl").read_text()
