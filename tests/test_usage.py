"""Provider usage: parsing, LLMClient reporting, and loop emission."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import lingcore.tools.builtin  # noqa: F401  (registers builtins)
from lingcore.agent import Agent
from lingcore.composer import StaticComposer
from lingcore.config import AgentProfile
from lingcore.errors import LLMStreamError
from lingcore.events import Final, TextDelta, ToolResultEvent, UsageReported
from lingcore.llm import LLMChunk, LLMClient
from lingcore.memory import WindowMemory
from lingcore.message import ToolCall
from lingcore.tools import REGISTRY, ToolContext, ToolRegistry
from lingcore.usage import TokenUsage, UsageMeter, usage_from_openai
from tests.fakes import _Choice, _Delta, _Event, make_openai_stream


def _usage(**kw):
    base = {"prompt_tokens": 100, "completion_tokens": 20}
    base.update(kw)
    return SimpleNamespace(**base)


def test_usage_parses_openai_details_and_deepseek_cache_hits() -> None:
    parsed = usage_from_openai(
        "gpt",
        _usage(
            prompt_tokens_details=SimpleNamespace(cached_tokens=60),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )
    assert parsed == TokenUsage("gpt", 100, 20, 60, 5)
    deepseek = usage_from_openai(
        "deepseek",
        {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 7},
    )
    assert deepseek == TokenUsage("deepseek", 10, 2, 7, 0)
    assert usage_from_openai("m", None) is None


def test_usage_parser_clamps_malformed_counts() -> None:
    parsed = usage_from_openai(
        "m",
        {
            "prompt_tokens": -3,
            "completion_tokens": True,
            "prompt_tokens_details": {"cached_tokens": 50},
            "completion_tokens_details": {"reasoning_tokens": "9"},
        },
    )
    assert parsed == TokenUsage("m", 0, 0, 0, 0)


def _client(**kw) -> LLMClient:
    return LLMClient(model="alias", api_key="sk-test", base_url="http://x/v1", **kw)


async def test_client_requests_usage_and_reports_served_model(monkeypatch) -> None:
    seen: list[TokenUsage] = []
    client = _client(usage_sink=seen.append)
    kwargs: dict = {}

    async def fake_create(**kw):
        kwargs.update(kw)
        return make_openai_stream(
            [
                _Event([_Choice(_Delta(content="hi"))], model="alias-2026"),
                _Event([_Choice(_Delta(), finish_reason="stop")], model="alias-2026"),
                _Event([], usage=_usage(), model="alias-2026"),
            ]
        )

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    chunks = [c async for c in client.stream([])]
    assert kwargs["stream_options"] == {"include_usage": True}
    assert chunks[-1].finish_reason == "stop"
    assert seen == [TokenUsage("alias-2026", 100, 20)]


async def test_stream_usage_opt_out_omits_stream_options(monkeypatch) -> None:
    client = _client(stream_usage=False)
    kwargs: dict = {}

    async def fake_create(**kw):
        kwargs.update(kw)
        return make_openai_stream([_Event([_Choice(_Delta(), finish_reason="stop")])])

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    [c async for c in client.stream([])]
    assert "stream_options" not in kwargs


async def test_truncated_stream_still_reports_billed_usage(monkeypatch) -> None:
    seen: list[TokenUsage] = []
    client = _client(usage_sink=seen.append)

    async def fake_open(messages, tools):
        return make_openai_stream(
            [_Event([_Choice(_Delta(content="par"))]), _Event([], usage=_usage())]
        )

    monkeypatch.setattr(client, "_open_stream", fake_open)
    with pytest.raises(LLMStreamError, match="truncated"):
        [c async for c in client.stream([])]
    assert seen == [TokenUsage("alias", 100, 20)]


class _MeteredLLM:
    """Scripted client that reports usage into the agent's meter."""

    def __init__(self, turns: list[list[LLMChunk]], meter: UsageMeter) -> None:
        self.turns, self.meter, self.calls = list(turns), meter, 0

    async def stream(self, messages, tools=None):
        self.calls += 1
        for chunk in self.turns.pop(0):
            yield chunk
        self.meter.record(TokenUsage("m", 10 * self.calls, self.calls))


def _agent(llm, meter: UsageMeter, workspace: Path) -> Agent:
    tools = ToolRegistry()
    tools.register(REGISTRY.get("read_file"))
    return Agent(
        llm=llm,
        tools=tools,
        tool_ctx=ToolContext(workspace=workspace),
        composer=StaticComposer("s"),
        memory=WindowMemory(model="gpt-4o"),
        usage_meter=meter,
    )


async def test_loop_emits_usage_per_request_before_terminal_event(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    meter = UsageMeter()
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    llm = _MeteredLLM(
        [
            [LLMChunk(tool_calls=[call], finish_reason="tool_calls")],
            [LLMChunk(text_delta="done"), LLMChunk(finish_reason="stop")],
        ],
        meter,
    )
    events = [e async for e in _agent(llm, meter, tmp_path).run("go")]
    kinds = [type(e).__name__ for e in events]
    usage = [e.usage for e in events if isinstance(e, UsageReported)]
    assert usage == [TokenUsage("m", 10, 1), TokenUsage("m", 20, 2)]
    assert kinds.index("UsageReported") < kinds.index("ToolCallStarted")
    assert isinstance(events[-1], Final)
    assert isinstance(events[-2], UsageReported)
    assert any(isinstance(e, ToolResultEvent) for e in events)


async def test_usage_recorded_before_stop_is_drained_after_finalize(tmp_path) -> None:
    meter = UsageMeter()
    gate = asyncio.Event()

    class Blocking:
        async def stream(self, messages, tools=None):
            meter.record(TokenUsage("m", 5, 0))
            yield LLMChunk(text_delta="partial")
            gate.set()
            await asyncio.Event().wait()

    agent = _agent(Blocking(), meter, tmp_path)
    seen: list = []

    async def drive():
        async for event in agent.run("go"):
            seen.append(event)

    task = asyncio.create_task(drive())
    await gate.wait()
    assert agent.cancel_turn()
    with pytest.raises(asyncio.CancelledError):
        await task
    agent.finalize_cancelled_turn()
    # The first TextDelta flushed the usage recorded before it.
    assert [type(e) for e in seen] == [UsageReported, TextDelta]
    meter.record(TokenUsage("m", 1, 1))
    assert [e.usage for e in agent.drain_usage()] == [TokenUsage("m", 1, 1)]
    assert agent.drain_usage() == []


def test_from_profile_meters_main_client(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    (tmp_path / "config.yaml").write_text(
        "name: t\nllm:\n  model: m\n  api_key_env: TEST_KEY\n  stream_usage: false\n",
        encoding="utf-8",
    )
    agent = Agent.from_profile(
        AgentProfile.load(tmp_path / "config.yaml"), base_dir=tmp_path
    )
    assert isinstance(agent.llm, LLMClient)
    assert agent.llm._stream_usage is False
    assert agent.llm.usage_sink == agent.usage_meter.record
