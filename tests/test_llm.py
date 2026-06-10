"""Tests for LLMClient streaming + tool-call delta reassembly (M1)."""

from __future__ import annotations

import json

import httpx
import pytest
from openai import BadRequestError, InternalServerError, RateLimitError

from lingcore.errors import LLMStreamError
from lingcore.llm import LLMClient
from tests.fakes import (
    _Choice,
    _Delta,
    _Event,
    _Fn,
    _ToolCallDelta,
    make_openai_stream,
)


async def _collect(client: LLMClient, monkeypatch, events):
    """Patch the client's stream opener to return our fabricated events."""

    async def fake_open(messages, tools):
        return make_openai_stream(events)

    monkeypatch.setattr(client, "_open_stream", fake_open)
    return [chunk async for chunk in client.stream(messages=[])]


@pytest.fixture
def client():
    return LLMClient(model="x", api_key="sk-test", base_url="http://localhost/v1")


async def test_text_streaming(client, monkeypatch):
    events = [
        _Event([_Choice(_Delta(content="Hel"))]),
        _Event([_Choice(_Delta(content="lo"))]),
        _Event([_Choice(_Delta(), finish_reason="stop")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    text = "".join(c.text_delta for c in chunks)
    assert text == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].tool_calls is None


async def test_tool_call_fragments_reassembled(client, monkeypatch):
    # A single tool call whose JSON arguments arrive split across chunks.
    events = [
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, id="call_1", function=_Fn(name="read_file")),
        ]))]),
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, function=_Fn(arguments='{"pa')),
        ]))]),
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, function=_Fn(arguments='th": "a.txt"}')),
        ]))]),
        _Event([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    final = chunks[-1]
    assert final.finish_reason == "tool_calls"
    assert final.tool_calls is not None
    assert len(final.tool_calls) == 1
    call = final.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.txt"}


async def test_parallel_tool_calls_kept_separate(client, monkeypatch):
    events = [
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, id="c0", function=_Fn(name="read_file", arguments='{"path":"a"}')),
            _ToolCallDelta(index=1, id="c1", function=_Fn(name="list_dir", arguments='{"path":"."}')),
        ]))]),
        _Event([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    calls = chunks[-1].tool_calls
    assert [c.name for c in calls] == ["read_file", "list_dir"]
    assert calls[0].arguments == {"path": "a"}
    assert calls[1].arguments == {"path": "."}


async def test_malformed_args_become_empty_dict(client, monkeypatch):
    events = [
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, id="c0", function=_Fn(name="x", arguments="{not json")),
        ]))]),
        _Event([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    chunks = await _collect(client, monkeypatch, events)
    assert chunks[-1].tool_calls[0].arguments == {}


# --------------------------------------------------------------------------- #
# Retry — delegated to the SDK, exercised through a mocked HTTP transport.     #
# `retry-after-ms: 1` (1ms) is honored by the SDK, so retries are effectively  #
# instant and the suite never really sleeps.                                   #
# --------------------------------------------------------------------------- #

def _sse(*chunks: dict) -> bytes:
    body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


_OK_STREAM = _sse(
    {"id": "1", "object": "chat.completion.chunk",
     "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
    {"id": "1", "object": "chat.completion.chunk",
     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
)


def _err(status: int, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        json={"error": {"message": "x", "type": "e", "code": None}},
    )


def _mock_client(handler, **kw) -> LLMClient:
    """An LLMClient whose SDK talks to a MockTransport, so the SDK's own
    (header-aware) retry policy runs for real against scripted responses."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LLMClient(model="x", api_key="k", base_url="http://test/v1",
                     http_client=http, **kw)


async def test_sdk_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:                       # two transient 429s …
            return _err(429, **{"retry-after-ms": "1"})
        return httpx.Response(                     # … then a real stream
            200, headers={"content-type": "text/event-stream"}, content=_OK_STREAM
        )

    client = _mock_client(handler, max_retries=5)
    chunks = [c async for c in client.stream(messages=[])]
    assert calls["n"] == 3                          # 2 retries + 1 success
    assert "".join(c.text_delta for c in chunks) == "hi"
    assert chunks[-1].finish_reason == "stop"


async def test_sdk_gives_up_after_max_retries():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _err(429, **{"retry-after-ms": "1"})

    client = _mock_client(handler, max_retries=2)
    with pytest.raises(RateLimitError):
        await client._open_stream([], None)
    assert calls["n"] == 3                           # 1 attempt + 2 retries


async def test_sdk_honors_x_should_retry_false():
    # 500 is normally retryable, but the server says don't — the SDK obeys the
    # header. This is the gap a pure exception-type retry would have missed.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _err(500, **{"x-should-retry": "false"})

    client = _mock_client(handler, max_retries=5)
    with pytest.raises(InternalServerError):
        await client._open_stream([], None)
    assert calls["n"] == 1                           # not retried


async def test_sdk_does_not_retry_bad_request():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _err(400)

    client = _mock_client(handler, max_retries=5)
    with pytest.raises(BadRequestError):
        await client._open_stream([], None)
    assert calls["n"] == 1                           # 4xx is non-transient


async def test_max_retries_zero_attempts_once():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _err(429, **{"retry-after-ms": "1"})

    client = _mock_client(handler, max_retries=0)
    with pytest.raises(RateLimitError):
        await client._open_stream([], None)
    assert calls["n"] == 1                           # retrying disabled


# --------------------------------------------------------------------------- #
# Failure classification — everything `stream()` raises is an LLMStreamError   #
# whose `retryable` flag tells the agent loop whether re-requesting helps.     #
# --------------------------------------------------------------------------- #


async def test_open_failure_wrapped_nonretryable():
    # The SDK already retried what was retryable before this surfaces, so the
    # loop must not retry again: retryable=False, original error chained.
    def handler(request):
        return _err(400)

    client = _mock_client(handler, max_retries=5)
    with pytest.raises(LLMStreamError) as ei:
        async for _ in client.stream(messages=[]):
            pass
    assert ei.value.retryable is False
    assert isinstance(ei.value.__cause__, BadRequestError)


async def test_exhausted_sdk_retries_wrapped_nonretryable():
    def handler(request):
        return _err(429, **{"retry-after-ms": "1"})

    client = _mock_client(handler, max_retries=1)
    with pytest.raises(LLMStreamError) as ei:
        async for _ in client.stream(messages=[]):
            pass
    assert ei.value.retryable is False
    assert isinstance(ei.value.__cause__, RateLimitError)


async def test_midstream_interruption_wrapped_retryable(client, monkeypatch):
    # The stream opens and yields, then the connection dies. The SDK never
    # retries this; it must surface as retryable=True so the loop can discard
    # the partial turn and re-request.
    closed = {"n": 0}

    class _BrokenStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield _Event([_Choice(_Delta(content="He"))])
            raise httpx.ReadError("connection lost")

        async def close(self):  # the abandoned stream must be released
            closed["n"] += 1

    async def fake_open(messages, tools):
        return _BrokenStream()

    monkeypatch.setattr(client, "_open_stream", fake_open)
    got = []
    with pytest.raises(LLMStreamError) as ei:
        async for chunk in client.stream(messages=[]):
            got.append(chunk)
    assert ei.value.retryable is True
    assert isinstance(ei.value.__cause__, httpx.ReadError)
    assert [c.text_delta for c in got] == ["He"]  # the partial did stream
    assert closed["n"] == 1


async def test_premature_eof_is_retryable_truncation(client, monkeypatch):
    # A stream that ends without ever sending a finish reason is a truncated
    # response — surfacing beats silently committing half a reply (or running
    # a tool on half its JSON arguments).
    events = [
        _Event([_Choice(_Delta(content="half a re"))]),
        _Event([_Choice(_Delta(tool_calls=[
            _ToolCallDelta(index=0, id="c0", function=_Fn(name="x", arguments='{"pa')),
        ]))]),
    ]
    chunks = []
    with pytest.raises(LLMStreamError) as ei:

        async def fake_open(messages, tools):
            return make_openai_stream(events)

        monkeypatch.setattr(client, "_open_stream", fake_open)
        async for chunk in client.stream(messages=[]):
            chunks.append(chunk)
    assert ei.value.retryable is True
    assert "finish reason" in str(ei.value)
    # No terminal chunk was yielded — the half-assembled tool call never escapes.
    assert all(c.tool_calls is None for c in chunks)


def test_retry_and_timeout_wired_to_sdk():
    client = LLMClient(model="x", api_key="k", base_url="http://test/v1",
                       max_retries=7, timeout=33.0)
    assert client._client.max_retries == 7
    assert client._client.timeout.read == 33.0       # read window honored
    assert client._client.timeout.connect == 10.0    # connect capped for fast-fail


async def test_retry_and_timeout_flow_from_profile_to_client(tmp_path, monkeypatch):
    # The config knobs must reach the client: YAML → LLMCfg → from_profile →
    # LLMClient(...). from_profile imports LLMClient by attribute at call time,
    # so patching the module attribute intercepts construction.
    import lingcore.llm as llm_mod
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile

    captured: dict = {}

    class _SpyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def stream(self, messages, tools=None):  # never called here
            yield None

    monkeypatch.setattr(llm_mod, "LLMClient", _SpyClient)
    root = tmp_path / "p"
    root.mkdir()
    (root / "config.yaml").write_text(
        "name: t\nllm:\n  model: gpt-4o\n  max_retries: 7\n  timeout: 42\ntools: []\n",
        encoding="utf-8",
    )
    Agent.from_profile(AgentProfile.load(root))  # llm=None → builds the (spy) client
    assert captured["max_retries"] == 7
    assert captured["timeout"] == 42.0


async def test_stream_retries_flow_from_profile_to_agent(tmp_path):
    # stream_retries is an llm.* knob but it parameterizes the *loop* (the
    # client only classifies failures; the loop owns the re-request).
    from lingcore.agent import Agent
    from lingcore.config import AgentProfile
    from tests.fakes import FakeLLMClient

    root = tmp_path / "p"
    root.mkdir()
    (root / "config.yaml").write_text(
        "name: t\nllm:\n  model: m\n  stream_retries: 5\ntools: []\n",
        encoding="utf-8",
    )
    agent = Agent.from_profile(AgentProfile.load(root), llm=FakeLLMClient([]))
    assert agent.stream_retries == 5


def test_negative_stream_retries_rejected(tmp_path):
    from lingcore.config import AgentProfile
    from lingcore.errors import ConfigError

    root = tmp_path / "p"
    root.mkdir()
    (root / "config.yaml").write_text(
        "name: t\nllm:\n  model: m\n  stream_retries: -1\ntools: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        AgentProfile.load(root)
