"""Llm anthropic tests: streaming."""

from __future__ import annotations

import json

import httpx2
import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import LLMStreamError
from lingcore.events import Error, StreamRetry, ToolCallStarted
from lingcore.message import Message, ToolCall, ToolResult
from lingcore.usage import TokenUsage
from tests.support.llm_anthropic import _start, _stop, _text, _thinking, _tool
from tests.support.llm_anthropic import make_client as make_client


async def test_sampling_and_tool_schemas_reach_the_api(make_client):
    client, requests, _ = make_client(
        sampling={"max_tokens": 1234, "temperature": 0.3, "top_p": 0.8, "top_k": 20}
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    [chunk async for chunk in client.stream([Message.user("read")], tools=tools)]
    payload = json.loads(requests[0].content)
    assert payload["max_tokens"] == 1234
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["tools"] == [
        {
            "name": "read_file",
            "description": "Read a file",
            "cache_control": {"type": "ephemeral"},
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]


async def test_fragmented_parallel_tool_calls_survive_real_sdk_stream(make_client):
    events = [
        _start(),
        *_text("Checking files"),
        *_tool(1, "call_1", "read_file", ['{"pa', 'th":"a.txt"}']),
        *_tool(2, "call_2", "list_dir", ['{"path":', '"."}']),
        *_stop("tool_use"),
    ]
    client, _, _ = make_client(events)
    chunks = [chunk async for chunk in client.stream([Message.user("go")])]
    assert "".join(chunk.text_delta for chunk in chunks) == "Checking files"
    assert all(chunk.tool_calls is None for chunk in chunks[:-1])
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].tool_calls == [
        ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"}),
        ToolCall(id="call_2", name="list_dir", arguments={"path": "."}),
    ]


@pytest.mark.parametrize("fragments", [[], ["{broken"], ["[]"], ["null"]])
async def test_empty_arguments_are_valid_but_invalid_arguments_are_marked(
    make_client, fragments
):
    client, _, _ = make_client(
        [_start(), *_tool(0, "c", "read_file", fragments), *_stop("tool_use")]
    )
    chunks = [chunk async for chunk in client.stream([Message.user("go")])]
    assert chunks[-1].tool_calls == [
        ToolCall(
            id="c",
            name="read_file",
            invalid_arguments="".join(fragments) if fragments else None,
        )
    ]


@pytest.mark.parametrize(
    "reason, hint",
    [
        ("max_tokens", "max_tokens"),
        ("model_context_window_exceeded", "context"),
    ],
)
@pytest.mark.parametrize("thinking", [False, True])
async def test_tool_limits_abort_batch_without_retry_or_dispatch(
    make_client, tmp_path, reason, hint, thinking
):
    seen = []
    prefix = _thinking(0, ["Inspect files"], ["signature"]) if thinking else []
    first_tool_index = int(thinking)
    client, requests, body = make_client(
        turns=[
            [
                _start(),
                *prefix,
                *_tool(
                    first_tool_index,
                    "c1",
                    "write_file",
                    ['{"path":"unexpected.txt","content":"executed"}'],
                ),
                *_tool(first_tool_index + 1, "c2", "list_dir", ['{"path":"subdir']),
                *_stop(reason),
            ],
            [_start(), *_text("done"), *_stop()],
        ],
        usage_sink=seen.append,
    )
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "alias", "backend": "anthropic", "stream_retries": 1},
            "tools": ["write_file", "list_dir"],
            "workspace": str(tmp_path),
            "sessions": {"enabled": False},
        }
    )
    agent = Agent.from_profile(profile, llm=client, base_dir=tmp_path)
    events = [event async for event in agent.run("inspect subdir")]

    # A limit applies to the whole response: even complete sibling calls must
    # not run before the truncated call can be recovered.
    assert not (tmp_path / "unexpected.txt").exists()
    assert not any(
        isinstance(event, (ToolCallStarted, StreamRetry)) for event in events
    )
    assert isinstance(events[-1], Error) and hint in events[-1].message
    assert "tool call response was truncated" in events[-1].message
    assert "2 tool call(s) were not run" in events[-1].message
    assert len(requests) == 1
    partial = agent.memory.messages[-1]
    assert partial.finish_reason == client._map_finish_reason(reason)
    assert not partial.tool_calls and partial.anthropic_content is None
    assert seen == [TokenUsage("served-model", 12, 9)]
    assert body.closed


async def test_preserves_all_system_prompts_and_tool_history(make_client):
    client, requests, _ = make_client()
    history = [
        Message.system("Policy A"),
        Message.system("Policy B"),
        Message.user("read"),
        Message.assistant(
            tool_calls=[ToolCall(id="c", name="read_file", arguments={})]
        ),
        Message.from_tool_result(
            ToolResult(call_id="c", name="read_file", content="body")
        ),
    ]
    [chunk async for chunk in client.stream(history)]
    payload = json.loads(requests[0].content)
    assert payload["system"] == [
        {
            "type": "text",
            "text": "Policy A\n\nPolicy B",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "read", "cache_control": {"type": "ephemeral"}}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "c", "name": "read_file", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c",
                    "content": "body",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


async def test_usage_merges_partial_deltas_and_includes_cache_tokens(make_client):
    seen = []
    client, _, _ = make_client(
        [
            _start(cache_read_input_tokens=80, cache_creation_input_tokens=20),
            *_text(),
            *_stop(),
        ],
        usage_sink=seen.append,
    )
    async for chunk in client.stream([Message.user("hi")]):
        if chunk.finish_reason:
            assert seen == [TokenUsage("served-model", 112, 9, 80)]
    assert seen == [TokenUsage("served-model", 112, 9, 80)]


async def test_cumulative_usage_updates_replace_counts(make_client):
    seen = []
    client, _, _ = make_client(
        [
            _start(cache_read_input_tokens=80, cache_creation_input_tokens=20),
            *_stop(
                input_tokens=15,
                cache_read_input_tokens=100,
                cache_creation_input_tokens=30,
            ),
        ],
        usage_sink=seen.append,
    )
    [chunk async for chunk in client.stream([Message.user("hi")])]
    assert seen == [TokenUsage("served-model", 145, 9, 100)]


@pytest.mark.parametrize(
    "reason, expected",
    [("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length")],
)
async def test_finish_reason_mapping(make_client, reason, expected):
    client, _, _ = make_client([_start(), *_text(), *_stop(reason)])
    chunks = [chunk async for chunk in client.stream([Message.user("hi")])]
    assert chunks[-1].finish_reason == expected


@pytest.mark.parametrize("status", [400, 429, 500])
async def test_open_failure_has_no_agent_retry_after_sdk_budget(make_client, status):
    client, requests, body = make_client(status=status)
    with pytest.raises(LLMStreamError, match="request failed") as error:
        [chunk async for chunk in client.stream([Message.user("hi")])]
    assert not error.value.retryable
    assert len(requests) == 1


@pytest.mark.parametrize("ending", [[], [httpx2.ReadError("connection dropped")]])
async def test_interrupted_stream_is_retryable_and_reports_usage(make_client, ending):
    seen = []
    client, _, body = make_client(
        [_start(), *_text("partial"), *ending], usage_sink=seen.append
    )
    chunks = []
    with pytest.raises(LLMStreamError) as error:
        async for chunk in client.stream([Message.user("hi")]):
            chunks.append(chunk)
    assert error.value.retryable
    assert not any(chunk.finish_reason for chunk in chunks)
    assert seen == [TokenUsage("served-model", 12, 1)]
    assert body.closed


async def test_closing_consumer_releases_http_stream(make_client):
    client, _, body = make_client()
    response = client.stream([Message.user("hi")])
    assert (await anext(response)).text_delta == "Hello"
    await response.aclose()
    assert body.closed
