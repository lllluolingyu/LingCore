"""Llm anthropic tests: core behavior."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.events import Final
from lingcore.message import Message, ToolCall, ToolResult
from lingcore.sessions import SessionStore
from tests.support.llm_anthropic import _start, _stop, _text
from tests.support.llm_anthropic import make_client as make_client


async def test_default_request_streams_with_required_limit_and_sdk_timeout(make_client):
    client, requests, body = make_client(sampling={})
    chunks = [chunk async for chunk in client.stream([Message.user("hi")])]

    assert "".join(chunk.text_delta for chunk in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].tool_calls is None
    assert sum(chunk.finish_reason is not None for chunk in chunks) == 1
    request = requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert json.loads(request.content)["max_tokens"] > 0
    assert request.extensions["timeout"]["connect"] == 10.0
    assert request.extensions["timeout"]["read"] == 120.0
    assert body.closed


@pytest.mark.parametrize("prompt_caching", [False, True])
@pytest.mark.parametrize("resume", [False, True])
async def test_empty_reply_is_not_replayed_live_or_after_resume(
    make_client, tmp_path, prompt_caching, resume
):
    client, requests, _ = make_client(
        turns=[
            [_start(), *_stop(output_tokens=2)],
            [_start(), *_text("continued"), *_stop()],
        ],
        prompt_caching=prompt_caching,
    )
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "alias", "backend": "anthropic"},
            "tools": [],
            "workspace": str(tmp_path),
        }
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        agent = Agent.from_profile(
            profile, llm=client, base_dir=tmp_path, session_store=store
        )
        events = [event async for event in agent.run("hello")]
        assert isinstance(events[-1], Final) and events[-1].content == ""
        session_id = agent.tool_ctx.session_id
        original_history = store.messages(session_id)
        assert [(m.role, m.content) for m in original_history] == [
            ("user", "hello"),
            ("assistant", ""),
        ]

        if resume:
            await agent.aclose()
            agent = Agent.from_profile(
                profile,
                llm=client,
                base_dir=tmp_path,
                session_store=store,
                session_id=session_id,
            )
        events = [event async for event in agent.run("Please continue")]
        assert isinstance(events[-1], Final) and events[-1].content == "continued"
        wire = json.loads(requests[1].content)["messages"]
        assert [message["role"] for message in wire] == ["user", "user"]
        assert all(message["content"] for message in wire)
        # Rendering must not rewrite the canonical transcript.
        assert store.messages(session_id)[:2] == original_history


@pytest.mark.parametrize("snapshot", [None, []])
async def test_empty_assistant_at_history_start_is_not_prefilled(make_client, snapshot):
    client, requests, _ = make_client(prompt_caching=False)
    messages = [
        Message.assistant(anthropic_content=snapshot),
        Message.user("hello"),
    ]
    before = [message.model_dump() for message in messages]
    [chunk async for chunk in client.stream(messages)]
    assert json.loads(requests[0].content)["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert [message.model_dump() for message in messages] == before


async def test_default_prompt_caching_marks_system_and_user_prefix(make_client):
    client, requests, _ = make_client()
    messages = [Message.system("Reusable instructions"), Message.user("hello")]
    before = [message.model_dump() for message in messages]
    [chunk async for chunk in client.stream(messages)]
    payload = json.loads(requests[0].content)

    assert payload["system"] == [
        {
            "type": "text",
            "text": "Reusable instructions",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hello",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    # Use block breakpoints for compatibility with native gateways, including
    # ones that do not implement the newer top-level automatic caching field.
    assert "cache_control" not in payload
    assert [message.model_dump() for message in messages] == before


@pytest.mark.parametrize("parallel_calls", [1, 25])
async def test_prompt_cache_keeps_previous_boundary_across_tool_batches(
    make_client, parallel_calls
):
    client, requests, _ = make_client(
        turns=[[_start(), *_text(), *_stop()] for _ in range(3)]
    )
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        {"type": "function", "function": {"name": "list_dir", "parameters": {}}},
    ]
    original_tools = deepcopy(tools)
    history = [Message.system("Stable policy"), Message.user("inspect files")]
    [chunk async for chunk in client.stream(history, tools=tools)]

    calls = [
        ToolCall(id=f"c{i}", name="read_file", arguments={})
        for i in range(parallel_calls)
    ]
    signed = [
        {"type": "thinking", "thinking": "Inspect", "signature": "opaque"},
        {"type": "redacted_thinking", "data": "opaque-redacted"},
        *[
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in calls
        ],
    ]
    history.append(Message.assistant(tool_calls=calls, anthropic_content=signed))
    history.extend(
        Message.from_tool_result(ToolResult(call_id=c.id, name=c.name, content="file"))
        for c in calls
    )
    [chunk async for chunk in client.stream(history, tools=tools)]
    history.extend([Message.assistant("done"), Message.user("continue")])
    [chunk async for chunk in client.stream(history, tools=tools)]

    first, second, third = [json.loads(request.content) for request in requests]
    # The previous write is still an explicit lookup point even when a batch
    # adds more than the provider's 20-block lookback window.
    assert second["messages"][:1] == first["messages"]
    assert second["messages"][0]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    last_result = second["messages"][-1]
    assert last_result["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert third["messages"][-3] == last_result
    assert third["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert "cache_control" not in third["messages"][0]["content"][0]
    assert second["messages"][1]["content"] == signed
    assert third["messages"][1]["content"] == signed
    for payload in (first, second, third):
        assert payload["system"] == first["system"]
        assert payload["tools"] == first["tools"]
        assert "cache_control" not in payload["tools"][0]
        assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        blocks = [
            *payload["tools"],
            *payload["system"],
            *(block for message in payload["messages"] for block in message["content"]),
        ]
        assert sum("cache_control" in block for block in blocks) <= 4
    assert tools == original_tools
    assert history[2].anthropic_content == signed


async def test_prompt_caching_can_be_disabled(make_client):
    client, requests, _ = make_client(prompt_caching=False)
    [
        chunk
        async for chunk in client.stream([Message.system("policy"), Message.user("hi")])
    ]
    payload = json.loads(requests[0].content)
    assert payload["system"] == "policy"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "cache_control" not in payload


@pytest.mark.parametrize("extra_body", [False, True])
async def test_native_cache_control_overrides_default_breakpoints(
    make_client, extra_body
):
    control = {"type": "ephemeral", "ttl": "1h"}
    sampling = {"cache_control": control}
    if extra_body:
        sampling = {"extra_body": sampling}
    original_sampling = deepcopy(sampling)
    client, requests, _ = make_client(sampling=sampling)
    [
        chunk
        async for chunk in client.stream([Message.system("policy"), Message.user("hi")])
    ]
    payload = json.loads(requests[0].content)
    assert payload["cache_control"] == control
    assert payload["system"] == "policy"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert sampling == original_sampling


async def test_explicit_block_breakpoints_are_not_augmented(make_client):
    system = [
        {"type": "text", "text": f"part {i}", "cache_control": {"type": "ephemeral"}}
        for i in range(4)
    ]
    client, requests, _ = make_client(sampling={"system": system})
    [chunk async for chunk in client.stream([Message.user("hi")])]
    payload = json.loads(requests[0].content)
    assert payload["system"] == system
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "cache_control" not in payload


async def test_empty_text_and_thinking_are_not_cache_breakpoints(make_client):
    signed = [{"type": "thinking", "thinking": "", "signature": "opaque"}]
    client, requests, _ = make_client()
    [
        chunk
        async for chunk in client.stream(
            [
                Message.system(""),
                Message.user(""),
                Message.assistant(anthropic_content=signed),
            ]
        )
    ]
    payload = json.loads(requests[0].content)
    assert payload["messages"][-1]["content"] == signed
    assert "cache_control" not in json.dumps(payload)
