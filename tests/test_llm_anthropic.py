"""Exercise the Anthropic adapter through the real SDK and an HTTP transport."""

from __future__ import annotations

import json
from copy import deepcopy

import httpx2
import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import LLMStreamError
from lingcore.events import Error, Final, StreamRetry, TextDelta, ToolCallStarted
from lingcore.llm_anthropic import AnthropicLLMClient
from lingcore.message import Message, ToolCall, ToolResult
from lingcore.sessions import SessionStore
from lingcore.usage import TokenUsage


def _start(**usage):
    return {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "served-model",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1, **usage},
        },
    }


def _text(text="Hello", index=0):
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": index},
    ]


def _tool(index, call_id, name, fragments):
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": {},
            },
        },
        *[
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": fragment},
            }
            for fragment in fragments
        ],
        {"type": "content_block_stop", "index": index},
    ]


def _stop(reason="end_turn", **usage):
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
            "usage": {"output_tokens": 9, **usage},
        },
        {"type": "message_stop"},
    ]


def _thinking(index, fragments, signatures):
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        *[
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": fragment},
            }
            for fragment in fragments
        ],
        *[
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": signature},
            }
            for signature in signatures
        ],
        {"type": "content_block_stop", "index": index},
    ]


def _redacted(index, data):
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "redacted_thinking", "data": data},
        },
        {"type": "content_block_stop", "index": index},
    ]


class _SSEBody(httpx2.AsyncByteStream):
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield (f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())

    async def aclose(self):
        self.closed = True


@pytest.fixture
async def make_client():
    clients = []

    def make(events=None, *, turns=None, status=200, sampling=None, **kwargs):
        if turns is None:
            turns = [[_start(), *_text(), *_stop()] if events is None else events]
        bodies = [_SSEBody(events) for events in turns]
        requests = []

        def handle(request):
            requests.append(request)
            if status != 200:
                return httpx2.Response(
                    status,
                    json={
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "bad"},
                    },
                )
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=bodies[len(requests) - 1],
            )

        http = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
        clients.append(http)
        client = AnthropicLLMClient(
            model="alias",
            api_key="test-key",
            sampling={"max_tokens": 256} if sampling is None else sampling,
            http_client=http,
            max_retries=0,
            **kwargs,
        )
        return client, requests, bodies[0]

    yield make
    for client in clients:
        await client.aclose()


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
async def test_empty_or_invalid_tool_arguments_allow_tool_validation(
    make_client, fragments
):
    client, _, _ = make_client(
        [_start(), *_tool(0, "c", "read_file", fragments), *_stop("tool_use")]
    )
    chunks = [chunk async for chunk in client.stream([Message.user("go")])]
    assert chunks[-1].tool_calls == [ToolCall(id="c", name="read_file", arguments={})]


@pytest.mark.parametrize(
    "reason, hint",
    [("max_tokens", "max_tokens"), ("model_context_window_exceeded", "context")],
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
    assert len(requests) == 1
    assert not any(message.role == "assistant" for message in agent.memory.render(""))
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


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "enabled", "budget_tokens": 1024},
        {"type": "adaptive", "display": "summarized"},
    ],
)
async def test_thinking_retains_signatures_and_separates_reasoning(
    make_client, preserve, thinking
):
    client, requests, _ = make_client(
        [
            _start(),
            *_thinking(0, ["Inspect ", "the file."], ["sig-", "123"]),
            *_text("Answer", index=1),
            *_stop(),
        ],
        sampling={"max_tokens": 4096, "thinking": thinking},
        preserve_reasoning=preserve,
    )
    chunks = [chunk async for chunk in client.stream([Message.user("hi")])]
    assert json.loads(requests[0].content)["thinking"] == thinking
    assert "".join(chunk.text_delta for chunk in chunks) == "Answer"
    assert "".join(chunk.reasoning_delta for chunk in chunks) == (
        "Inspect the file." if preserve else ""
    )
    assert all(chunk.anthropic_content is None for chunk in chunks[:-1])
    assert chunks[-1].anthropic_content == [
        {"type": "thinking", "thinking": "Inspect the file.", "signature": "sig-123"},
        {"type": "text", "text": "Answer"},
    ]


async def test_default_thinking_preserves_redacted_and_signature_only_blocks(
    make_client,
):
    client, _, _ = make_client(
        [
            _start(),
            *_redacted(0, "opaque-data"),
            *_thinking(1, [""], ["opaque-signature"]),
            *_text("done", index=2),
            *_stop(),
        ]
    )
    chunks = [chunk async for chunk in client.stream([Message.user("hi")])]
    assert all(chunk.reasoning_delta == "" for chunk in chunks)
    assert chunks[-1].anthropic_content == [
        {"type": "redacted_thinking", "data": "opaque-data"},
        {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
        {"type": "text", "text": "done"},
    ]


async def test_thinking_tool_loop_replays_exact_content_after_session_resume(
    make_client, tmp_path
):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    first_content = [
        {"type": "thinking", "thinking": "Inspect the file.", "signature": "signed-1"},
        {"type": "text", "text": "Checking"},
        {"type": "redacted_thinking", "data": "redacted-1"},
        {"type": "thinking", "thinking": "", "signature": "signed-2"},
        {
            "type": "tool_use",
            "id": "c1",
            "name": "read_file",
            "input": {"path": "a.txt"},
        },
    ]
    last_content = [
        {"type": "thinking", "thinking": "", "signature": "signed-3"},
        {"type": "text", "text": "done"},
    ]
    client, requests, _ = make_client(
        turns=[
            [
                _start(),
                *_thinking(0, ["Inspect ", "the file."], ["signed-1"]),
                *_text("Checking", index=1),
                *_redacted(2, "redacted-1"),
                *_thinking(3, [""], ["signed-2"]),
                *_tool(4, "c1", "read_file", ['{"path":"a.txt"}']),
                *_stop("tool_use"),
            ],
            [
                _start(),
                *_thinking(0, [""], ["signed-3"]),
                *_text("done", index=1),
                *_stop(),
            ],
            [_start(), *_text("again"), *_stop()],
        ],
        sampling={"max_tokens": 4096, "thinking": {"type": "adaptive"}},
        preserve_reasoning=True,
    )
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "alias", "backend": "anthropic"},
            "tools": ["read_file"],
            "workspace": str(tmp_path),
        }
    )
    db_path = tmp_path / "sessions.db"
    with SessionStore(db_path) as store:
        agent = Agent.from_profile(
            profile, llm=client, base_dir=tmp_path, session_store=store
        )
        events = [event async for event in agent.run("read a.txt")]
        assert isinstance(events[-1], Final) and events[-1].content == "done"
        assert (
            "".join(event.text for event in events if isinstance(event, TextDelta))
            == "Checkingdone"
        )
        followup = json.loads(requests[1].content)["messages"]
        assert (
            next(message for message in followup if message["role"] == "assistant")[
                "content"
            ]
            == first_content
        )
        assert followup[-1]["content"][0]["tool_use_id"] == "c1"
        assert followup[-1]["content"][0]["content"] == "1\thello"
        session_id = agent.tool_ctx.session_id

    with SessionStore(db_path) as store:
        resumed = Agent.from_profile(
            profile,
            llm=client,
            base_dir=tmp_path,
            session_store=store,
            session_id=session_id,
        )
        events = [event async for event in resumed.run("again")]
        assert isinstance(events[-1], Final)
        replayed = [
            message["content"]
            for message in json.loads(requests[2].content)["messages"]
            if message["role"] == "assistant"
        ]
        assert replayed == [first_content, last_content]


@pytest.mark.parametrize("signatures", [[], ["signature"]])
@pytest.mark.parametrize(
    "reason", ["end_turn", "max_tokens", "model_context_window_exceeded"]
)
async def test_incomplete_thinking_cannot_commit_replay_state(
    make_client, signatures, reason
):
    seen = []
    events = _thinking(0, ["unfinished"], signatures)
    if signatures:
        events.pop()  # Missing content_block_stop, even though a signature arrived.
    client, _, _ = make_client(
        [_start(), *events, *_stop(reason)], usage_sink=seen.append
    )
    with pytest.raises(LLMStreamError, match="thinking") as error:
        [chunk async for chunk in client.stream([Message.user("hi")])]
    assert error.value.retryable is (reason == "end_turn")
    if reason == "max_tokens":
        assert "max_tokens" in str(error.value)
    elif reason == "model_context_window_exceeded":
        assert "context" in str(error.value)
    assert seen == [TokenUsage("served-model", 12, 9)]


@pytest.mark.parametrize(
    "reason, hint",
    [("max_tokens", "max_tokens"), ("model_context_window_exceeded", "context")],
)
async def test_thinking_exhausting_output_limit_is_actionable_without_retry(
    make_client, reason, hint
):
    client, _, _ = make_client(
        [_start(), *_thinking(0, ["still thinking"], ["signed"]), *_stop(reason)]
    )
    with pytest.raises(LLMStreamError, match=hint) as error:
        [chunk async for chunk in client.stream([Message.user("hi")])]
    assert not error.value.retryable


async def test_thinking_binding_controls_reach_api_from_profile(make_client):
    profile = AgentProfile.model_validate(
        {
            "llm": {
                "model": "claude-test",
                "backend": "anthropic",
                "sampling": {
                    "max_tokens": 16000,
                    "thinking": {
                        "type": "adaptive",
                        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
                    },
                    "extra_headers": {
                        "anthropic-beta": "thinking-binding-controls-2026-08-01"
                    },
                },
            }
        }
    )
    client, requests, _ = make_client(sampling=profile.llm.sampling.as_kwargs())
    [chunk async for chunk in client.stream([Message.user("hi")])]
    assert (
        requests[0].headers["anthropic-beta"] == "thinking-binding-controls-2026-08-01"
    )
    assert json.loads(requests[0].content)["thinking"] == {
        "type": "adaptive",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }


async def test_thinking_usage_comes_from_provider_counts(make_client):
    seen = []
    client, _, _ = make_client(
        [
            _start(),
            *_thinking(0, ["summary"], ["sig"]),
            *_text(index=1),
            *_stop(output_tokens=2000, output_tokens_details={"thinking_tokens": 1900}),
        ],
        usage_sink=seen.append,
    )
    [chunk async for chunk in client.stream([Message.user("hi")])]
    assert seen == [TokenUsage("served-model", 12, 2000, 0, 1900)]


async def test_retry_discards_partial_thinking_state(
    make_client, tmp_path, monkeypatch
):
    monkeypatch.setattr("lingcore.agent._backoff_seconds", lambda _: 0)
    client, requests, _ = make_client(
        turns=[
            [
                _start(),
                *_thinking(0, ["discard me"], ["stale-signature"]),
                httpx2.ReadError("dropped"),
            ],
            [
                _start(),
                *_thinking(0, ["kept"], ["valid-signature"]),
                *_text("done", index=1),
                *_stop(),
            ],
        ],
        preserve_reasoning=True,
    )
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "alias", "backend": "anthropic", "stream_retries": 1},
            "tools": [],
            "workspace": str(tmp_path),
        }
    )
    agent = Agent.from_profile(profile, llm=client, base_dir=tmp_path)
    events = [event async for event in agent.run("hello")]
    assert isinstance(events[-1], Final) and events[-1].content == "done"
    assert sum(isinstance(event, StreamRetry) for event in events) == 1
    assert (
        json.loads(requests[0].content)["messages"]
        == json.loads(requests[1].content)["messages"]
    )
    assistant = next(
        message for message in agent.memory.render("") if message.role == "assistant"
    )
    assert assistant.reasoning_content == "kept"
    assert assistant.anthropic_content == [
        {"type": "thinking", "thinking": "kept", "signature": "valid-signature"},
        {"type": "text", "text": "done"},
    ]
