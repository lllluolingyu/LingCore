"""Responses integration through the real SDK with deterministic HTTP streams."""

from __future__ import annotations

import asyncio
import base64
import json
from copy import deepcopy

import httpx
import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import LLMStreamError
from lingcore.events import Error, Final, StreamRetry, ToolCallStarted
from lingcore.llm_responses import ResponsesLLMClient
from lingcore.memory import WindowMemory
from lingcore.message import Attachment, Message, ToolCall
from lingcore.sessions import SessionStore
from lingcore.tracing.content import sanitized_content


def reasoning(value="opaque-completed-state"):
    return {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": value,
    }


def call(arguments='{"path":"a.txt"}', call_id="call_1"):
    return {
        "type": "function_call",
        "id": "fc_" + call_id,
        "call_id": call_id,
        "name": "read_file",
        "arguments": arguments,
        "status": "completed",
    }


def message(text="done", *, refusal=False):
    part = (
        {"type": "refusal", "refusal": text}
        if refusal
        else {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
    )
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [part],
    }


def terminal(output=None, *, status="completed", reason=None, error=None):
    return {
        "type": "response." + status,
        "sequence_number": 20,
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 1234,
            "status": status,
            "model": "served-sol",
            "output": output or [],
            "incomplete_details": {"reason": reason} if reason else None,
            "error": {"code": error, "message": "provider details"} if error else None,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "input_tokens_details": {"cached_tokens": 12},
                "output_tokens_details": {"reasoning_tokens": 7},
                "total_tokens": 30,
            },
        },
    }


def delta(text, *, refusal=False):
    return {
        "type": "response.refusal.delta" if refusal else "response.output_text.delta",
        "delta": text,
        "sequence_number": 1,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "logprobs": [],
    }


class Body(httpx.AsyncByteStream):
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            if isinstance(event, asyncio.Event):
                await event.wait()
                continue
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()

    async def aclose(self):
        self.closed = True


@pytest.fixture
async def make_client():
    clients = []

    def make(turns=None, *, statuses=None, **kwargs):
        bodies = [
            Body(events)
            for events in (turns or [[delta("done"), terminal([message()])]])
        ]
        requests = []

        def handle(request):
            index = len(requests)
            requests.append(request)
            status = statuses[index] if statuses else 200
            if status != 200:
                return httpx.Response(
                    status,
                    headers={"retry-after-ms": "1"},
                    json={"error": {"message": "bad", "type": "error"}},
                )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=bodies[index]
            )

        client = ResponsesLLMClient(
            model="alias",
            api_key="test",
            base_url="http://test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
            **kwargs,
        )
        clients.append(client)
        return client, requests, bodies

    yield make
    for client in clients:
        await client._client.close()


async def test_native_request_settings_schema_and_usage(make_client):
    usages = []
    sampling = {
        "reasoning_effort": "high",
        "reasoning": {"summary": "auto"},
        "max_tokens": 900,
    }
    client, requests, bodies = make_client(
        sampling=sampling,
        usage_sink=usages.append,
        prompt_cache_key="session",
        stream_usage=False,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"optional": {"type": "string"}},
                },
            },
        }
    ]
    original = deepcopy(tools)
    chunks = [
        c
        async for c in client.stream([Message.system("sys"), Message.user("hi")], tools)
    ]
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/responses"
    assert body["input"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["max_output_tokens"] == 900 and body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["prompt_cache_key"] == "session"
    assert (
        not {
            "messages",
            "stream_options",
            "previous_response_id",
            "reasoning_effort",
            "max_tokens",
        }
        & body.keys()
    )
    assert body["tools"] == [
        {"type": "function", "strict": False, **tools[0]["function"]}
    ]
    assert tools == original and sampling["reasoning"] == {"summary": "auto"}
    assert "".join(c.text_delta for c in chunks) == "done"
    assert chunks[-1].responses_content == [message()]
    assert len(usages) == 1
    usage = usages[0]
    assert (
        usage.model,
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_input_tokens,
        usage.reasoning_tokens,
    ) == ("served-sol", 20, 10, 12, 7)
    assert bodies[0].closed


async def test_native_items_tool_followup_and_session_resume(make_client, tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    output = [reasoning(), call(), call(call_id="call_2")]
    partial = {**reasoning(), "encrypted_content": "unfinished-state"}
    client, requests, _ = make_client(
        turns=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "sequence_number": 0,
                    "item": partial,
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 1,
                    "sequence_number": 1,
                    "item_id": "fc_call_1",
                    "delta": '{"path":',
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "sequence_number": 2,
                    "item": reasoning(),
                },
                terminal(output),
            ],
            [terminal([reasoning("second-state"), message()])],
            [terminal([message("again")])],
        ]
    )
    profile = AgentProfile.model_validate(
        {
            "llm": {"backend": "openai-responses", "model": "alias"},
            "tools": ["read_file"],
            "workspace": str(tmp_path),
        }
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        agent = Agent.from_profile(
            profile, llm=client, base_dir=tmp_path, session_store=store
        )
        events = [e async for e in agent.run("read a.txt twice")]
        assert isinstance(events[-1], Final) and events[-1].content == "done"
        assert len([e for e in events if isinstance(e, ToolCallStarted)]) == 2
        followup = json.loads(requests[1].content)["input"]
        assert followup[-5:-2] == output
        assert [item["call_id"] for item in followup[-2:]] == ["call_1", "call_2"]
        assert all(
            item["type"] == "function_call_output" and item["output"] == "1\thello"
            for item in followup[-2:]
        )
        session_id = agent.tool_ctx.session_id
        await agent.aclose()
        resumed = Agent.from_profile(
            profile,
            llm=client,
            base_dir=tmp_path,
            session_store=store,
            session_id=session_id,
        )
        events = [e async for e in resumed.run("continue")]
        replay = json.loads(requests[2].content)["input"]
        assert replay[:-3] == followup
        assert replay[-3:] == [
            reasoning("second-state"),
            message(),
            {"role": "user", "content": "continue"},
        ]
        assert store.messages(session_id)[1].responses_content == output
        assert isinstance(events[-1], Final)
        await resumed.aclose()


@pytest.mark.parametrize("arguments", ['{"path":', "[]", "null", ""])
async def test_malformed_arguments_cannot_execute(make_client, tmp_path, arguments):
    client, requests, _ = make_client(
        turns=[[terminal([call(arguments)])], [terminal([message()])]]
    )
    profile = AgentProfile.model_validate(
        {"llm": {"model": "alias"}, "tools": ["read_file"], "workspace": str(tmp_path)}
    )
    agent = Agent.from_profile(profile, llm=client, base_dir=tmp_path)
    events = [e async for e in agent.run("read")]
    result = json.loads(requests[1].content)["input"][-1]
    assert "not run: tool arguments must be a complete JSON object" in result["output"]
    assert isinstance(events[-1], Final)
    await agent.aclose()


@pytest.mark.parametrize("cutoff", ["max_output_tokens", "content_filter"])
@pytest.mark.parametrize(
    "streamed,snapshot",
    [("partial", "partial"), ("partial", "part"), ("partial", None), ("", "partial")],
)
async def test_incomplete_drops_all_tools_and_native_state(
    make_client, tmp_path, cutoff, streamed, snapshot
):
    client, requests, bodies = make_client(
        turns=[
            [
                delta(streamed),
                terminal(
                    [reasoning(), call()]
                    + ([message(snapshot)] if snapshot is not None else []),
                    status="incomplete",
                    reason=cutoff,
                ),
            ]
        ]
    )
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "alias", "stream_retries": 2},
            "tools": ["read_file"],
            "workspace": str(tmp_path),
        }
    )
    agent = Agent.from_profile(profile, llm=client, base_dir=tmp_path)
    events = [e async for e in agent.run("go")]
    assert isinstance(events[-1], Error)
    assert not any(isinstance(e, ToolCallStarted) for e in events)
    assert not any(isinstance(e, StreamRetry) for e in events)
    assert len(requests) == 1 and bodies[0].closed
    assert all(m.responses_content is None for m in agent.memory.messages)
    if cutoff == "max_output_tokens":
        assert agent.memory.messages[-1].content == "partial"
        assert agent.memory.messages[-1].finish_reason == "length"
    await agent.aclose()


async def test_truncation_retry_does_not_replay_partial_reasoning(
    make_client, tmp_path, monkeypatch
):
    import lingcore.agent as agent_module

    async def no_wait(*args):
        pass

    monkeypatch.setattr(agent_module.asyncio, "sleep", no_wait)
    client, requests, bodies = make_client(
        turns=[[delta("abandoned"), httpx.ReadError("lost")], [terminal([message()])]]
    )
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "alias", "stream_retries": 1},
            "tools": [],
            "workspace": str(tmp_path),
        }
    )
    agent = Agent.from_profile(profile, llm=client, base_dir=tmp_path)
    events = [e async for e in agent.run("go")]
    assert any(isinstance(e, StreamRetry) for e in events)
    assert isinstance(events[-1], Final)
    assert json.loads(requests[0].content) == json.loads(requests[1].content)
    assert "abandoned" not in str([m.model_dump() for m in agent.memory.messages])
    assert all(body.closed for body in bodies)
    await agent.aclose()


@pytest.mark.parametrize(
    "events,retryable,category",
    [
        ([delta("half")], True, "interrupted"),
        ([terminal(status="failed", error="server_error")], True, "transient"),
        ([terminal(status="failed", error="invalid_prompt")], False, "request"),
        (
            [
                {
                    "type": "error",
                    "code": "server_error",
                    "message": "hidden",
                    "sequence_number": 1,
                }
            ],
            True,
            "transient",
        ),
    ],
)
async def test_stream_failure_contract(make_client, events, retryable, category):
    client, _, bodies = make_client(turns=[events])
    with pytest.raises(LLMStreamError) as caught:
        [c async for c in client.stream([])]
    assert caught.value.retryable is retryable and caught.value.category == category
    assert bodies[0].closed


@pytest.mark.parametrize(
    "status,category",
    [(400, "request"), (401, "auth"), (404, "configuration"), (429, "transient")],
)
async def test_open_error_contract(make_client, status, category):
    client, requests, _ = make_client(statuses=[status], max_retries=0)
    with pytest.raises(LLMStreamError) as caught:
        [c async for c in client.stream([])]
    assert caught.value.retryable is False and caught.value.category == category
    assert len(requests) == 1


async def test_sdk_open_retry_and_refusal(make_client):
    client, requests, _ = make_client(
        turns=[
            [],
            [
                delta("Cannot", refusal=True),
                terminal([message("Cannot", refusal=True)]),
            ],
        ],
        statuses=[429, 200],
        max_retries=1,
    )
    chunks = [c async for c in client.stream([])]
    assert len(requests) == 2 and chunks[-1].finish_reason == "stop"
    assert "".join(c.text_delta for c in chunks) == "Cannot"


async def test_cancellation_closes_stream(make_client):
    gate = asyncio.Event()
    client, _, bodies = make_client(turns=[[delta("partial"), gate]])
    stream = client.stream([])
    assert (await anext(stream)).text_delta == "partial"
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bodies[0].closed


async def test_stop_removes_committed_native_tool_turn_from_memory_and_session(
    make_client, tmp_path
):
    client, _, bodies = make_client(
        turns=[[terminal([reasoning(), call()])], [delta("waiting"), asyncio.Event()]]
    )
    profile = AgentProfile.model_validate(
        {"llm": {"model": "alias"}, "tools": ["read_file"], "workspace": str(tmp_path)}
    )
    reached = asyncio.Event()
    with SessionStore(tmp_path / "sessions.db") as store:
        agent = Agent.from_profile(
            profile, llm=client, base_dir=tmp_path, session_store=store
        )

        async def drive():
            from lingcore.events import TextDelta

            async for event in agent.run("go"):
                if isinstance(event, TextDelta) and event.text == "waiting":
                    reached.set()

        task = asyncio.create_task(drive())
        await asyncio.wait_for(reached.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        agent.finalize_cancelled_turn()
        persisted = store.messages(agent.tool_ctx.session_id)
        assert [(m.role, m.content) for m in persisted] == [("user", "go")]
        assert all(m.responses_content is None for m in agent.memory.messages)
        assert all(body.closed for body in bodies)
        await agent.aclose()


@pytest.mark.parametrize(
    "sampling",
    [
        {"reasoning_effort": "high", "reasoning": {"effort": "max"}},
        {"max_tokens": 1, "max_output_tokens": 2},
        {"previous_response_id": "r"},
        {"extra_body": {"store": True}},
        {"extra_body": 42},
        {"extra_body": []},
        {"tool_choice": {"type": "function", "function": {}}},
        {"tool_choice": {"type": "function", "function": "read_file"}},
        {"tool_choice": {"type": "function"}},
        {"tool_choice": []},
    ],
)
async def test_conflicting_or_stateful_settings_fail_before_http(make_client, sampling):
    client, requests, _ = make_client(sampling=sampling)
    with pytest.raises(LLMStreamError) as caught:
        [c async for c in client.stream([])]
    assert not requests and caught.value.category == "configuration"


@pytest.mark.parametrize(
    "sampling",
    [
        {"max_tokens": 1, "max_output_tokens": 2},
        {"previous_response_id": "r"},
        {"extra_body": 42},
        {"tool_choice": {"type": "function", "function": {}}},
    ],
)
def test_invalid_responses_settings_fail_during_profile_validation(sampling):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentProfile.model_validate(
            {
                "llm": {
                    "model": "alias",
                    "backend": "openai-responses",
                    "sampling": sampling,
                }
            }
        )


def test_media_legacy_history_and_opaque_state():
    image = Attachment(
        kind="image",
        media_type="image/png",
        data=base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode(),
        fallback_text="image description",
    )
    pdf = Attachment(
        kind="file",
        media_type="application/pdf",
        data=base64.b64encode(b"%PDF-1.4\n").decode(),
        name="a.pdf",
    )
    msg = Message.user("inspect", [image, pdf])
    parts = msg.to_responses()[0]["content"]
    assert [p["type"] for p in parts] == ["input_text", "input_image", "input_file"]
    assert parts[-1]["filename"] == "a.pdf" and parts[-1]["file_data"].startswith(
        "data:application/pdf;base64,"
    )
    fallback = msg.to_responses(attachment_modalities=frozenset())[0]["content"]
    assert "image description" in fallback and image.data not in fallback
    legacy = Message.assistant(
        "checking", tool_calls=[ToolCall(id="c", name="f", arguments={"x": 1})]
    )
    assert legacy.to_responses() == [
        {"role": "assistant", "content": "checking"},
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": '{"x": 1}'},
    ]
    native = Message.assistant(
        "done", responses_content=[reasoning("opaque" * 100), message()]
    )
    native.to_responses()[0]["encrypted_content"] = "mutated"
    assert native.responses_content[0]["encrypted_content"] == "opaque" * 100
    memory = WindowMemory(model="offline")
    assert memory._tokens(native) > memory._tokens(Message.assistant("done")) + 100
    data, _ = sanitized_content(native.model_dump(), (), 10000)
    assert b"opaque" not in data and b"responses_content" not in data


async def test_profile_selects_responses_for_main_and_vision(tmp_path):
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "alias", "backend": "openai-responses", "modalities": []},
            "tools": [],
            "media_fallback": {
                "image": {"model": "vision", "backend": "openai-responses"}
            },
            "workspace": str(tmp_path),
        }
    )
    agent = Agent.from_profile(profile, base_dir=tmp_path)
    # Without tracing the clients are used directly. Construction stays offline.
    assert profile.llm.base_url == "https://api.openai.com/v1"
    assert isinstance(agent.llm, ResponsesLLMClient)
    assert isinstance(agent.media_adapter.vision, ResponsesLLMClient)
    await agent.aclose()
