"""Fixtures and builders for llm anthropic tests."""

from __future__ import annotations

import json

import httpx2
import pytest

from lingcore.llm_anthropic import AnthropicLLMClient


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
