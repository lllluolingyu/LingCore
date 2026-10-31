"""Llm anthropic tests: thinking."""

from __future__ import annotations

import json

import httpx2
import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import LLMStreamError
from lingcore.events import Final, StreamRetry, TextDelta
from lingcore.message import Message
from lingcore.sessions import SessionStore
from lingcore.usage import TokenUsage
from tests.support.llm_anthropic import (
    _redacted,
    _start,
    _stop,
    _text,
    _thinking,
    _tool,
)
from tests.support.llm_anthropic import make_client as make_client


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


async def test_thinking_limit_keeps_provider_hint_in_agent_history(
    make_client, tmp_path
):
    client, requests, _ = make_client(
        [_start(), *_thinking(0, ["still thinking"], ["signed"]), *_stop("max_tokens")]
    )
    profile = AgentProfile(llm={"model": "alias"}, tools=[], workspace=str(tmp_path))
    async with Agent.from_profile(profile, llm=client) as agent:
        events = [event async for event in agent.run("think")]
        assert "reduce the thinking budget" in events[-1].message
        assert agent.memory.messages[-1].finish_detail == events[-1].message
        assert len(requests) == 1


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
