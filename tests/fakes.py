"""Shared test doubles.

``FakeLLMClient`` is scripted: each call to ``stream`` pops the next scripted
turn and yields it as ``LLMChunk``s. This lets the agent loop be tested with
zero network and zero API cost. A scripted turn may also be a
``StreamFailure`` — text that streams and then dies — to exercise the loop's
mid-stream recovery.

``make_openai_stream`` fabricates an async iterator shaped like the real
OpenAI streaming response, used to test ``LLMClient``'s delta reassembly
without monkeypatching the SDK's transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from lingcore.errors import LLMStreamError
from lingcore.llm import LLMChunk
from lingcore.message import Message, ToolCall


@dataclass
class ScriptedTurn:
    """One assistant turn the fake LLM will produce."""

    text: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = "stop"


@dataclass
class StreamFailure:
    """A scripted mid-stream failure: yield ``text``, then raise.

    Models a connection drop / truncation after ``text`` already streamed —
    exactly what the agent loop's stream-retry recovery must handle.
    """

    text: str = ""
    reason: str = "stream interrupted: connection dropped"
    retryable: bool = True


class FakeLLMClient:
    """A scripted stand-in for ``LLMClient`` with the same ``stream`` shape."""

    def __init__(self, turns: list[ScriptedTurn | StreamFailure]):
        self._turns = list(turns)
        self.calls: list[list[Message]] = []  # records messages seen each turn
        self.tool_schemas: list[list[dict[str, Any]] | None] = []  # tools per turn

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]:
        self.calls.append(list(messages))
        self.tool_schemas.append(tools)
        if not self._turns:
            # Default terminal behaviour if over-pumped: empty final reply.
            yield LLMChunk(tool_calls=None, finish_reason="stop")
            return
        turn = self._turns.pop(0)
        if turn.text:
            for piece in _chunk_text(turn.text):
                yield LLMChunk(text_delta=piece)
        if isinstance(turn, StreamFailure):
            raise LLMStreamError(turn.reason, retryable=turn.retryable)
        yield LLMChunk(tool_calls=turn.tool_calls, finish_reason=turn.finish_reason)


def _chunk_text(text: str, size: int = 4) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


# --- OpenAI streaming-shape fakes (for testing LLMClient itself) ----------


@dataclass
class _Fn:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _ToolCallDelta:
    index: int
    id: str | None = None
    function: _Fn | None = None


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list[_ToolCallDelta] | None = None


@dataclass
class _Choice:
    delta: _Delta
    finish_reason: str | None = None


@dataclass
class _Event:
    choices: list[_Choice] = field(default_factory=list)


async def make_openai_stream(events: list[_Event]):
    for ev in events:
        yield ev
