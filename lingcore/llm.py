"""Async LLM client — the single seam over the OpenAI SDK.

The agent loop talks only to ``LLMClient`` and never imports ``openai``
directly. That isolation is what preserves the option to swap in a different
backend (or LangGraph) later without touching tools, config, or frontends.

All streaming quirks live here: OpenAI streams tool-call arguments as
index-keyed fragments of a JSON string across many chunks. This client
accumulates them internally and yields a single terminal chunk carrying the
fully-assembled, parsed ``ToolCall`` list — so the loop never sees partial
JSON.

Transient-failure retry is delegated to the OpenAI SDK rather than hand-rolled:
the SDK is header-aware (it honors ``Retry-After`` / ``retry-after-ms`` timing
and the ``x-should-retry`` hint) and retries the right statuses (408/409/429 +
5xx, plus connection/timeout) with exponential backoff. We only set the attempt
budget (``max_retries``) and a per-attempt ``timeout``; the retry covers the
initial request that opens the stream and never the iteration, so already-
emitted tokens are never replayed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

from lingcore.message import Message, ToolCall

# Connect phase fails fast even when ``timeout`` (the read/inactivity window) is
# generous — a black-hole host shouldn't hold a slot for the full read window.
_MAX_CONNECT_SECONDS = 10.0


@dataclass(slots=True)
class LLMChunk:
    """One unit of streamed output.

    ``text_delta`` carries incremental assistant text for live rendering.
    ``tool_calls`` is populated only on the final chunk of a turn, once all
    tool-call fragments have been assembled and their arguments parsed.
    """

    text_delta: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


@dataclass
class _ToolCallAccumulator:
    """Reassembles a streamed tool call keyed by its delta index."""

    id: str = ""
    name: str = ""
    args_fragments: list[str] = field(default_factory=list)

    def build(self) -> ToolCall:
        raw = "".join(self.args_fragments).strip()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            # Malformed args: hand the model an empty dict so the tool's
            # pydantic validation reports the problem back to it cleanly.
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return ToolCall(id=self.id, name=self.name, arguments=parsed)


class LLMClient:
    """Thin async wrapper over ``openai.AsyncOpenAI`` (chat completions)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        sampling: dict[str, Any] | None = None,
        max_retries: int = 10,
        timeout: float = 120.0,
        http_client: Any = None,
    ) -> None:
        self.model = model
        self.sampling = sampling or {}
        self._max_retries = max(0, max_retries)
        # Hand retrying to the SDK (header-aware, correct status handling). The
        # timeout is httpx's read (inactivity) window, not a total wall-clock
        # cap: it stops a *stalled* attempt from hanging on the SDK's 600s
        # default, but a steadily-streaming response can run longer, and across
        # max_retries attempts + backoff the total wait can still be minutes.
        # Connect is capped shorter so a dead host fails fast, not per attempt.
        sdk_timeout = httpx.Timeout(
            timeout, connect=min(timeout, _MAX_CONNECT_SECONDS)
        )
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": self._max_retries,
            "timeout": sdk_timeout,
        }
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = AsyncOpenAI(**client_kwargs)

    async def _open_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.sampling,
        }
        # Only send `tools` when non-empty; some OpenAI-compatible servers
        # reject an empty tools array.
        if tools:
            kwargs["tools"] = tools
        # The SDK applies the client's retry policy to this request: a transient
        # 429/5xx/connection failure before the stream opens is retried (with
        # backoff that honors Retry-After), and once the stream is yielding
        # tokens it is not — so emitted text is never duplicated.
        return await self._client.chat.completions.create(**kwargs)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream a single assistant turn.

        Yields text deltas as they arrive, then exactly one terminal chunk
        carrying any assembled tool calls plus the finish reason.
        """
        wire = [m.to_openai() for m in messages]
        stream = await self._open_stream(wire, tools)

        accumulators: dict[int, _ToolCallAccumulator] = {}
        finish_reason: str | None = None

        async for event in stream:
            if not event.choices:
                continue  # e.g. a trailing usage-only chunk
            choice = event.choices[0]
            delta = choice.delta

            if getattr(delta, "content", None):
                yield LLMChunk(text_delta=delta.content)

            for tc in getattr(delta, "tool_calls", None) or []:
                acc = accumulators.setdefault(tc.index, _ToolCallAccumulator())
                if tc.id:
                    acc.id = tc.id
                if tc.function and tc.function.name:
                    acc.name = tc.function.name
                if tc.function and tc.function.arguments:
                    acc.args_fragments.append(tc.function.arguments)

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        tool_calls = (
            [accumulators[i].build() for i in sorted(accumulators)]
            if accumulators
            else None
        )
        yield LLMChunk(tool_calls=tool_calls, finish_reason=finish_reason)
