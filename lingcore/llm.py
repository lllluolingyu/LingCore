"""Async LLM client — the single seam over the OpenAI SDK.

The agent loop talks only to ``LLMClient`` and never imports ``openai``
directly. That isolation is what preserves the option to swap in a different
backend (or LangGraph) later without touching tools, config, or frontends.

All streaming quirks live here: OpenAI streams tool-call arguments as
index-keyed fragments of a JSON string across many chunks. This client
accumulates them internally and yields a single terminal chunk carrying the
fully-assembled, parsed ``ToolCall`` list — so the loop never sees partial
JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lingcore.message import Message, ToolCall

# Errors worth retrying: transient network / rate-limit / 5xx. Imported lazily
# so a missing optional symbol in some openai version never breaks import.
try:  # pragma: no cover - import shim
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    _RETRYABLE: tuple[type[Exception], ...] = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
except Exception:  # pragma: no cover
    _RETRYABLE = (Exception,)


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
    ) -> None:
        self.model = model
        self.sampling = sampling or {}
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
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
