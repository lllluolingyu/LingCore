"""Async LLM client — the single seam over the OpenAI SDK.

The agent loop talks only to ``LLMClient`` and never imports ``openai``
directly. That isolation is what preserves the option to swap in a different
backend (or LangGraph) later without touching tools, config, or frontends.

All streaming quirks live here: OpenAI streams tool-call arguments as
index-keyed fragments of a JSON string across many chunks. This client
accumulates them internally and yields a single terminal chunk carrying the
fully-assembled, parsed ``ToolCall`` list — so the loop never sees partial
JSON.

Transient-failure retry is two-tier. *Opening* the stream is delegated to the
OpenAI SDK rather than hand-rolled: the SDK is header-aware (it honors
``Retry-After`` / ``retry-after-ms`` timing and the ``x-should-retry`` hint)
and retries the right statuses (408/409/429 + 5xx, plus connection/timeout)
with exponential backoff, bounded by ``max_retries`` and a per-attempt
``timeout``. Once the stream is yielding, the SDK never retries — so this
client *classifies* instead of retrying: every failure is raised as a typed
``LLMStreamError``. ``retryable=True`` marks mid-stream interruption, stall,
or truncation (a stream that ends without a finish reason) — failures no
SDK-level retry will ever cover; ``retryable=False`` marks failures where the
open already spent the SDK's budget or the request itself is invalid. The
agent loop owns the recovery for retryable failures (discard the partial
turn, announce it, re-request) — so already-emitted tokens are never silently
replayed or duplicated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

from lingcore.errors import LLMStreamError
from lingcore.message import Message, ToolCall

# Connect phase fails fast even when ``timeout`` (the read/inactivity window) is
# generous — a black-hole host shouldn't hold a slot for the full read window.
_MAX_CONNECT_SECONDS = 10.0


def _describe(exc: BaseException) -> str:
    """Compact one-line cause description for error messages."""
    name = type(exc).__name__
    msg = " ".join(str(exc).split())
    if len(msg) > 200:
        msg = msg[:200] + "…"
    return f"{name}: {msg}" if msg else name


async def _close_quietly(stream: Any) -> None:
    """Best-effort close of an abandoned stream so its connection is freed
    before a retry opens a new one. Never raises."""
    closer = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception:
        pass


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

        Failure contract: every failure surfaces as ``LLMStreamError``.
        Opening the request has already been through the SDK's header-aware
        retry policy, so an open failure is raised ``retryable=False``; an
        interruption *after* the stream started — or a stream that ends
        without a finish reason (truncation) — is ``retryable=True``, because
        no SDK retry ever covers it and only the caller can decide to discard
        the partial turn and re-request.
        """
        wire = [m.to_openai() for m in messages]
        try:
            stream = await self._open_stream(wire, tools)
        except Exception as e:
            raise LLMStreamError(
                f"request failed: {_describe(e)}", retryable=False
            ) from e

        accumulators: dict[int, _ToolCallAccumulator] = {}
        finish_reason: str | None = None

        try:
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
        except Exception as e:
            await _close_quietly(stream)
            raise LLMStreamError(
                f"stream interrupted: {_describe(e)}", retryable=True
            ) from e

        if finish_reason is None:
            # The server closed the stream without ever sending a finish
            # reason: a truncated response. Surfacing it beats silently
            # committing a half reply — or running a tool on half its JSON
            # arguments.
            raise LLMStreamError(
                "stream ended without a finish reason (response truncated)",
                retryable=True,
            )

        tool_calls = (
            [accumulators[i].build() for i in sorted(accumulators)]
            if accumulators
            else None
        )
        yield LLMChunk(tool_calls=tool_calls, finish_reason=finish_reason)
