"""Native Responses transport with stateless, durable reasoning/tool replay.

Conversation state belongs to Message/SessionStore, not an API-side response
chain. Only completed output snapshots escape the stream boundary. Therefore
Stop, retries, Edit, compaction and resume all use the same retained history.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

from lingcore._responses_sampling import responses_sampling
from lingcore.errors import LLMStreamError
from lingcore.llm import (
    _PENDING_CLOSES,
    LLMChunk,
    LLMClient,
    _close_quietly,
    _describe,
    _open_error_category,
    _ToolCallAccumulator,
)
from lingcore.message import Message, ToolCall
from lingcore.usage import usage_from_responses


def _failure(code: str | None) -> LLMStreamError:
    # Do not echo arbitrary provider messages containing submitted prompts.
    if code in {"context_length_exceeded", "model_context_window_exceeded"}:
        return LLMStreamError(
            "the context window was exhausted",
            retryable=False,
            finish_reason="model_context_window_exceeded",
        )
    transient = code in {"server_error", "rate_limit_exceeded"}
    return LLMStreamError(
        f"Responses request failed ({code or 'unknown error'})",
        retryable=transient,
        category="transient" if transient else "request",
    )


class ResponsesLLMClient(LLMClient):
    """Use /responses; share SDK connection, timeout and opening-retry policy."""

    async def _open_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Any:
        kwargs: dict[str, Any] = {
            **responses_sampling(self.sampling),
            "model": self.model,
            "input": messages,
            "stream": True,
            "store": False,
        }
        if tools:
            converted = []
            for tool in tools:
                if tool.get("type") != "function" or "function" not in tool:
                    raise ValueError(
                        "Responses backend supports registered function tools only"
                    )
                # Responses otherwise normalizes optional properties into
                # required ones. LingCore validates its existing schemas itself.
                converted.append(
                    {"type": "function", "strict": False, **deepcopy(tool["function"])}
                )
            kwargs["tools"] = converted
        if self._prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = self._prompt_cache_key
        return await self._client.responses.create(**kwargs)

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]:
        try:
            wire = [
                item
                for message in messages
                for item in message.to_responses(attachment_modalities=self._modalities)
            ]
            stream = await self._open_stream(wire, tools)
        except Exception as exc:
            raise LLMStreamError(
                f"request failed: {_describe(exc)}",
                retryable=False,
                category="configuration"
                if isinstance(exc, ValueError)
                else _open_error_category(exc),
            ) from exc

        response: Any = None
        terminal: str | None = None
        text_parts: list[str] = []
        try:
            async for event in stream:
                kind = event.type
                if terminal is not None:
                    raise LLMStreamError(
                        "events after terminal response", retryable=True
                    )
                if kind in {"response.output_text.delta", "response.refusal.delta"}:
                    text_parts.append(event.delta)
                    yield LLMChunk(text_delta=event.delta)
                elif kind in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    response = event.response
                    terminal = kind
                elif kind == "error":
                    raise _failure(getattr(event, "code", None))
                # Tool argument fragments and output_item.added can be partial.
                # The terminal response contains the complete ordered output,
                # including final encrypted reasoning and complete call JSON.
        except LLMStreamError:
            raise
        except Exception as exc:
            raise LLMStreamError(
                f"stream interrupted: {_describe(exc)}", retryable=True
            ) from exc
        finally:
            propagating = sys.exc_info()[1] is not None
            sink_error: Exception | None = None
            try:
                if response is not None:
                    self._report_usage(
                        getattr(response, "model", None),
                        getattr(response, "usage", None),
                    )
            except Exception as exc:
                sink_error = exc
            # A second cancellation must not strand the underlying connection.
            close_task = asyncio.ensure_future(_close_quietly(stream))
            _PENDING_CLOSES.add(close_task)
            close_task.add_done_callback(_PENDING_CLOSES.discard)
            await asyncio.shield(close_task)
            if sink_error is not None and not propagating:
                raise sink_error

        if terminal is None:
            raise LLMStreamError(
                "stream ended without a terminal response (response truncated)",
                retryable=True,
            )
        if getattr(response, "status", None) != terminal.removeprefix("response."):
            raise LLMStreamError(
                "terminal response status disagrees with event", retryable=True
            )
        if terminal == "response.failed":
            raise _failure(getattr(getattr(response, "error", None), "code", None))
        output = [item.model_dump(exclude_none=True) for item in response.output]
        # Also support a compliant endpoint that sends completed text without
        # deltas. Never silently replay content that disagrees with live output.
        text = "".join(
            part.get("text", "")
            if part.get("type") == "output_text"
            else part.get("refusal", "")
            for item in output
            if item["type"] == "message"
            for part in item["content"]
        )
        emitted = "".join(text_parts)
        if text.startswith(emitted):
            if text[len(emitted) :]:
                yield LLMChunk(text_delta=text[len(emitted) :])
        elif terminal != "response.incomplete":
            raise LLMStreamError(
                "completed text disagrees with streamed text", retryable=True
            )
        # An incomplete snapshot can omit unfinished text already delivered
        # in deltas. Its declared limit/filter remains authoritative.
        if terminal == "response.incomplete":
            reason = getattr(
                getattr(response, "incomplete_details", None), "reason", None
            )
            raise LLMStreamError(
                "Responses output incomplete: " + str(reason),
                retryable=False,
                finish_reason="length" if reason == "max_output_tokens" else None,
                dropped_tool_calls=sum(
                    item["type"] == "function_call" for item in output
                ),
                category="request",
            )
        calls: list[ToolCall] = []
        call_ids: set[str] = set()
        for item in output:
            if item.get("status") in {"in_progress", "incomplete"}:
                raise LLMStreamError(
                    "completed response contains an unfinished item", retryable=True
                )
            if item["type"] == "function_call":
                if not item.get("call_id") or not item.get("name"):
                    raise LLMStreamError(
                        "function call is missing its identity", retryable=True
                    )
                if item["call_id"] in call_ids:
                    raise LLMStreamError(
                        "duplicate function call identity", retryable=True
                    )
                call_ids.add(item["call_id"])
                calls.append(
                    _ToolCallAccumulator(
                        id=item["call_id"],
                        name=item["name"],
                        args_fragments=[item["arguments"]],
                    ).build(cut_off=True)
                )
            elif item["type"] not in {"message", "reasoning"}:
                raise LLMStreamError(
                    "unsupported Responses output item",
                    retryable=False,
                    category="configuration",
                )
        yield LLMChunk(
            tool_calls=calls or None,
            finish_reason="tool_calls" if calls else "stop",
            responses_content=output,
        )

    def _report_usage(self, served_model: str | None, usage: Any) -> None:
        parsed = usage_from_responses(served_model or self.model, usage)
        if self.usage_sink is not None and parsed is not None:
            self.usage_sink(parsed)
