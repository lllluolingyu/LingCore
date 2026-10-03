"""Anthropic SDK backend for LLMClient.

Provides native Anthropic Messages API support for models that don't work well
through OpenAI-compatible gateways. Implements the same streaming interface as
the OpenAI-based LLMClient so the agent loop needs no changes.

Tool call handling: Anthropic streams indexed tool-use blocks and partial JSON
input deltas. This backend reassembles them and yields a single terminal chunk
with all tool calls assembled.

Thinking is configured through sampling.thinking (enabled with a token budget,
or adaptive). Returned thinking blocks, signatures, and redacted data are always
preserved for tool continuation, including models that think by default. The
preserve_reasoning flag additionally exposes thinking text as reasoning_delta;
it neither enables thinking nor controls the required signed replay state.

Prompt caching is enabled by default with explicit five-minute breakpoints on
tools, system instructions, and the last two user/tool-result turns. Explicit
breakpoints also work on gateways without top-level automatic caching support.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Collection
from copy import deepcopy
from itertools import groupby
from typing import Any

from anthropic import AsyncAnthropic, Timeout

from lingcore.errors import LLMStreamError
from lingcore.llm import LLMChunk, _close_quietly, _describe, _ToolCallAccumulator
from lingcore.message import NATIVE_MODALITIES, Message
from lingcore.usage import UsageSink

# Connect phase fails fast even when timeout is generous
_MAX_CONNECT_SECONDS = 10.0


def _cache_request_prefixes(kwargs: dict[str, Any]) -> None:
    """Add at most four breakpoints without changing caller-owned content.

    Native cache settings own the whole policy: mixing automatic caching or
    user-supplied breakpoints with ours can exceed the API's four-slot limit
    or mix incompatible TTLs. Body overrides also bypass our default policy.
    """
    extra_body = kwargs.get("extra_body") or {}
    if "cache_control" in kwargs or any(
        key in extra_body for key in ("cache_control", "tools", "system", "messages")
    ):
        return

    def as_blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        return deepcopy(content)

    tools = deepcopy(kwargs.get("tools") or [])
    system = as_blocks(kwargs.get("system") or "")
    messages = [
        {**message, "content": as_blocks(message["content"])}
        for message in kwargs["messages"]
    ]
    blocks = [*tools, *system]
    for message in messages:
        blocks.extend(message["content"])
    if any("cache_control" in block for block in blocks):
        return

    def mark(block: dict[str, Any]) -> None:
        block["cache_control"] = {"type": "ephemeral"}

    if tools:
        mark(tools[-1])
        kwargs["tools"] = tools
    for block in reversed(system):
        if block.get("type") == "text" and block.get("text"):
            mark(block)
            kwargs["system"] = system
            break

    # Consecutive user messages form one API turn (e.g. parallel tool results
    # and hoisted attachments). Keep the previous turn's endpoint explicitly
    # discoverable even if the newest batch exceeds the 20-block lookback.
    marked = 0
    for role, turn in groupby(reversed(messages), key=lambda message: message["role"]):
        if role != "user":
            continue
        candidates = (
            block
            for message in turn
            for block in reversed(message["content"])
            if block.get("type") in {"text", "image", "document", "tool_result"}
            and (block["type"] != "text" or block.get("text"))
        )
        if (candidate := next(candidates, None)) is not None:
            mark(candidate)
            marked += 1
            if marked == 2:
                break
    kwargs["messages"] = messages


class AnthropicLLMClient:
    """Anthropic Messages API backend with the same interface as LLMClient."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        sampling: dict[str, Any] | None = None,
        max_retries: int = 10,
        timeout: float = 120.0,
        http_client: Any = None,
        modalities: Collection[str] | None = None,
        prompt_cache_key: str | None = None,
        stream_usage: bool = True,
        usage_sink: UsageSink | None = None,
        preserve_reasoning: bool = False,
        prompt_caching: bool = True,
    ) -> None:
        self.model = model
        self._preserve_reasoning = preserve_reasoning
        self._stream_usage = stream_usage
        self.usage_sink = usage_sink
        self.sampling = sampling or {}
        self._prompt_cache_key = prompt_cache_key
        self._prompt_caching = prompt_caching

        narrowed = None if modalities is None else frozenset(modalities)
        self._modalities = None if narrowed == NATIVE_MODALITIES else narrowed
        self._max_retries = max(0, max_retries)

        sdk_timeout = Timeout(timeout, connect=min(timeout, _MAX_CONNECT_SECONDS))
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": self._max_retries,
            "timeout": sdk_timeout,
        }
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = AsyncAnthropic(**client_kwargs)

    async def _open_stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "system": system,
            "stream": True,
            # Unlike chat completions, Messages requires an explicit limit.
            "max_tokens": 4096,
            **self.sampling,
        }
        if tools:
            kwargs["tools"] = tools

        # SDK 1.x removed these named parameters, but older models and native
        # gateways can still accept them in the request body. Preserve the
        # profile's pass-through behavior and extra_body's override precedence.
        legacy_sampling = {
            name: kwargs.pop(name)
            for name in ("temperature", "top_p", "top_k")
            if name in kwargs
        }
        if legacy_sampling:
            kwargs["extra_body"] = {
                **legacy_sampling,
                **(kwargs.get("extra_body") or {}),
            }

        if self._prompt_caching:
            _cache_request_prefixes(kwargs)
        return await self._client.messages.create(**kwargs)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream a single assistant turn using Anthropic Messages API.

        Yields text deltas as they arrive, then exactly one terminal chunk
        carrying any assembled tool calls plus the finish reason.
        """
        # Anthropic separates system prompt from messages
        system_prompts = []
        wire_messages = []
        for m in messages:
            anthropic_msg = m.to_anthropic(attachment_modalities=self._modalities)
            if anthropic_msg.get("role") == "system":
                if anthropic_msg["content"]:
                    system_prompts.append(anthropic_msg["content"])
            elif anthropic_msg["role"] == "assistant" and not anthropic_msg["content"]:
                # Empty end_turn replies are valid responses but invalid
                # history. Filter the wire representation, keeping stored
                # messages and nonempty tool/thinking blocks unchanged.
                continue
            else:
                wire_messages.append(anthropic_msg)

        # A compacted session may begin with an assistant message.
        if wire_messages and wire_messages[0].get("role") == "assistant":
            wire_messages.insert(0, {"role": "user", "content": "[session start]"})

        wire_tools = None
        if tools:
            wire_tools = [self._convert_tool_to_anthropic(t) for t in tools]

        try:
            stream = await self._open_stream(
                "\n\n".join(system_prompts), wire_messages, wire_tools
            )
        except Exception as e:
            raise LLMStreamError(
                f"request failed: {_describe(e)}", retryable=False
            ) from e

        accumulators: dict[int, _ToolCallAccumulator] = {}
        content_blocks: dict[int, dict[str, Any]] = {}
        closed_blocks: set[int] = set()
        finish_reason: str | None = None
        usage_data: dict[str, Any] = {}
        served_model: str | None = None

        try:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        served_model = getattr(msg, "model", None)
                        if hasattr(msg, "usage"):
                            usage_data.update(self._extract_usage(msg.usage))

                elif event_type == "content_block_start":
                    block = event.content_block
                    content_blocks[event.index] = block.model_dump(
                        mode="json", exclude_none=True
                    )
                    if block.type == "tool_use":
                        acc = _ToolCallAccumulator(id=block.id, name=block.name)
                        if block.input:
                            acc.args_fragments.append(json.dumps(block.input))
                        accumulators[event.index] = acc
                    elif block.type == "text" and block.text:
                        yield LLMChunk(text_delta=block.text)
                    elif (
                        block.type == "thinking"
                        and self._preserve_reasoning
                        and block.thinking
                    ):
                        yield LLMChunk(reasoning_delta=block.thinking)

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", None)
                        content_block = content_blocks.get(event.index)
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                if content_block is not None:
                                    content_block["text"] += text
                                yield LLMChunk(text_delta=text)
                        elif delta_type == "input_json_delta":
                            if (tool_acc := accumulators.get(event.index)) is not None:
                                tool_acc.args_fragments.append(delta.partial_json)
                        elif (
                            delta_type == "thinking_delta" and content_block is not None
                        ):
                            content_block["thinking"] += delta.thinking
                            if self._preserve_reasoning and delta.thinking:
                                yield LLMChunk(reasoning_delta=delta.thinking)
                        elif (
                            delta_type == "signature_delta"
                            and content_block is not None
                        ):
                            content_block["signature"] += delta.signature

                elif event_type == "content_block_stop":
                    closed_blocks.add(event.index)

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        stop_reason = getattr(delta, "stop_reason", None)
                        if stop_reason:
                            finish_reason = self._map_finish_reason(stop_reason)
                    usage = getattr(event, "usage", None)
                    if usage:
                        usage_data.update(self._extract_usage(usage))

        except Exception as e:
            raise LLMStreamError(
                f"stream interrupted: {_describe(e)}", retryable=True
            ) from e
        finally:
            # Also release the connection on cancellation or consumer aclose().
            await _close_quietly(stream)
            self._report_usage(served_model, usage_data)

        if finish_reason is None:
            raise LLMStreamError(
                "stream ended without a stop reason (response truncated)",
                retryable=True,
            )

        thinking_blocks = {
            index: block
            for index, block in content_blocks.items()
            if block.get("type") in {"thinking", "redacted_thinking"}
        }
        incomplete_thinking = any(
            index not in closed_blocks
            or (block["type"] == "thinking" and not block.get("signature"))
            for index, block in thinking_blocks.items()
        )
        has_response = bool(accumulators) or any(
            block.get("type") == "text" and block.get("text")
            for block in content_blocks.values()
        )
        if thinking_blocks and (incomplete_thinking or not has_response):
            # A provider-declared limit won't improve by re-requesting the
            # same prompt and budget. Avoid repeated billed thinking-only turns.
            if finish_reason == "length":
                raise LLMStreamError(
                    "thinking exhausted max_tokens before completing a response; "
                    "increase sampling.max_tokens or reduce the thinking budget",
                    retryable=False,
                )
            if finish_reason == "model_context_window_exceeded":
                raise LLMStreamError(
                    "thinking exhausted the context window; shorten or compact "
                    "the conversation before retrying",
                    retryable=False,
                )
        if incomplete_thinking:
            raise LLMStreamError(
                "stream ended with incomplete thinking (response truncated)",
                retryable=True,
            )

        if accumulators and finish_reason in {
            "length",
            "model_context_window_exceeded",
        }:
            # A limit can cut off tool JSON even after content_block_stop.
            # Reject the whole batch before malformed arguments become {},
            # which could execute tools with unintended defaults. Retrying
            # the same request cannot recover an exhausted budget.
            hint = (
                "max_tokens was exhausted; increase sampling.max_tokens"
                if finish_reason == "length"
                else "the context window was exhausted; shorten or compact the conversation"
            )
            raise LLMStreamError(
                f"tool call response was truncated: {hint}", retryable=False
            )

        tool_calls = []
        for index in sorted(accumulators):
            call = accumulators[index].build()
            tool_calls.append(call)
            content_blocks[index]["input"] = call.arguments

        yield LLMChunk(
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            anthropic_content=(
                [content_blocks[index] for index in sorted(content_blocks)]
                if thinking_blocks
                else None
            ),
        )

    def _convert_tool_to_anthropic(self, openai_tool: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI tool format to Anthropic tool format."""
        func = openai_tool.get("function", {})
        return {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }

    def _map_finish_reason(self, anthropic_reason: str) -> str:
        """Map Anthropic stop reasons to OpenAI-compatible finish reasons."""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        return mapping.get(anthropic_reason, anthropic_reason)

    def _extract_usage(self, usage: Any) -> dict[str, Any]:
        """Extract usage data from Anthropic usage object."""
        # message_delta usage is partial and cumulative. Missing/None fields
        # must not erase counts from message_start or an earlier delta.
        return {
            name: value
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "output_tokens_details",
            )
            if (value := getattr(usage, name, None)) is not None
        }

    def _report_usage(
        self, served_model: str | None, usage_data: dict[str, Any]
    ) -> None:
        if self.usage_sink is None or not usage_data:
            return
        from lingcore.usage import usage_from_anthropic

        parsed = usage_from_anthropic(served_model or self.model, usage_data)
        if parsed is not None:
            self.usage_sink(parsed)
