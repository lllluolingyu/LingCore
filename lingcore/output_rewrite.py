"""Experimental wording pass, isolated from the agent's tools and conversation."""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from lingcore.errors import INCOMPLETE_FINISH_REASONS, LLMStreamError
from lingcore.message import Message
from lingcore.tracing.requests import StreamingClient, closing_stream

if TYPE_CHECKING:
    from lingcore.config import OutputRewriteCfg

_LOG = logging.getLogger(__name__)


class OutputRewriter:
    def __init__(self, client: StreamingClient, config: OutputRewriteCfg) -> None:
        self.client, self.config = client, config

    async def rewrite(self, content: str) -> str:
        """Return a complete rewrite or the original; Stop/budgets propagate."""
        if not content.strip() or len(content) > self.config.max_input_chars:
            return content
        messages = [
            Message.system(self.config.prompt),
            Message.user(content),
        ]
        retries = self.config.llm.stream_retries if self.config.llm else 0
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                for attempt in range(retries + 1):
                    try:
                        return await self._collect(messages)
                    except LLMStreamError as exc:
                        if (
                            not exc.retryable
                            or exc.finish_reason in INCOMPLETE_FINISH_REASONS
                            or attempt == retries
                        ):
                            raise
                        await asyncio.sleep(min(0.25 * 2**attempt, 2))
        except Exception as exc:
            # Providers may echo credentials or request text in errors.
            _LOG.warning(
                "Experimental output rewrite failed (%s); using original reply",
                type(exc).__name__,
            )
        return content

    async def _collect(self, messages: list[Message]) -> str:
        parts: list[str] = []
        size = 0
        finish: str | None = None
        async with closing_stream(self.client.stream(messages, tools=None)) as stream:
            async for chunk in stream:
                if chunk.tool_calls:
                    raise ValueError("rewrite returned tool calls")
                if chunk.text_delta:
                    size += len(chunk.text_delta)
                    if size > self.config.max_output_chars:
                        raise ValueError("rewrite exceeded output limit")
                    parts.append(chunk.text_delta)
                finish = chunk.finish_reason or finish
        result = "".join(parts)
        if finish != "stop" or not result.strip():
            raise ValueError("rewrite was empty or incomplete")
        return result


def replace_reply_text(message: Message, content: str) -> None:
    """Keep native replay text in sync without losing signed/opaque reasoning.

    Text annotations belong to the original wording and cannot be reused after
    an edit. Non-text blocks retain their original order and exact payload.
    Only called on a completed, tool-free reply, before it enters memory.
    """
    if content == message.content:
        return
    message.content = content
    if message.anthropic_content is not None:
        message.anthropic_content = _replace_parts(
            message.anthropic_content, content, "text"
        )
    if message.responses_content is not None:
        items: list[dict[str, Any]] = []
        replaced = False
        for original in message.responses_content:
            item = deepcopy(original)
            if item.get("type") == "message":
                if replaced:
                    continue
                item["content"] = [{"type": "output_text", "text": content}]
                replaced = True
            items.append(item)
        if not replaced:
            items.append({"role": "assistant", "content": content, "type": "message"})
        message.responses_content = items


def _replace_parts(
    parts: list[dict[str, Any]], content: str, kind: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    replaced = False
    for part in parts:
        if part.get("type") == kind:
            if not replaced:
                result.append({"type": kind, "text": content})
                replaced = True
        else:
            result.append(deepcopy(part))
    if not replaced:
        result.append({"type": kind, "text": content})
    return result
