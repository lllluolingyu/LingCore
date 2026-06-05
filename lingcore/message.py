"""Canonical message data model.

Everything in LingCore speaks ``Message`` — the loop, memory, the LLM client,
and frontends. This module depends on nothing else in the package so it can
sit at the bottom of the dependency graph. ``Message.to_openai`` is the single
place that knows the chat-completions wire shape, keeping the OpenAI coupling
contained.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A single tool invocation requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class ToolResult(BaseModel):
    """The outcome of executing one ``ToolCall``.

    ``ok=False`` marks an in-domain failure (a ``ToolError``); the content is
    still fed back to the model so it can recover.
    """

    call_id: str
    name: str
    content: str
    ok: bool = True


class Message(BaseModel):
    """One turn in a conversation.

    A role="assistant" message may carry ``tool_calls``. A role="tool"
    message carries ``tool_call_id`` + ``name`` linking it to the call it
    answers — OpenAI rejects a tool message that does not follow a matching
    assistant tool_call, so the two must never be separated (see WindowMemory).
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    # --- constructors -------------------------------------------------
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str = "", tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Message:
        return cls(
            role="tool",
            content=result.content,
            tool_call_id=result.call_id,
            name=result.name,
        )

    # --- wire format --------------------------------------------------
    def to_openai(self) -> dict[str, Any]:
        """Render to a chat-completions message dict."""
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


class Conversation(BaseModel):
    """An ordered list of messages. A thin convenience wrapper."""

    messages: list[Message] = Field(default_factory=list)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def to_openai(self) -> list[dict[str, Any]]:
        return [m.to_openai() for m in self.messages]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):  # type: ignore[override]
        return iter(self.messages)
