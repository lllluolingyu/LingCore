"""Agent events.

The loop emits a stream of these; frontends render events and never touch
LLM internals. This decoupling is what lets a CLI today and a web/Discord
adapter later drive the exact same ``Agent.run`` without changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from lingcore.message import ToolCall, ToolResult


@dataclass(slots=True)
class TextDelta:
    """A streamed chunk of assistant text."""

    text: str


@dataclass(slots=True)
class ToolCallStarted:
    """A tool is about to be executed."""

    call: ToolCall


@dataclass(slots=True)
class ToolResultEvent:
    """A tool finished (successfully or not)."""

    result: ToolResult


@dataclass(slots=True)
class Final:
    """The assistant produced its final, tool-free reply for this turn."""

    content: str


@dataclass(slots=True)
class Error:
    """A turn-level error the frontend should surface."""

    message: str


@dataclass(slots=True)
class SkillActivated:
    """A skill was activated or deactivated during a turn."""

    name: str
    active: bool


AgentEvent = (
    TextDelta | ToolCallStarted | ToolResultEvent | Final | Error | SkillActivated
)
