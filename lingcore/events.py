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
class StreamRetry:
    """The in-flight model response failed and is being re-requested.

    Whatever this turn had already streamed (``discarded_chars`` of text) is
    void — the reply will be regenerated from scratch, and may differ.
    Frontends should mark the rupture so a user never reads the partial and
    the regenerated text as one continuous reply.
    """

    attempt: int
    max_attempts: int
    reason: str
    discarded_chars: int = 0


@dataclass(slots=True)
class Error:
    """A turn-level error the frontend should surface."""

    message: str


@dataclass(slots=True)
class SkillActivated:
    """A skill was activated or deactivated during a turn."""

    name: str
    active: bool


@dataclass(slots=True)
class Compacted:
    """Old history was summarized into a compact note at the start of a turn.

    A turn-boundary event: the working set was over ``compact_at_ratio`` of the
    token budget, so ``summarized_messages`` older messages were replaced by one
    summary, shrinking the window from ``before_tokens`` to ``after_tokens``.
    Frontends surface it so a user understands why earlier turns now read as a
    summary. The full history is still preserved in the session store.
    """

    summarized_messages: int
    before_tokens: int
    after_tokens: int


AgentEvent = (
    TextDelta
    | ToolCallStarted
    | ToolResultEvent
    | StreamRetry
    | Final
    | Error
    | SkillActivated
    | Compacted
)
