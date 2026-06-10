"""Typed exception hierarchy for LingCore.

The agent loop branches on these: a ``ToolError`` is an expected, in-domain
failure that gets fed back to the model as a tool result, whereas an
unexpected ``Exception`` is contained but surfaced verbatim. ``ConfigError``
and ``MaxIterationsError`` propagate to the caller/frontend.
"""

from __future__ import annotations


class LingCoreError(Exception):
    """Base class for all LingCore-raised errors."""


class ConfigError(LingCoreError):
    """A profile/framework config is invalid or references a missing secret."""


class ToolError(LingCoreError):
    """An expected, in-domain tool failure.

    Raised by tools for conditions the model can reasonably recover from
    (missing file, path escaping the workspace, ambiguous edit, etc.). The
    loop converts this into a ``ToolResult(ok=False)`` and feeds it back to
    the model rather than crashing the run.
    """


class LLMStreamError(LingCoreError):
    """A streamed model request failed.

    Part of the LLM-seam contract: any ``LLMClient``-shaped backend raises
    this so the agent loop can react without knowing the SDK underneath.
    ``retryable=True`` means re-requesting the turn is worthwhile — the
    failure interrupted or truncated an in-flight stream, which no SDK-level
    retry ever covers. ``retryable=False`` means retrying cannot help: the
    request itself is invalid, or opening the stream already exhausted the
    backend's own (header-aware) retry budget.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class MaxIterationsError(LingCoreError):
    """The agent loop hit its iteration cap without producing a final reply."""


class SessionError(LingCoreError):
    """A session-store operation failed (unknown/ambiguous id, schema mismatch)."""
