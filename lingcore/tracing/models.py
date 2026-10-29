"""Versioned, immutable trace values. No provider or exporter dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

Scalar = str | int | float | bool | None
SpanStatus = Literal[
    "ok", "error", "cancelled", "denied", "incomplete", "budget_exceeded"
]

# Values, especially exception messages and arbitrary tool arguments, must never
# accidentally become metadata. Adding an instrument requires an explicit schema.
COMMON: dict[str, type | tuple[type, ...]] = {
    "duration_ns": int,
    "error.type": str,
}
ATTRIBUTES: dict[str, dict[str, type | tuple[type, ...]]] = {
    "agent.turn": {
        "input_origin": str,
        "iterations": int,
        "outcome": str,
        "session.message_seq": int,
        "committed": bool,
    },
    "turn.rollback": {"reason": str, "session.message_seq": int},
    "turn.finalize": {"committed": bool, "outcome": str},
    "session.link": {"source_session_id": str, "through_seq": int, "relation": str},
    "prompt.compose": {"characters": int, "layers": int},
    "llm.request": {
        "request_id": str,
        "purpose": str,
        "model": str,
        "provider": str,
        "finish_reason": str,
        "ttft_ns": int,
        "first_delta_ns": int,
        "max_retries": int,
        "transport_attempts": int,
        "previous_request_id": str,
    },
    "request.usage": {
        "request_id": str,
        "purpose": str,
        "model": str,
        "availability": str,
        "input_tokens": int,
        "output_tokens": int,
        "cached_input_tokens": int,
        "reasoning_tokens": int,
        "cache_creation_input_tokens": int,
        "cache_creation_5m_tokens": int,
        "cache_creation_1h_tokens": int,
    },
    "tool.dispatch": {
        "tool.name": str,
        "tool.call_id": str,
        "request_id": str,
        "decision": str,
        "reason": str,
    },
    "tool.execute": {
        "tool.name": str,
        "tool.call_id": str,
        "request_id": str,
        "ok": bool,
        "output_bytes": int,
        "attachments": int,
    },
    "approval.wait": {"decision": str, "policy": str},
    "policy.decision": {"tool.name": str, "decision": str, "policy": str},
    "plugin.hook": {"plugin": str, "hook": str, "decision": str},
    "memory.compact": {"input_messages": int, "output_messages": int},
    "retry.backoff": {
        "attempt": int,
        "delay_seconds": (int, float),
        "previous_request_id": str,
    },
    "content.capture": {
        "artifact_id": str,
        "content.kind": str,
        "redacted": bool,
        "truncated": bool,
        "omitted_items": int,
    },
    "trace.cap": {},
    "evaluation.trial": {"case_id": str, "repetition": int},
    "execution.outcome": {
        "status": str,
        "cause": str,
        "limit_type": str,
        "provider_error_category": str,
    },
    "evaluation.grade": {"grader": str, "passed": bool},
}


def allow_attributes(name: str, attributes: Mapping[str, Scalar]) -> dict[str, Scalar]:
    if name not in ATTRIBUTES:
        raise ValueError("unknown trace operation")
    allowed = COMMON | ATTRIBUTES[name]
    clean: dict[str, Scalar] = {}
    for key, value in attributes.items():
        expected = allowed.get(key)
        if expected is None or (
            value is not None
            and (
                not isinstance(value, expected)
                or (isinstance(value, bool) and expected is not bool)
            )
        ):
            raise ValueError("trace attribute is not allowlisted or has the wrong type")
        if isinstance(value, str) and len(value) > 256:
            raise ValueError("trace metadata value exceeds its bound")
        clean[key] = value
    return clean


class TraceLink(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    relation: Literal["trial", "source", "retry"] = "source"


class SpanContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    parent_span_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    turn_id: str | None = None
    session_id: str | None = None
    trial_id: str | None = None
    execution_id: str | None = None
    links: tuple[TraceLink, ...] = ()


class TraceRecord(SpanContext):
    schema_version: Literal["lingcore.trace.v1"] = "lingcore.trace.v1"
    event_id: str
    run_id: str
    seq: int = Field(ge=1)
    timestamp: str
    elapsed_ns: int = Field(ge=0)
    kind: Literal["span.start", "span.end", "event"]
    name: str
    status: SpanStatus | None = None
    attributes: Mapping[str, Scalar] = Field(default_factory=dict)

    @field_validator("attributes")
    @classmethod
    def _immutable(cls, value: Mapping[str, Scalar]) -> Mapping[str, Scalar]:
        return MappingProxyType(dict(value))

    @field_serializer("attributes")
    def _serialize_attributes(self, value: Mapping[str, Scalar]) -> dict[str, Scalar]:
        return dict(value)


class TraceHealth(BaseModel):
    model_config = ConfigDict(frozen=True)
    complete: bool = True
    dropped_records: int = 0
    disk_errors: int = 0
    capped: bool = False
    flush_timed_out: bool = False
    written_records: int = 0
    written_bytes: int = 0
