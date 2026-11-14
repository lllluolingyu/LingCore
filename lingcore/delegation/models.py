"""Immutable authority and durable worker/run values."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WorkspaceMode = Literal["shared", "frozen", "snapshot", "scratch", "ticket"]
NodeClass = Literal["pure", "frozen", "live", "effecting"]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode()
    ).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tools: frozenset[str]
    high_risk: frozenset[str] = frozenset()
    workspace: WorkspaceMode = "shared"
    depth: int = 1
    confirm: Literal["broker", "deny"] = "broker"
    constraints_digest: str = ""

    def identity(self) -> str:
        return digest(
            self.model_dump(mode="json")
            | {"tools": sorted(self.tools), "high_risk": sorted(self.high_risk)}
        )


class UsageSummary(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    complete: bool = True


class WorkerReport(BaseModel):
    worker_id: str
    role: str
    label: str
    status: Literal[
        "completed", "blocked", "failed", "cancelled", "budget_exceeded", "timed_out"
    ]
    result: Any = None
    summary: str = ""
    question: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    session_id: str | None = None
    error: str | None = None


class RunRecord(BaseModel):
    id: str
    parent_session_id: str | None = None
    workspace: str | None = None
    anchor_seq: int = -1
    kind: Literal["delegate", "workflow", "ticket"] = "delegate"
    status: str = "pending"
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    started_at: float = 0
    background: bool = False
    name: str = ""
    spec: dict[str, Any] | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    request: str = ""
    limits: dict[str, Any] = Field(default_factory=dict)
    input_revision: str | None = None
    source_revision: str | None = None
    input_captured_at: str | None = None
    skipped: list[str] = Field(default_factory=list)
    result: Any = None
    usage: UsageSummary = Field(default_factory=UsageSummary)
    agents: int = 0
    tool_calls: int = 0
    wall_seconds_spent: float = 0
    reason: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)


class NodeRecord(BaseModel):
    run_id: str
    identity: str
    key: str
    worker_id: str
    role: str
    classification: NodeClass
    envelope: Envelope
    status: str = "running"
    report: WorkerReport | None = None
    session_id: str | None = None
    prompt: str = ""
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    artifacts_digest: dict[str, str] = Field(default_factory=dict)
    report_digest: str | None = None
    model_config = ConfigDict(populate_by_name=True)
