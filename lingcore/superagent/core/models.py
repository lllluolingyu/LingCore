"""I/O-free domain records. Outside text is never a grant."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lingcore.errors import ToolError


class SlotsFull(ToolError):
    """Ordinary queue backpressure, not an effect failure."""


class Trigger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    external_id: str
    actor: str
    thread: str
    address: dict[str, str]
    text: str = Field(max_length=20000)
    addressed: bool = False
    bot: bool = False
    timestamp: float

    @property
    def key(self) -> str:
        return self.source + ":" + self.external_id


class Mandate(BaseModel):
    model_config = ConfigDict(frozen=True)
    policy_version: str
    strength: Literal["explicit", "inferred"]
    requester: str
    trigger: Trigger
    rules: dict[str, str] = Field(default_factory=dict)
    role: str
    inherited: frozenset[str]
    requested: frozenset[str]
    envelope_digest: str
    constraints_digest: str


class Approval(BaseModel):
    model_config = ConfigDict(frozen=True)
    actor: str
    timestamp: float
    tools: frozenset[str]
    envelope_digest: str


class Assignment(BaseModel):
    id: str
    status: Literal["pending", "running", "ended"] = "pending"
    awaiting_approval: bool = False
    created_at: float
    started_at: float | None = None
    ended_at: float | None = None
    followup: bool = False
    recovery: bool = False
    run_id: str | None = None
    worker_id: str | None = None
    session_id: str | None = None
    result: str | None = None
    input_cursor: int = 0


class Ticket(BaseModel):
    id: str
    name: str
    mandate: Mandate
    status: Literal["open", "ended"] = "open"
    withdrawn: bool = False
    approval: Approval | None = None
    prepare_started: bool = False
    prepared: bool = False
    assignments: list[Assignment] = Field(default_factory=list)
    inputs: list[Trigger] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    subscriptions: list[str] = Field(default_factory=list)
    question: str | None = None
    report: str | None = None


class Evaluation(BaseModel):
    id: str
    thread: str
    triggers: list[str]
    status: Literal["pending", "running", "ended", "unavailable"] = "pending"
    attempts: int = 0
    started_at: float | None = None
    resolved: list[str] = Field(default_factory=list)


class Effect(BaseModel):
    id: str
    kind: Literal["assessment", "launch", "delivery", "relay", "reply"]
    payload: dict[str, Any]
    created_at: float
    due_at: float
    attempts: int = 0
    status: Literal["pending", "running", "done"] = "pending"
    last_error: str | None = None


class State(BaseModel):
    router_session_id: str | None = None
    authority: dict[str, Any] = Field(default_factory=dict)
    triggers: dict[str, Trigger] = Field(default_factory=dict)
    tickets: dict[str, Ticket] = Field(default_factory=dict)
    evaluations: dict[str, Evaluation] = Field(default_factory=dict)
    effects: dict[str, Effect] = Field(default_factory=dict)
    thread_queues: dict[str, list[str]] = Field(default_factory=dict)
    source_positions: dict[str, str] = Field(default_factory=dict)
    sequence: int = 0
    seen: dict[str, float] = Field(default_factory=dict)
    last_maintenance: float = 0
