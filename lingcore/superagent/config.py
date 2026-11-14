"""Operator policy and service configuration; never populated from messages."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from lingcore.delegation.config import StrictModel
from lingcore.delegation.models import digest


class Rule(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    instruction: str = Field(min_length=1, max_length=4000)
    role: str | None = None

    def revision(self) -> str:
        return digest(self.model_dump())


class SourceConfig(StrictModel):
    config: str | None = None
    enabled: bool = True
    allowlist: list[str] = Field(default_factory=list)
    bots: list[str] = Field(default_factory=list)
    interval_seconds: float = Field(default=300, ge=1)
    options: dict[str, Any] = Field(default_factory=dict)


class Policy(StrictModel):
    accept_mode: Literal["off", "observe", "auto"] = "observe"
    approvers: list[str] = Field(default_factory=list)
    approval_source: Literal["lark", "telegram"] | None = None
    rules: list[Rule] = Field(default_factory=list)
    tools: list[str] | None = None

    def revision(self) -> str:
        return digest(self.model_dump())


class RouterConfig(StrictModel):
    role: str = "router"
    eval_timeout_seconds: float = Field(default=300, gt=0)


class ExecutorConfig(StrictModel):
    role: str = "executor"
    proactive_role: str = "observer"
    max_workers: int = Field(default=3, ge=1, le=32)
    approval_timeout_seconds: float = Field(default=86400, gt=0)
    prepare: str | None = None
    background_limit_seconds: float = Field(default=1800, gt=0, le=86400)


class SuperagentConfig(StrictModel):
    store: str = "superagent.sqlite3"
    panel_actor: str | None = None
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    policy: Policy = Field(default_factory=Policy)
    router: RouterConfig = Field(default_factory=RouterConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    retention_days: int = Field(default=30, ge=1)
    max_closed_tickets: int = Field(default=1000, ge=1)
    max_history: int = Field(default=10000, ge=100)
