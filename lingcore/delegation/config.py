"""Strict operator configuration for delegated work."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunLimits(StrictModel):
    max_agents: int = Field(default=50, ge=1, le=4096)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    reported_tokens: int | None = Field(default=1_000_000, ge=0)
    model_requests: int | None = Field(default=2000, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    wall_seconds: float | None = Field(default=3600, gt=0)
    usage_missing: Literal["ignore", "stop"] = "ignore"

    def narrow(self, values: dict[str, Any]) -> RunLimits:
        candidate = self.model_validate(self.model_dump() | values)
        for key, old in self.model_dump().items():
            new = getattr(candidate, key)
            if key == "usage_missing":
                if old == "stop" and new != "stop":
                    raise ValueError("usage_missing may only tighten to stop")
            elif old is not None and (new is None or new > old):
                raise ValueError(f"limits.{key} may only narrow")
        return candidate

    def execution(self) -> Any:
        from lingcore.execution import ExecutionLimits

        return ExecutionLimits(
            agents=self.max_agents,
            **self.model_dump(exclude={"max_agents", "max_concurrency"}),
        )


class WorkspaceConfig(StrictModel):
    snapshot_max_bytes: int = Field(default=268435456, gt=0)
    snapshot_max_files: int = Field(default=20000, gt=0)
    keep_snapshots: bool = False


class RetentionConfig(StrictModel):
    max_age_days: int = Field(default=30, ge=1)
    max_runs: int = Field(default=500, ge=1)


class DelegationConfig(StrictModel):
    roles_dir: str = "roles"
    default_role: str = "general"
    max_depth: int = Field(default=1, ge=1, le=4)
    max_workers_per_turn: int = Field(default=8, ge=1, le=4096)
    confirm_over_agents: int = Field(default=4, ge=0)
    background: bool = True
    limits: RunLimits = Field(default_factory=RunLimits)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    @field_validator("roles_dir")
    @classmethod
    def safe_path(cls, value: str) -> str:
        from lingcore.plugins.manifest import relative_component

        return relative_component(value)

    @field_validator("models")
    @classmethod
    def models_valid(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        from lingcore.config import LLMCfg

        for config in value.values():
            LLMCfg.model_validate(config)
        return value


class WorkflowConfig(StrictModel):
    dir: str = "workflows"
    inline: bool = True

    @field_validator("dir")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return DelegationConfig.safe_path(value)
