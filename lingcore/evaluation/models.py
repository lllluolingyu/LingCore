"""Native suite, case, trial, and normalized result contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lingcore.execution import ExecutionLimits, ExecutionOutcome
from lingcore.tracing import TraceHealth


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LockPolicy(StrictModel):
    required: bool = False
    capture: list[str] = Field(default_factory=list)


class LocalSource(StrictModel):
    kind: Literal["local"] = "local"
    cases: list[str] = Field(min_length=1)
    lock: LockPolicy = Field(default_factory=LockPolicy)


class Candidate(StrictModel):
    kind: Literal["lingcore"] = "lingcore"
    profile: str
    tool_binding: Literal["builtin"] = "builtin"
    endpoint_alias: str | None = None


class NativeExecutionLimits(ExecutionLimits):
    """Native defaults apply independently to every omitted limit."""

    wall_seconds: float | None = Field(default=60, gt=0)
    model_requests: int | None = Field(default=20, ge=0)
    tool_calls: int | None = Field(default=40, ge=0)


RetryCause = Literal["runner", "environment", "provider_transient"]


def default_retry_on() -> list[RetryCause]:
    return ["runner", "environment"]


class ExecutionConfig(StrictModel):
    runner: Literal["native"] = "native"
    repetitions: int = Field(default=1, ge=1, le=1000)
    concurrency: Literal[1] = 1
    infrastructure_retries: int = Field(default=0, ge=0, le=5)
    retry_on: list[RetryCause] = Field(default_factory=default_retry_on)
    state: Literal["fresh_per_execution"] = "fresh_per_execution"
    approval_policy: Literal["deny"] = "deny"
    limits: NativeExecutionLimits = Field(default_factory=NativeExecutionLimits)


class TracePolicy(StrictModel):
    required: bool = True
    content: Literal["metadata", "redacted"] = "metadata"
    max_bytes: int = Field(default=8388608, ge=4096)
    artifact_max_bytes: int = Field(default=1048576, ge=256)
    exports: list[str] = Field(default_factory=list, max_length=0)


class GradingPolicy(StrictModel):
    authority: Literal["local"] = "local"
    success: Literal["all_required"] = "all_required"


class ReportPolicy(StrictModel):
    headline: Literal["mean_case_success"] = "mean_case_success"
    include: list[str] = Field(
        default_factory=lambda: [
            "coverage",
            "rewards",
            "failures",
            "latency",
            "usage",
            "trace_health",
        ]
    )


class Suite(StrictModel):
    schema_version: Literal["lingcore.eval.suite.v1"]
    name: str
    track: Literal["development"] = "development"
    source: LocalSource
    agent: Candidate
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    trace: TracePolicy = Field(default_factory=TracePolicy)
    grading: GradingPolicy = Field(default_factory=GradingPolicy)
    report: ReportPolicy = Field(default_factory=ReportPolicy)


class Requirements(StrictModel):
    tools: list[str] = Field(default_factory=list)


class ScriptedTurn(StrictModel):
    user: str = Field(min_length=1)


class CaseInput(StrictModel):
    turns: list[ScriptedTurn] = Field(min_length=1)


class Fixture(StrictModel):
    workspace: Literal["empty"] = "empty"
    files: dict[str, str] = Field(default_factory=dict)


class ArtifactText(StrictModel):
    id: str
    kind: Literal["artifact_text"]
    required: bool = True
    path: str
    equals: str
    max_bytes: int = Field(default=1048576, ge=1, le=16777216)


class ToolAssertion(StrictModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPolicy(StrictModel):
    id: str
    kind: Literal["tool_policy"]
    required: bool = True
    require: list[ToolAssertion] = Field(default_factory=list)
    causal_order: list[tuple[str, str]] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)


GraderSpec = Annotated[ArtifactText | ToolPolicy, Field(discriminator="kind")]


class EvidenceDeclaration(StrictModel):
    artifacts: list[str] = Field(default_factory=list)


class Case(StrictModel):
    schema_version: Literal["lingcore.eval.case.v1"]
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    revision: int | str
    tags: list[str] = Field(default_factory=list)
    requires: Requirements = Field(default_factory=Requirements)
    input: CaseInput
    fixture: Fixture = Field(default_factory=Fixture)
    graders: list[GraderSpec] = Field(min_length=1)
    evidence: EvidenceDeclaration = Field(default_factory=EvidenceDeclaration)

    @model_validator(mode="after")
    def _grader_ids(self) -> Case:
        if len({g.id for g in self.graders}) != len(self.graders):
            raise ValueError("grader ids must be unique")
        if not any(g.required for g in self.graders):
            raise ValueError("a case needs at least one required grader")
        return self


class TrialSpec(StrictModel):
    trial_id: str
    case_id: str
    repetition: int


class GraderResult(StrictModel):
    id: str
    kind: str
    required: bool
    status: Literal["valid", "not_evaluable"]
    passed: bool | None = None
    score: float | None = None
    reason: str | None = None
    error_type: str | None = None
    version: int = 1


class Grade(StrictModel):
    status: Literal["valid", "invalid", "not_run"]
    passed: bool | None = None
    scores: dict[str, float | None] = Field(default_factory=dict)
    graders: list[GraderResult] = Field(default_factory=list)
    reason: str | None = None


EvaluationPhase = Literal[
    "preflight", "setup", "execute", "collect", "grade", "cleanup", "runner"
]


class EvaluationDiagnostic(StrictModel):
    phase: EvaluationPhase
    code: str
    message: str
    error_type: str | None = None


class UsageSummary(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    coverage: Literal["complete", "incomplete"] = "complete"
    estimated_cost_usd: float | None = None


class Comparison(StrictModel):
    track: Literal["development"] = "development"
    locked: bool = False
    benchmark_compatible: Literal[False] = False


class TrialResult(StrictModel):
    schema_version: Literal["lingcore.eval-result.v1"] = "lingcore.eval-result.v1"
    trial_id: str
    execution_id: str
    case_id: str
    case_revision: int | str
    repetition: int
    locked: bool = False
    attempt: int = Field(default=1, ge=1)
    lock_digest: str | None = None
    plan_digest: str | None = None
    infrastructure_failure: RetryCause | None = None
    served_models: list[str] = Field(default_factory=list)
    benchmark_compatible: Literal[False] = False
    execution: ExecutionOutcome
    grade: Grade
    usage: UsageSummary
    trace: TraceHealth
    content: Literal["metadata", "redacted"]
    elapsed_ms: int = 0
    cleanup_failed: bool = False
    diagnostics: list[EvaluationDiagnostic] = Field(default_factory=list)
    comparison: Comparison = Field(default_factory=Comparison)
