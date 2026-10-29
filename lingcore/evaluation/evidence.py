"""Bounded, content-addressed evidence, independent of agent execution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import Field

from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.graders import (
    Artifact,
    TrialEvidence,
    captured_content,
    collect_artifact,
    evidence_needs,
)
from lingcore.evaluation.identity import Digest, digest
from lingcore.evaluation.models import ArtifactText, Case, StrictModel
from lingcore.evaluation.storage import (
    read_bytes,
    read_json,
    write_bytes,
    write_json,
)
from lingcore.paths import PathEscapeError
from lingcore.tracing import TraceHealth
from lingcore.tracing.inspect import inspect_trace


class ArtifactEvidence(StrictModel):
    path: str
    status: Literal["present", "missing", "unsafe"]
    digest: Digest | None = None
    bytes: int = Field(default=0, ge=0)
    max_bytes: int = Field(default=1048576, ge=1, le=16777216)


class EvidenceManifest(StrictModel):
    schema_version: Literal["lingcore.eval-evidence.v1"] = "lingcore.eval-evidence.v1"
    trace_complete: bool
    content: Literal["metadata", "redacted"]
    artifacts: list[ArtifactEvidence]


def collect(
    workspace: Path,
    execution: Path,
    case: Case,
    content: Literal["metadata", "redacted"],
    health: TraceHealth,
) -> EvidenceManifest:
    limits = {Path(path).as_posix(): 1048576 for path in case.evidence.artifacts}
    for grader in case.graders:
        if isinstance(grader, ArtifactText):
            path = Path(grader.path).as_posix()
            limits[path] = max(limits.get(path, 0), grader.max_bytes)
    artifacts = []
    for path, limit in sorted(limits.items()):
        artifact = collect_artifact(
            workspace,
            ArtifactText(
                id="collect",
                kind="artifact_text",
                path=path,
                equals="",
                max_bytes=limit,
            ),
        )
        if artifact.data is None:
            artifacts.append(
                ArtifactEvidence(
                    path=path,
                    status="missing"
                    if artifact.issue == "missing_artifact"
                    else "unsafe",
                    max_bytes=limit,
                )
            )
            continue
        identity = digest(artifact.data)
        file = execution / "artifacts" / (identity.removeprefix("sha256:") + ".bin")
        try:
            write_bytes(file, artifact.data)
        except FileExistsError:
            if read_bytes(file, max_bytes=limit) != artifact.data:
                raise EvaluationConfigError(
                    "artifact digest collision or corrupt evidence", "invalid_evidence"
                ) from None
        artifacts.append(
            ArtifactEvidence(
                path=path,
                status="present",
                digest=identity,
                bytes=len(artifact.data),
                max_bytes=limit,
            )
        )
    manifest = EvidenceManifest(
        trace_complete=health.complete, content=content, artifacts=artifacts
    )
    write_json(execution / "evidence.json", manifest.model_dump(mode="json"))
    return manifest


def load_manifest(execution: Path) -> EvidenceManifest:
    try:
        return EvidenceManifest.model_validate(read_json(execution / "evidence.json"))
    except (OSError, ValueError, PathEscapeError):
        raise EvaluationConfigError(
            "execution has no valid preserved evidence", "missing_evidence"
        ) from None


def check_needs(case: Case, manifest: EvidenceManifest) -> None:
    paths = {a.path: a for a in manifest.artifacts}
    for grader in case.graders:
        needs = evidence_needs([grader])
        entry = (
            paths.get(Path(grader.path).as_posix())
            if isinstance(grader, ArtifactText)
            else None
        )
        if (
            set(needs["artifacts"]) - paths.keys()
            or needs["content"]
            and manifest.content != "redacted"
            or isinstance(grader, ArtifactText)
            and entry is not None
            and entry.status == "unsafe"
            and grader.max_bytes > entry.max_bytes
        ):
            raise EvaluationConfigError(
                f"case {case.id}, grader {grader.id}: required evidence was never preserved",
                "evidence_not_preserved",
            )


def load(
    execution: Path, case: Case, *, artifact_max_bytes: int = 1048576
) -> TrialEvidence:
    manifest = load_manifest(execution)
    check_needs(case, manifest)
    paths = {a.path: a for a in manifest.artifacts}
    artifacts = []
    for grader in case.graders:
        if not isinstance(grader, ArtifactText):
            continue
        entry = paths[Path(grader.path).as_posix()]
        data = None
        issue = None
        if entry.status == "present":
            if entry.digest is None:
                raise EvaluationConfigError(
                    "present evidence has no digest", "invalid_evidence"
                )
            data = read_bytes(
                execution
                / "artifacts"
                / (entry.digest.removeprefix("sha256:") + ".bin"),
                max_bytes=entry.max_bytes,
            )
            if digest(data) != entry.digest or len(data) != entry.bytes:
                raise EvaluationConfigError(
                    "preserved artifact failed integrity check", "invalid_evidence"
                )
            if len(data) > grader.max_bytes:
                data, issue = None, "unsafe_or_oversized_artifact"
        else:
            issue = (
                "missing_artifact"
                if entry.status == "missing"
                else "unsafe_or_oversized_artifact"
            )
        artifacts.append(Artifact(grader.id, data, issue))
    trace = None
    captured: tuple[bytes, ...] = ()
    try:
        trace = inspect_trace(execution / "trace")
        if not manifest.trace_complete:
            trace = replace(trace, complete=False)
        captured = captured_content(
            execution / "trace", trace, max_bytes=artifact_max_bytes
        )
    except (OSError, ValueError, PathEscapeError):
        pass
    return TrialEvidence(tuple(artifacts), trace, captured)
