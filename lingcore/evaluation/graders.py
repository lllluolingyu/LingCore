"""Deterministic measurements over collected immutable evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lingcore.evaluation.models import (
    ArtifactText,
    Grade,
    GraderResult,
    GraderSpec,
    ToolAssertion,
    ToolPolicy,
)
from lingcore.evaluation.storage import directory as confined
from lingcore.paths import PathEscapeError
from lingcore.tracing.inspect import TraceInspection


@dataclass(frozen=True)
class Artifact:
    grader_id: str
    data: bytes | None
    issue: str | None = None


@dataclass(frozen=True)
class TrialEvidence:
    artifacts: tuple[Artifact, ...]
    trace: TraceInspection | None
    captured: tuple[bytes, ...]


class Grader(Protocol):
    async def grade(self, evidence: TrialEvidence) -> GraderResult: ...


def collect_artifact(workspace: Path, spec: ArtifactText) -> Artifact:
    path = Path(spec.path)
    try:
        with confined(workspace / path.parent) as directory:
            data = directory.read_regular(path.name, max_bytes=spec.max_bytes)
        return Artifact(spec.id, data)
    except FileNotFoundError:
        return Artifact(spec.id, None, "missing_artifact")
    except (OSError, PathEscapeError, ValueError):
        return Artifact(spec.id, None, "unsafe_or_oversized_artifact")


def captured_content(
    directory: Path, trace: TraceInspection, *, max_bytes: int
) -> tuple[bytes, ...]:
    result: list[bytes] = []
    for record in trace.records:
        if record.name != "content.capture":
            continue
        artifact_id = record.attributes.get("artifact_id")
        if (
            not isinstance(artifact_id, str)
            or len(artifact_id) != 32
            or any(c not in "0123456789abcdef" for c in artifact_id)
        ):
            continue
        try:
            with confined(directory / "artifacts") as root:
                result.append(
                    root.read_regular(artifact_id + ".json", max_bytes=max_bytes)
                )
        except (OSError, PathEscapeError, ValueError):
            continue
    return tuple(result)


def _result(
    spec: GraderSpec, passed: bool | None, reason: str | None = None
) -> GraderResult:
    return GraderResult(
        id=spec.id,
        kind=spec.kind,
        required=spec.required,
        version=GRADERS[spec.kind].version,
        status="not_evaluable" if passed is None else "valid",
        passed=passed,
        score=None if passed is None else float(passed),
        reason=reason,
    )


class ArtifactTextGrader:
    kind = "artifact_text"
    version = 1

    def __init__(self, spec: ArtifactText) -> None:
        self.spec = spec

    async def grade(self, evidence: TrialEvidence) -> GraderResult:
        artifact = next(
            (a for a in evidence.artifacts if a.grader_id == self.spec.id), None
        )
        if artifact is None:
            return _result(self.spec, None, "artifact_not_collected")
        return _result(
            self.spec, artifact.data == self.spec.equals.encode("utf-8"), artifact.issue
        )


def _subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _subset(v, actual[k]) for k, v in expected.items()
        )
    # JSON booleans and numbers must not compare interchangeably.
    return type(expected) is type(actual) and expected == actual


@dataclass(frozen=True)
class Invocation:
    tool: str
    start: int
    end: int
    arguments: Mapping[str, Any] | None
    path_args: tuple[str, ...]


def _matches(assertion: ToolAssertion, invocation: Invocation) -> bool:
    if assertion.tool != invocation.tool:
        return False
    if not assertion.arguments:
        return True
    if invocation.arguments is None:
        return False
    expected = dict(assertion.arguments)
    for key in invocation.path_args:
        if isinstance(expected.get(key), str):
            path = Path(expected[key])
            # Captured paths were resolved against the actual trial workspace.
            # Case predicates are workspace-relative and cannot name host state.
            if path.is_absolute() or ".." in path.parts:
                return False
            expected[key] = str(path)
    return _subset(expected, dict(invocation.arguments))


class ToolPolicyGrader:
    kind = "tool_policy"
    version = 1

    def __init__(self, spec: ToolPolicy) -> None:
        self.spec = spec

    async def grade(self, evidence: TrialEvidence) -> GraderResult:
        trace = evidence.trace
        if trace is None or not trace.complete:
            return _result(self.spec, None, "metadata_incomplete")
        attempted = {
            r.attributes.get("tool.name")
            for r in trace.records
            if r.name == "tool.dispatch" and r.kind == "span.start"
        }
        if attempted & set(self.spec.forbid):
            return _result(self.spec, False, "forbidden_tool_attempted")
        arguments: dict[str, dict[str, Any]] = {}
        for payload in evidence.captured:
            value = json.loads(payload)
            if not isinstance(value, list):
                continue
            for item in value:
                if (
                    isinstance(item, dict)
                    and item.get("kind") == "tool.arguments"
                    and not item.get("truncated")
                ):
                    detail = item.get("value", {})
                    if (
                        isinstance(detail, dict)
                        and isinstance(detail.get("arguments"), dict)
                        and isinstance(detail.get("span_id"), str)
                    ):
                        arguments[detail["span_id"]] = detail
        starts = {
            r.span_id: r
            for r in trace.records
            if r.kind == "span.start" and r.name == "tool.execute"
        }
        calls: list[Invocation] = []
        for record in trace.records:
            if (
                record.kind != "span.end"
                or record.name != "tool.execute"
                or record.status != "ok"
                or not record.attributes.get("ok")
            ):
                continue
            start = starts.get(record.span_id)
            if start is None:
                return _result(self.spec, None, "missing_tool_start")
            detail = arguments.get(record.span_id)
            calls.append(
                Invocation(
                    str(record.attributes["tool.name"]),
                    start.elapsed_ns,
                    record.elapsed_ns,
                    detail["arguments"] if detail else None,
                    tuple(detail.get("workspace_path_args", ())) if detail else (),
                )
            )
        for assertion in self.spec.require:
            candidates = [c for c in calls if c.tool == assertion.tool]
            if any(_matches(assertion, c) for c in candidates):
                continue
            if (
                assertion.arguments
                and candidates
                and any(c.arguments is None for c in candidates)
            ):
                return _result(self.spec, None, "argument_evidence_missing")
            return _result(self.spec, False, "required_tool_missing")
        for first, second in self.spec.causal_order:

            def matching(name: str) -> list[Invocation]:
                predicates = [a for a in self.spec.require if a.tool == name]
                return [
                    c
                    for c in calls
                    if c.tool == name
                    and (not predicates or any(_matches(a, c) for a in predicates))
                ]

            if not any(
                a.end <= b.start for a in matching(first) for b in matching(second)
            ):
                return _result(self.spec, False, "causal_order_failed")
        return _result(self.spec, True)


GRADERS: dict[str, type[ArtifactTextGrader] | type[ToolPolicyGrader]] = {
    "artifact_text": ArtifactTextGrader,
    "tool_policy": ToolPolicyGrader,
}


def evidence_needs(
    specs: list[GraderSpec], extra: list[str] | None = None
) -> dict[str, Any]:
    return {
        "artifacts": sorted(
            {Path(path).as_posix() for path in extra or ()}
            | {Path(g.path).as_posix() for g in specs if isinstance(g, ArtifactText)}
        ),
        "content": any(
            isinstance(g, ToolPolicy) and any(a.arguments for a in g.require)
            for g in specs
        ),
        "metadata": any(isinstance(g, ToolPolicy) for g in specs),
    }


async def grade(specs: list[GraderSpec], evidence: TrialEvidence) -> Grade:
    results: list[GraderResult] = []
    for spec in specs:
        grader = (
            ArtifactTextGrader(spec)
            if isinstance(spec, ArtifactText)
            else ToolPolicyGrader(spec)
        )
        try:
            results.append(await grader.grade(evidence))
        except Exception as exc:
            results.append(
                _result(spec, None, "grader_error").model_copy(
                    update={"error_type": type(exc).__name__}
                )
            )
    evaluable = all(r.status == "valid" for r in results if r.required)
    return Grade(
        status="valid" if evaluable else "invalid",
        passed=all(r.passed for r in results if r.required) if evaluable else None,
        scores={r.id: r.score for r in results},
        graders=results,
        reason="grader_error" if any(r.error_type for r in results) else None,
    )
