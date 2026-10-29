"""Sequential native trials with fresh state, bounded evidence and cleanup."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from lingcore.agent import Agent
from lingcore.config import TracingCfg
from lingcore.errors import ConfigError
from lingcore.evaluation.config import (
    EvaluationConfigError,
    ResolvedSuite,
    load_yaml,
    resolve_suite,
)
from lingcore.evaluation.graders import (
    TrialEvidence,
    captured_content,
    collect_artifact,
    grade,
)
from lingcore.evaluation.models import (
    ArtifactText,
    EvaluationDiagnostic,
    EvaluationPhase,
    Grade,
    Suite,
    TrialResult,
    TrialSpec,
    UsageSummary,
)
from lingcore.evaluation.report import build_report, exit_code
from lingcore.execution import ExecutionBudget, ExecutionOutcome, run_headless
from lingcore.paths import confined_directory
from lingcore.tracing import TraceHealth
from lingcore.tracing.config import known_secrets
from lingcore.tracing.content import sanitized_content
from lingcore.tracing.context import new_context
from lingcore.tracing.inspect import inspect_trace
from lingcore.tracing.requests import StreamingClient
from lingcore.usage import RequestUsage

_LOG = logging.getLogger(__name__)


def _usage_summary(records: tuple[RequestUsage, ...]) -> UsageSummary:
    return UsageSummary(
        input_tokens=sum(u.input_tokens or 0 for u in records),
        output_tokens=sum(u.output_tokens or 0 for u in records),
        cached_input_tokens=sum(u.cached_input_tokens or 0 for u in records),
        reasoning_tokens=sum(u.reasoning_tokens or 0 for u in records),
        requests=len(records),
        coverage="complete"
        if all(u.availability == "reported" for u in records)
        else "incomplete",
    )


def _write(directory: Path, name: str, value: Any) -> None:
    data = (
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    with confined_directory(
        Path(directory.anchor),
        directory.relative_to(directory.anchor),
        create=True,
        permissions=0o700,
    ) as root:
        with root.open_exclusive(name, 0o600) as handle:
            handle.write(data)


def _copy_trace(source: Path, destination: Path, max_bytes: int) -> None:
    """Copy only our confined journal and bounded artifacts, before state cleanup."""
    with confined_directory(
        Path(destination.anchor),
        destination.relative_to(destination.anchor),
        create=True,
        permissions=0o700,
    ) as target:
        with confined_directory(
            Path(source.anchor), source.relative_to(source.anchor)
        ) as origin:
            for name in ("manifest.json", "events.jsonl", "summary.json"):
                if origin.entry_exists(name):
                    data = origin.read_regular(name, max_bytes=max_bytes)
                    with target.open_exclusive(name, 0o600) as output:
                        output.write(data)
            if origin.entry_exists("artifacts"):
                with origin.subdirectory("artifacts") as artifacts:
                    with target.subdirectory(
                        "artifacts", create=True, permissions=0o700
                    ) as outputs:
                        for name, is_dir, is_file, is_link in artifacts.iter_entries():
                            if (
                                is_file
                                and not is_dir
                                and not is_link
                                and name.endswith(".json")
                            ):
                                with outputs.open_exclusive(name, 0o600) as output:
                                    output.write(
                                        artifacts.read_regular(
                                            name, max_bytes=max_bytes
                                        )
                                    )


async def _trial(
    resolved: ResolvedSuite,
    spec: TrialSpec,
    destination: Path,
    client_factory: Callable[[TrialSpec], StreamingClient] | None,
    on_result: Callable[[TrialResult], None] | None = None,
) -> TrialResult:
    case = next(c for c in resolved.cases if c.id == spec.case_id)
    suite = resolved.suite
    execution_id = str(uuid4())
    result_dir = destination / execution_id
    await asyncio.to_thread(
        _write, result_dir, "trial.json", spec.model_dump(mode="json")
    )
    temporary = tempfile.TemporaryDirectory(prefix="lingcore-eval-")
    private = Path(temporary.name)
    agent: Agent | None = None
    usage = UsageSummary()
    health = TraceHealth(complete=False)
    outcome = ExecutionOutcome(status="failed", cause="runner")
    grading = Grade(status="not_run")
    elapsed = 0
    cleanup_failed = False
    execution_finished = False
    cancelled = False
    trace_retained = False
    phase: EvaluationPhase = "setup"
    diagnostics: list[EvaluationDiagnostic] = []
    started = time.monotonic()
    try:
        profile = resolved.profile.model_copy(deep=True)
        profile.workspace = str(private / "workspace")
        profile.tracing = TracingCfg(
            enabled=True,
            content=suite.trace.content,
            max_bytes=suite.trace.max_bytes,
            artifact_max_bytes=suite.trace.artifact_max_bytes,
        )
        raw = client_factory(spec) if client_factory else None
        agent = Agent.from_profile(
            profile,
            llm=raw,
            state_dir=private / "state",
            execution_budget=ExecutionBudget(suite.execution.limits),
        )
        phase = "execute"
        headless = await run_headless(
            agent,
            [t.user for t in case.input.turns],
            allow_automatic_turns=suite.execution.limits.automatic_turns > 0,
            trace_context=new_context(
                trial_id=spec.trial_id, execution_id=execution_id
            ),
        )
        outcome, health, elapsed = headless.outcome, headless.trace, headless.elapsed_ms
        execution_finished = True
        cleanup_failed = headless.cleanup_failed
        usage = _usage_summary(headless.usage)
        phase = "collect"
        artifacts = tuple(
            [
                await asyncio.to_thread(collect_artifact, agent.tool_ctx.workspace, g)
                for g in case.graders
                if isinstance(g, ArtifactText)
            ]
        )
        for i, artifact in enumerate(artifacts):
            if artifact.data is not None:
                # A JSON string preserves exact bytes via a hexadecimal encoding;
                # no HTML or agent-generated executable report is opened.
                await asyncio.to_thread(
                    _write,
                    result_dir / "artifacts",
                    f"{i}.json",
                    {"grader_id": artifact.grader_id, "hex": artifact.data.hex()},
                )
        trace = None
        content: tuple[bytes, ...] = ()
        source = getattr(agent.recorder.sink, "directory", None)
        if source is not None:
            try:
                await asyncio.to_thread(
                    _copy_trace, source, result_dir / "trace", suite.trace.max_bytes
                )
                trace_retained = True
                trace = await asyncio.to_thread(
                    inspect_trace, result_dir / "trace", max_bytes=suite.trace.max_bytes
                )
                content = await asyncio.to_thread(
                    captured_content,
                    result_dir / "trace",
                    trace,
                    max_bytes=suite.trace.artifact_max_bytes,
                )
            except (ValueError, OSError):
                health = health.model_copy(update={"complete": False})
        # Capture failures invalidate trajectory grading, never an independent
        # file assertion. The report preserves both measurements.
        if trace is not None and not health.complete:
            from dataclasses import replace

            trace = replace(trace, complete=False)
        phase = "grade"
        grading = await grade(case.graders, TrialEvidence(artifacts, trace, content))
        for grader_result in grading.graders:
            if grader_result.error_type is not None:
                diagnostics.append(
                    EvaluationDiagnostic(
                        phase="grade",
                        code="grader_error",
                        message=f"grader {grader_result.id} failed ({grader_result.error_type})",
                        error_type=grader_result.error_type,
                    )
                )
        if cleanup_failed:
            grading = grading.model_copy(update={"status": "invalid", "passed": None})
    except asyncio.CancelledError:
        cancelled = True
        if not execution_finished:
            outcome = ExecutionOutcome(status="cancelled", cause="runner")
        grading = Grade(
            status="invalid" if execution_finished else "not_run",
            reason=f"{phase}_cancelled",
        )
        diagnostics.append(
            EvaluationDiagnostic(
                phase=phase, code="cancelled", message=f"{phase} cancelled"
            )
        )
        raise
    except Exception as exc:
        if not execution_finished:
            outcome = ExecutionOutcome(
                status="failed",
                cause="configuration" if isinstance(exc, ConfigError) else "runner",
            )
        grading = Grade(
            status="invalid" if execution_finished else "not_run",
            reason=f"{phase}_failed",
        )
        diagnostics.append(
            EvaluationDiagnostic(
                phase=phase,
                code=f"{phase}_failed",
                message=f"{phase} failed ({type(exc).__name__})",
                error_type=type(exc).__name__,
            )
        )
    finally:
        if agent is not None and not agent._closed:
            try:
                async with asyncio.timeout(5):
                    await agent.aclose()
            except Exception:
                cleanup_failed = True
        if agent is not None:
            if not execution_finished:
                usage = _usage_summary(tuple(agent.request_ledger.records))
                health = agent.recorder.health
                elapsed = round((time.monotonic() - started) * 1000)
            source = getattr(agent.recorder.sink, "directory", None)
            if source is not None and not trace_retained:
                try:
                    await asyncio.to_thread(
                        _copy_trace, source, result_dir / "trace", suite.trace.max_bytes
                    )
                except Exception as exc:
                    health = health.model_copy(update={"complete": False})
                    diagnostics.append(
                        EvaluationDiagnostic(
                            phase="collect",
                            code="trace_collection_failed",
                            message=f"trace collection failed ({type(exc).__name__})",
                            error_type=type(exc).__name__,
                        )
                    )
        try:
            async with asyncio.timeout(5):
                await asyncio.to_thread(temporary.cleanup)
        except Exception:
            cleanup_failed = True
        result = TrialResult(
            trial_id=spec.trial_id,
            execution_id=execution_id,
            case_id=case.id,
            case_revision=case.revision,
            repetition=spec.repetition,
            execution=outcome,
            grade=grading,
            usage=usage,
            trace=health,
            content=suite.trace.content,
            elapsed_ms=elapsed,
            cleanup_failed=cleanup_failed,
            diagnostics=diagnostics,
        )
        try:
            await asyncio.to_thread(
                _write, result_dir, "result.json", result.model_dump(mode="json")
            )
        except Exception as exc:
            if not cancelled:
                raise
            _LOG.warning(
                "Cancelled trial result persistence failed (%s)", type(exc).__name__
            )
        finally:
            if on_result is not None:
                on_result(result)
    return result


async def run_suite(
    path: Path,
    output: Path,
    *,
    client_factory: Callable[[TrialSpec], StreamingClient] | None = None,
) -> tuple[Path, dict[str, Any]]:
    job = output.expanduser().absolute() / ("job-" + uuid4().hex)
    resolved: ResolvedSuite | None = None
    results: list[TrialResult] = []
    planned = 0
    configuration_error = False
    interrupted = False
    diagnostic: EvaluationDiagnostic | None = None
    try:
        await asyncio.to_thread(
            _write,
            job,
            "job.json",
            {
                "schema_version": "lingcore.eval-job.v1",
                "locked": False,
                "benchmark_compatible": False,
                "mode": "unlocked_native",
            },
        )
        declared = await asyncio.to_thread(load_yaml, path.absolute(), Suite)
        planned = len(declared.source.cases) * declared.execution.repetitions
        resolved = await asyncio.to_thread(
            resolve_suite, path.absolute(), require_credentials=client_factory is None
        )
        planned = len(resolved.cases) * resolved.suite.execution.repetitions
        # Fingerprint sanitized configuration, never environment/credential values.
        values = resolved.profile.model_dump(exclude={"workspace", "tracing"})
        known = known_secrets(resolved.profile)
        data, _ = sanitized_content(values, known, 8 * 1024 * 1024)
        await asyncio.to_thread(
            _write,
            job,
            "plan.json",
            {
                "locked": False,
                "benchmark_compatible": False,
                "planned": planned,
                "candidate_fingerprint": hashlib.sha256(data).hexdigest(),
                "limits": resolved.suite.execution.limits.model_dump(),
                "capture": resolved.suite.trace.model_dump(),
                "cases": [{"id": c.id, "revision": c.revision} for c in resolved.cases],
            },
        )
        for case in resolved.cases:
            for repetition in range(1, resolved.suite.execution.repetitions + 1):
                spec = TrialSpec(
                    trial_id=f"{case.id}--candidate--{repetition}",
                    case_id=case.id,
                    repetition=repetition,
                )
                result = await _trial(
                    resolved,
                    spec,
                    job / "executions",
                    client_factory,
                    on_result=results.append,
                )
                if result.execution.status == "cancelled":
                    break
            if results and results[-1].execution.status == "cancelled":
                break
    except asyncio.CancelledError:
        interrupted = True
        raise
    except EvaluationConfigError as exc:
        configuration_error = True
        diagnostic = exc.diagnostic
    except Exception as exc:
        configuration_error = True
        diagnostic = EvaluationDiagnostic(
            phase="preflight" if resolved is None else "runner",
            code="evaluation_failed",
            message=f"evaluation failed ({type(exc).__name__})",
            error_type=type(exc).__name__,
        )
    finally:
        report = build_report(results, planned, configuration_error=configuration_error)
        if interrupted:
            report["interrupted"] = True
            report["complete"] = False
        if configuration_error:
            assert diagnostic is not None
            report["configuration_error"] = diagnostic.message
            report["diagnostics"] = [diagnostic.model_dump(mode="json")]
        try:
            await asyncio.to_thread(_write, job, "report.json", report)
        except Exception as exc:
            if not interrupted:
                raise
            _LOG.warning(
                "Cancelled evaluation report persistence failed (%s)",
                type(exc).__name__,
            )
    return job, report


def evaluation_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lingcore eval")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Execute an unlocked native development suite"
    )
    run.add_argument("suite", type=Path)
    from lingcore.profiles import state_home

    run.add_argument("--output", type=Path, default=state_home() / "evaluations")
    show = commands.add_parser("report", help="Read a saved per-case report")
    show.add_argument("job", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            with confined_directory(args.job) as root:
                report = json.loads(
                    root.read_regular("report.json", max_bytes=16 * 1024 * 1024)
                )
        else:
            job, report = asyncio.run(run_suite(args.suite, args.output))
            print(f"Evaluation artifacts: {job}")
        for case in report["cases"]:
            print(
                f"{case['case_id']}: {case['passed']}/{case['valid']} valid trials passed; {case['started']} started"
            )
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
        return exit_code(report)
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except (ValueError, OSError, ConfigError) as exc:
        print(f"evaluation failed ({type(exc).__name__})", file=sys.stderr)
        return 2
