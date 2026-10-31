"""One Harbor execution under the shared append-only job ledger."""

from __future__ import annotations

import asyncio
import fcntl
import os
import platform
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from lingcore.agent import Agent
from lingcore.evaluation import jobs
from lingcore.evaluation.config import EvaluationConfigError, ResolvedSuite
from lingcore.evaluation.harbor.binding import HarborTerminal
from lingcore.evaluation.harbor.listener import Listener
from lingcore.evaluation.harbor.outcomes import map_outcome
from lingcore.evaluation.harbor.peer import Peer
from lingcore.evaluation.harbor.process import bridge_environment, invoke
from lingcore.evaluation.harbor.protocol import (
    Abort,
    Finish,
    ProtocolFailure,
    Usage,
    UsageTotals,
)
from lingcore.evaluation.harbor.tools import terminal_registry
from lingcore.evaluation.identity import (
    HarborExecutionIdentity,
    HarborGradingIdentity,
    HarborSourceIdentity,
    PlanIdentity,
    digest,
    identity_body,
    value_digest,
)
from lingcore.evaluation.lock import utc_now
from lingcore.evaluation.models import Comparison, Grade, TrialResult, TrialSpec
from lingcore.evaluation.storage import (
    directory,
    finish_io,
    read_bytes,
    write_bytes,
    write_json,
)
from lingcore.execution import (
    ExecutionBudget,
    ExecutionOutcome,
    RunnerSignal,
    run_headless,
)
from lingcore.paths import ReadLimitError
from lingcore.tracing import TraceHealth, TraceRecorder
from lingcore.tracing.config import known_secrets
from lingcore.tracing.content import redact_known_secrets
from lingcore.tracing.context import new_context
from lingcore.tracing.local import LocalTraceSink
from lingcore.usage import RequestUsage

if TYPE_CHECKING:
    from lingcore.evaluation.runner import ClientFactory


def usage_totals(
    records: list[RequestUsage], secrets: tuple[str, ...] = ()
) -> UsageTotals:
    from lingcore.evaluation.runner import _usage_summary

    per_model: dict[str, dict[str, int]] = {}
    for row in records:
        counts = per_model.setdefault(redact_known_secrets(row.model, secrets), {})
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            counts[name] = counts.get(name, 0) + (getattr(row, name) or 0)
    return UsageTotals(
        **_usage_summary(tuple(records)).model_dump(exclude={"estimated_cost_usd"}),
        per_model=per_model,
    )


def check_harness_stopped(path: Path) -> None:
    with directory(path) as root:
        if not root.entry_exists("harbor.lock"):
            return
        with root.open_regular("harbor.lock") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise EvaluationConfigError(
                    f"execution {path.name}: Harbor harness is still running",
                    "harness_running",
                ) from None


async def stop_process(process: asyncio.subprocess.Process) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return True
    try:
        async with asyncio.timeout(120):
            await process.wait()
            # The leader can exit while Harbor's Docker children still hold
            # pipes or run cleanup. Wait for the group, not only its leader.
            while True:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.05)
        return True
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        return False


async def capture_trajectory(
    recorder: TraceRecorder, root: Path, plan: PlanIdentity, execution_id: str
) -> tuple[
    dict[str, Any] | None, Literal["metadata_only", "disabled", "incomplete"] | None
]:
    if "atif" not in plan.trace.exports:
        return None, "metadata_only" if plan.trace.content == "metadata" else "disabled"
    from lingcore.evaluation.harbor.trajectory import export_trajectory

    if recorder.sink is None:
        return None, "incomplete"
    try:
        async with asyncio.timeout(30):
            health = await recorder.sink.flush()
            if health.flush_timed_out:
                return None, "incomplete"
            trajectory = await asyncio.to_thread(
                export_trajectory, root / "trace", plan, execution_id
            )
        return trajectory, None if trajectory is not None else "incomplete"
    except Exception:
        # An optional export must never change the execution/retry outcome.
        return None, "incomplete"


def collect_evidence(root: Path, limit: int) -> None:
    files: list[dict[str, Any]] = []
    harbor = root / "harbor"
    if harbor.exists():
        for path in sorted(harbor.rglob("*")):
            if path.is_symlink():
                files.append(
                    {"path": path.relative_to(root).as_posix(), "status": "unsafe"}
                )
                continue
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                data = read_bytes(path, max_bytes=limit)
                files.append(
                    {
                        "path": relative,
                        "digest": digest(data),
                        "bytes": len(data),
                        "status": "captured",
                    }
                )
            except ReadLimitError:
                files.append({"path": relative, "status": "oversized"})
    write_json(
        root / "evidence.json",
        {"schema_version": "lingcore.harbor-evidence.v1", "files": files},
    )


async def run_trial(
    resolved: ResolvedSuite,
    spec: TrialSpec,
    destination: Path,
    client_factory: ClientFactory | None,
    *,
    plan: PlanIdentity,
    attempt: int,
    locked: bool,
    lock_digest: str | None,
) -> TrialResult:
    from lingcore.evaluation.runner import _usage_summary

    assert (
        isinstance(plan.source, HarborSourceIdentity)
        and isinstance(plan.execution, HarborExecutionIdentity)
        and isinstance(plan.grading, HarborGradingIdentity)
    )
    assert resolved.bridge is not None and resolved.dataset is not None
    task = next(t for t in plan.source.tasks if t.id == spec.case_id)
    execution_id = str(uuid4())
    root = destination / execution_id
    scratch = Path(tempfile.gettempdir()) / ("lingcore-eval-" + execution_id)
    plan_digest = value_digest(identity_body(plan))
    await finish_io(
        write_json,
        root / "trial.json",
        jobs.TrialRecord(
            **spec.model_dump(),
            execution_id=execution_id,
            attempt=attempt,
            started_at=utc_now(),
            plan_digest=plan_digest,
            scratch_dir=str(scratch),
        ).model_dump(mode="json"),
    )
    profile = resolved.profile.model_copy(deep=True)
    profile.workspace = str(scratch / "workspace")
    environment = bridge_environment(profile, plan.execution.pass_env)
    secrets = known_secrets(profile)
    listener = Listener(execution_id)
    peer: Peer | None = None
    process: asyncio.subprocess.Process | None = None
    agent: Agent | None = None
    recorder: TraceRecorder | None = None
    observers: list[asyncio.Task[Any]] = []
    drains: list[asyncio.Task[Any]] = []
    hello_seen = unsupported = cleanup_failed = protocol_failed = cancelled = False
    finish_failed = digest_mismatch = False
    cleanup_known = True
    outcome = ExecutionOutcome(status="failed", cause="runner")
    summary: dict[str, Any] | None = None
    grading = Grade(status="not_run")
    started = time.monotonic()
    agent_elapsed: int | None = None
    try:
        scratch.mkdir(mode=0o700)
        await listener.start()
        recorder = TraceRecorder(
            LocalTraceSink(
                root / "trace",
                manifest={
                    "schema_version": "lingcore.trace.v1",
                    "run_id": execution_id,
                    "trial_id": spec.trial_id,
                    "plan_digest": plan_digest,
                    "capture": plan.trace.model_dump(mode="json"),
                },
                max_bytes=plan.trace.max_bytes,
            ),
            run_id=execution_id,
            content=plan.trace.content,
            artifact_max_bytes=plan.trace.artifact_max_bytes,
            secrets=known_secrets(profile),
        )
        args = [
            str(resolved.bridge),
            "trial",
            "--task",
            str(resolved.dataset / task.path),
            "--task-name",
            task.name,
            "--socket",
            str(listener.path),
            "--execution-id",
            execution_id,
            "--output",
            str(root / "harbor"),
        ]
        multipliers = plan.execution.harness["timeout_multipliers"]
        for name, value in multipliers.items():
            args += ["--" + name, str(value)]
        process = await asyncio.create_subprocess_exec(
            *args,
            env=environment,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def drain(stream: asyncio.StreamReader | None, name: str) -> None:
            assert stream is not None
            data = bytearray()
            while chunk := await stream.read(65536):
                data.extend(chunk[: max(0, plan.trace.artifact_max_bytes - len(data))])
            await finish_io(write_bytes, root / name, bytes(data))

        drains = [
            asyncio.create_task(drain(process.stdout, "bridge.stdout")),
            asyncio.create_task(drain(process.stderr, "bridge.stderr")),
        ]
        accept = asyncio.create_task(
            listener.accept(
                3600 * multipliers["build"] + 360 * multipliers["setup"] + 60
            )
        )
        exited = asyncio.create_task(process.wait())
        try:
            done, _ = await asyncio.wait(
                {accept, exited}, return_when=asyncio.FIRST_COMPLETED
            )
            if accept not in done:
                raise ProtocolFailure("bridge exited before hello")
            channel, hello = accept.result()
        finally:
            accept.cancel()
            exited.cancel()
            await asyncio.gather(accept, exited, return_exceptions=True)
        if (
            hello.bridge.model_dump() != plan.execution.harness["bridge"]
            or hello.harbor_version != plan.execution.harness["harbor"]["version"]
            or hello.task != task.name
            or digest(hello.instruction.encode()) != task.instruction_digest
        ):
            raise ProtocolFailure("bridge hello differs from the lock")
        hello_seen = True
        unsupported = not hello.probe.supported
        peer = Peer(channel)
        if unsupported:
            await peer.send(Abort(reason="protocol_error"))
        else:
            terminal = HarborTerminal(peer, profile.terminal_options, secrets=secrets)
            agent = Agent.from_profile(
                profile,
                llm=client_factory(spec) if client_factory else None,
                tool_registry=terminal_registry(),
                environment_binding=terminal,
                state_dir=scratch / "state",
                recorder=recorder,
                execution_budget=ExecutionBudget(resolved.suite.execution.limits),
            )
            runner_signal: asyncio.Future[RunnerSignal] = (
                asyncio.get_running_loop().create_future()
            )
            changed = asyncio.Event()
            agent.request_ledger.on_record = changed.set

            async def monitor() -> None:
                assert peer is not None
                signal_value = await asyncio.shield(peer.signal)
                failure = ProtocolFailure("bridge stopped during agent execution")
                if (
                    isinstance(signal_value, Abort)
                    and signal_value.reason == "harness_timeout"
                ):
                    failure.outcome = ExecutionOutcome(
                        status="timed_out", cause="agent", limit_type="harness_agent"
                    )
                if not runner_signal.done():
                    runner_signal.set_result(failure)

            async def publish_usage() -> None:
                assert peer is not None and agent is not None
                while True:
                    await changed.wait()
                    changed.clear()
                    try:
                        await peer.send(
                            Usage(
                                usage=usage_totals(
                                    agent.request_ledger.records, secrets
                                )
                            )
                        )
                    except ProtocolFailure:
                        return

            observers = [
                asyncio.create_task(monitor()),
                asyncio.create_task(publish_usage()),
            ]
            result = await run_headless(
                agent,
                [hello.instruction],
                trace_context=new_context(
                    trial_id=spec.trial_id, execution_id=execution_id
                ),
                runner_signal=runner_signal,
                benchmark_environment=True,
            )
            outcome, cleanup_failed = result.outcome, result.cleanup_failed
            agent_elapsed = result.elapsed_ms
            for observer in observers:
                observer.cancel()
            await asyncio.gather(*observers, return_exceptions=True)
            protocol_failed = outcome.cause == "runner" and outcome.status == "failed"
            if peer.signal.done():
                ended = peer.signal.result()
                if isinstance(ended, Abort) and ended.reason == "harness_timeout":
                    outcome = ExecutionOutcome(
                        status="timed_out", cause="agent", limit_type="harness_agent"
                    )
                    protocol_failed = False
                else:
                    protocol_failed = True
            if not peer.signal.done():
                trajectory, atif_reason = await capture_trajectory(
                    recorder, root, plan, execution_id
                )
                try:
                    async with asyncio.timeout(60):
                        finished = await peer.finish(
                            Finish(
                                outcome=outcome.model_dump(mode="json"),
                                usage=usage_totals(
                                    agent.request_ledger.records, secrets
                                ),
                                metadata={
                                    "execution_id": execution_id,
                                    "trial_id": spec.trial_id,
                                    "plan_digest": plan_digest,
                                },
                                trajectory=trajectory,
                                atif_reason=atif_reason,
                            )
                        )
                    cleanup_failed |= finished.sweep == "failed"
                except (ProtocolFailure, TimeoutError):
                    finish_failed = True
                    # Harbor may be cancelling the final sweep. Let it finish
                    # verification, then classify using its timeout evidence.
        async with asyncio.timeout(
            task.verifier["timeout_sec"] * multipliers["verifier"]
            + 3600 * multipliers["build"]
            + 180
        ):
            await process.wait()
    except asyncio.CancelledError:
        cancelled = True
        outcome = ExecutionOutcome(status="cancelled", cause="runner")
        if peer is not None:
            try:
                await peer.send(Abort(reason="cancelled"))
            except ProtocolFailure:
                pass
        raise
    except (ProtocolFailure, Exception):
        protocol_failed = True
    finally:
        for observer in observers:
            observer.cancel()
        await asyncio.gather(*observers, return_exceptions=True)
        if agent is not None and not agent._closed:
            try:
                await asyncio.wait_for(agent.aclose(), 5)
            except Exception:
                cleanup_failed = True
        if peer is not None:
            await peer.aclose()
        await listener.aclose()
        if process is not None:
            cleanup_known = await stop_process(process)
        await asyncio.gather(*drains, return_exceptions=True)
        if recorder is not None:
            await recorder.aclose()
        try:
            summary = await finish_io(
                invoke,
                resolved.bridge,
                ["summarize", "--output", str(root / "harbor")],
                environment,
            )
            await finish_io(write_json, root / "harbor-summary.json", summary)
            if summary["task_digest"] != task.harbor_digest:
                digest_mismatch = True
        except Exception:
            summary = None
        cleanup_failed |= not cleanup_known or bool(
            summary and summary.get("metadata", {}).get("sweep") == "failed"
        )
        harness_timeout = bool(
            peer is not None
            and peer.signal.done()
            and isinstance(ended := peer.signal.result(), Abort)
            and ended.reason == "harness_timeout"
        ) or bool(
            summary
            and any(
                e["type"] == "AgentTimeoutError" for e in summary.get("exceptions", [])
            )
        )
        if finish_failed and not harness_timeout:
            protocol_failed = True
        if harness_timeout and not protocol_failed:
            outcome = ExecutionOutcome(
                status="timed_out", cause="agent", limit_type="harness_agent"
            )
        if not cancelled:
            outcome, grading = map_outcome(
                summary,
                outcome,
                plan.grading.reward,
                hello=hello_seen,
                unsupported=unsupported,
                cleanup_failed=cleanup_failed,
                protocol_failed=protocol_failed,
            )
        else:
            grading = Grade(status="not_run", reason="cancelled")
        if digest_mismatch and not cancelled:
            outcome = ExecutionOutcome(status="failed", cause="configuration")
            grading = Grade(status="invalid", reason="task_digest_mismatch")
        try:
            await finish_io(collect_evidence, root, plan.trace.artifact_max_bytes)
        except Exception:
            grading = Grade(status="invalid", reason="evidence_failed")
        if scratch.exists():
            await finish_io(shutil.rmtree, scratch)
        records = tuple(agent.request_ledger.records) if agent else ()
        values = summary or {}
        trial_result = TrialResult(
            **spec.model_dump(),
            execution_id=execution_id,
            case_revision=task.revision,
            locked=locked,
            attempt=attempt,
            lock_digest=lock_digest,
            plan_digest=plan_digest,
            execution=outcome,
            grade=grading,
            usage=_usage_summary(records),
            trace=recorder.health if recorder else TraceHealth(complete=False),
            content=plan.trace.content,
            elapsed_ms=agent_elapsed
            if agent_elapsed is not None
            else round((time.monotonic() - started) * 1000),
            cleanup_failed=cleanup_failed,
            comparison=Comparison(locked=locked, track=plan.suite.track),
            served_models=sorted(
                {
                    redact_known_secrets(u.model, secrets)
                    for u in records
                    if u.availability != "missing"
                }
            ),
            rewards=values.get("rewards"),
            reward_valid=values.get("reward_valid"),
            harbor_failure_cause=values.get("failure_cause"),
            harbor_exceptions=values.get("exceptions", []),
            timing=values.get("timing", {}),
            harbor_task_digest=values.get("task_digest"),
            harbor_trial="harbor/" + values["trial"] if "trial" in values else None,
            atif=values.get("metadata", {}).get("atif", "absent"),
            atif_reason=values.get("metadata", {}).get("atif_reason"),
            environment_cleanup="complete" if cleanup_known else "unknown",
            execution_platform=platform.machine(),
            unsupported=unsupported,
        )
        cause = jobs.infrastructure_cause(trial_result)
        if cause in plan.execution.retry.retry_on:
            trial_result = trial_result.model_copy(
                update={"infrastructure_failure": cause}
            )
        await finish_io(
            write_json, root / "result.json", trial_result.model_dump(mode="json")
        )
    return trial_result
