"""One Harbor invocation and normalized, bounded summaries of its own files."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from harbor.models.trial.result import TrialResult

from lingcore_harbor_bridge.describe import (
    MAX_JSON_BYTES,
    DescriptionError,
    read_metadata,
)


def trial(args: Any) -> dict[str, Any]:
    from harbor.cli.main import app

    os.environ["HARBOR_TELEMETRY"] = "off"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        args.output.parent / "harbor.lock",
        os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [
            "run",
            "-p",
            str(args.task),
            "-a",
            "lingcore_harbor_bridge.agent:BridgeAgent",
            "-k",
            "1",
            "-n",
            "1",
            "--max-retries",
            "0",
            "--jobs-dir",
            str(args.output),
            "--job-name",
            "trial",
            "--quiet",
        ]
        for key, value in {
            "socket_path": str(args.socket),
            "execution_id": args.execution_id,
            "task_name": args.task_name,
        }.items():
            command += ["--agent-kwarg", key + "=" + value]
        for name, flag in {
            "agent": "agent",
            "verifier": "verifier",
            "setup": "agent-setup",
            "build": "environment-build",
        }.items():
            command += ["--" + flag + "-timeout-multiplier", str(getattr(args, name))]
        with contextlib.redirect_stdout(sys.stderr):
            app(args=command, standalone_mode=False)
        return summarize(args.output)
    finally:
        os.close(fd)


def summarize(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise DescriptionError("unsafe_harbor_output")
    candidates = []
    for job in sorted(root.iterdir()):
        if job.is_symlink():
            raise DescriptionError("unsafe_harbor_output")
        if not job.is_dir():
            continue
        for trial in sorted(job.iterdir()):
            if trial.is_symlink():
                raise DescriptionError("unsafe_harbor_output")
            if trial.is_dir() and (trial / "result.json").exists():
                candidates.append(trial / "result.json")
    if len(candidates) != 1:
        raise DescriptionError("missing_or_ambiguous_trial_result")
    path = candidates[0]
    result = TrialResult.model_validate_json(read_metadata(path, MAX_JSON_BYTES))
    lock = json.loads(read_metadata(path.parent / "lock.json", MAX_JSON_BYTES))
    exceptions = [
        dict(
            type=exc.exception_type,
            policy={**asdict(exc.policy), "reward_valid": exc.policy.reward_valid},
        )
        for exc in result._recorded_exceptions()
    ]
    timing = {}
    for name in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
        value = getattr(result, name)
        timing[name] = (
            None
            if value is None or value.started_at is None or value.finished_at is None
            else (value.finished_at - value.started_at).total_seconds()
        )
    return dict(
        schema_version="lingcore.harbor-summary.v1",
        trial=path.parent.relative_to(root).as_posix(),
        rewards=result.verifier_result.rewards if result.verifier_result else {},
        reward_valid=result.reward_valid,
        failure_cause=result.failure_cause,
        exceptions=exceptions,
        timing=timing,
        task_digest=lock["task"]["digest"],
        metadata=(result.agent_result.metadata or {}).get("lingcore", {})
        if result.agent_result
        else {},
    )


def regrade(args: Any) -> dict[str, Any]:
    from harbor.cli.main import app

    os.environ["HARBOR_TELEMETRY"] = "off"
    # Each source job was created with one trial and Harbor retries disabled.
    with contextlib.redirect_stdout(sys.stderr):
        app(
            args=[
                "job",
                "regrade",
                str(args.source.parent),
                "-p",
                str(args.task),
                "--jobs-dir",
                str(args.output),
                "--job-name",
                "regrade",
                "-n",
                "1",
            ],
            standalone_mode=False,
        )
    return summarize(args.output)
