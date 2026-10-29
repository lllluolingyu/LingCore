"""CLI boundary: library cancellation propagates; only here it becomes 130."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from lingcore.errors import ConfigError
from lingcore.evaluation import runner
from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.lock import load_lock, resolve, validate, write_lock
from lingcore.evaluation.regrade import read_report, regrade
from lingcore.evaluation.report import exit_code
from lingcore.paths import PathEscapeError
from lingcore.profiles import state_home


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(coroutine)
    finally:
        coroutine.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lingcore eval")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "resolve", help="Freeze a local evaluation plan without a model call"
    )
    freeze.add_argument("suite", type=Path)
    freeze.add_argument("--lock", type=Path, required=True)
    freeze.add_argument("--replace", action="store_true")
    check = commands.add_parser(
        "validate", help="Check a lock offline and report value-free drift"
    )
    check.add_argument("--lock", type=Path, required=True)
    run = commands.add_parser(
        "run", help="Run a locked plan or an unlocked development suite"
    )
    run.add_argument("suite", type=Path, nargs="?")
    run.add_argument("--lock", type=Path)
    run.add_argument("--output", type=Path, default=state_home() / "evaluations")
    continuation = commands.add_parser(
        "resume", help="Continue unfinished trials from a locked job"
    )
    continuation.add_argument("job", type=Path)
    continuation.add_argument("--lock", type=Path)
    rescore = commands.add_parser(
        "regrade", help="Grade preserved evidence without running an agent"
    )
    rescore.add_argument("job", type=Path)
    rescore.add_argument("--lock", type=Path)
    report = commands.add_parser("report", help="Read an original or regraded report")
    report.add_argument("job", type=Path)
    report.add_argument("--grades", default="original")
    args = parser.parse_args(argv)
    if args.command == "run" and (args.suite is None) == (args.lock is None):
        parser.error("run accepts exactly one of <suite> or --lock <path>")
    printed: set[Path] = set()

    def show_job(job: Path) -> None:
        if job not in printed:
            printed.add(job)
            print(f"Evaluation artifacts: {job}", flush=True)

    try:
        if args.command == "resolve":
            lock = _run(resolve(args.suite, args.lock))
            write_lock(lock, args.lock, replace=args.replace)
            print(f"Lock written: {args.lock}")
            print(f"Planned trials: {len(lock.trials)}")
            for case in lock.source.cases:
                print(
                    f"{case.id}: {sum(t.case_id == case.id for t in lock.trials)} repetitions"
                )
            print(
                f"Model: {lock.candidate.model.id}; endpoint: {lock.candidate.model.endpoint}"
            )
            print("Limits: " + json.dumps(lock.execution.limits, sort_keys=True))
            print("Limitations: " + ", ".join(lock.limitations))
            return 0
        if args.command == "validate":
            drift = _run(validate(load_lock(args.lock), args.lock))
            for item in drift:
                print(item.describe())
            if not drift:
                print("ok")
            return 2 if drift else 0
        if args.command == "run":
            if args.lock is not None:
                job, result = _run(
                    runner.run_locked(args.lock, args.output, on_job=show_job)
                )
            else:
                job, result = _run(
                    runner.run_suite(args.suite, args.output, on_job=show_job)
                )
            show_job(job)
        elif args.command == "resume":
            show_job(args.job)
            result = _run(runner.resume(args.job, lock_path=args.lock))
        elif args.command == "regrade":
            show_job(args.job)
            result = _run(regrade(args.job, lock_path=args.lock))
        else:
            show_job(args.job)
            result = read_report(args.job, args.grades)
        for case in result["cases"]:
            print(
                f"{case['case_id']}: {case['passed']}/{case['valid']} valid trials passed; {case['started']} started"
            )
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
        return exit_code(result)
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except EvaluationConfigError as exc:
        print(exc.diagnostic.message, file=sys.stderr)
        return 2
    except (
        ConfigError,
        ValueError,
        TypeError,
        OSError,
        PathEscapeError,
        KeyError,
    ) as exc:
        print(f"evaluation failed ({type(exc).__name__})", file=sys.stderr)
        return 2
