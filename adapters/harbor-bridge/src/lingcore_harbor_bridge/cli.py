"""Bounded JSON command surface; never import the LingCore runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lingcore_harbor_bridge.describe import DescriptionError, canonical, describe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lingcore-harbor-bridge")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("describe", help="Describe a local export offline")
    inspect.add_argument("--dataset", type=Path, required=True)
    execute = commands.add_parser("trial")
    execute.add_argument("--task", type=Path, required=True)
    execute.add_argument("--task-name", required=True)
    execute.add_argument("--socket", type=Path, required=True)
    execute.add_argument("--execution-id", required=True)
    execute.add_argument("--output", type=Path, required=True)
    for name in ("agent", "verifier", "setup", "build"):
        execute.add_argument("--" + name, type=float, default=1)
    summary = commands.add_parser("summarize")
    summary.add_argument("--output", type=Path, required=True)
    rescore = commands.add_parser("regrade")
    rescore.add_argument("--source", type=Path, required=True)
    rescore.add_argument("--task", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            payload = describe(args.dataset.absolute())
        else:
            from lingcore_harbor_bridge.commands import regrade, summarize, trial

            payload = (
                summarize(args.output)
                if args.command == "summarize"
                else trial(args)
                if args.command == "trial"
                else regrade(args)
            )
    except KeyboardInterrupt:
        return 130
    except DescriptionError as exc:
        payload = {"error": {"code": str(exc)}}
    except SystemExit as exc:
        payload = {
            "error": {
                "code": "harbor_exit",
                "exit_code": exc.code if isinstance(exc.code, int) else 1,
            }
        }
    except Exception as exc:
        payload = {
            "error": {"code": "description_failed", "error_type": type(exc).__name__}
        }
    data = canonical(payload) + b"\n"
    if len(data) > 4 * 1024 * 1024:
        data = b'{"error":{"code":"response_too_large"}}\n'
        payload = {"error": {}}
    sys.stdout.buffer.write(data)
    return 2 if "error" in payload else 0
