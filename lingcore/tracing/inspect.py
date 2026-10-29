"""Offline inspection, including interrupted spans and truncated crash tails."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lingcore.paths import PathEscapeError, confined_directory
from lingcore.tracing.models import TraceHealth, TraceRecord


@dataclass(frozen=True)
class TraceInspection:
    records: tuple[TraceRecord, ...]
    warnings: tuple[str, ...]
    interrupted_spans: tuple[str, ...]
    complete: bool


def inspect_trace(
    directory: Path, *, max_bytes: int = 128 * 1024 * 1024
) -> TraceInspection:
    warnings: list[str] = []
    directory = directory.absolute()
    with confined_directory(
        Path(directory.anchor), directory.relative_to(directory.anchor)
    ) as root:
        data = root.read_regular("events.jsonl", max_bytes=max_bytes)
        health = (
            TraceHealth.model_validate_json(
                root.read_regular("summary.json", max_bytes=65536)
            )
            if root.entry_exists("summary.json")
            else None
        )
    if data and not data.endswith(b"\n"):
        warnings.append("Ignored crash-truncated final JSONL line")
        data = data.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in data else b""
    records: list[TraceRecord] = []
    opened: dict[str, TraceRecord] = {}
    previous = 0
    for line in data.splitlines():
        record = TraceRecord.model_validate_json(line)
        if record.seq != previous + 1:
            warnings.append("Journal sequence gap or out-of-order record")
        previous = record.seq
        if record.kind == "span.start":
            opened[record.span_id] = record
        elif record.kind == "span.end":
            if opened.pop(record.span_id, None) is None:
                warnings.append("Span end has no matching start")
        if record.name == "trace.cap":
            warnings.append("Run byte cap reached; capture stopped")
        records.append(record)
    if opened:
        warnings.append("Open spans are interrupted; outcome unknown")
    if health is None:
        warnings.append("Run has no shutdown health summary")
    elif not health.complete:
        warnings.append("Sink reports incomplete capture")
    return TraceInspection(tuple(records), tuple(warnings), tuple(opened), not warnings)


def trace_command(argv: list[str]) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(prog="lingcore trace")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser(
        "show", help="Inspect a local journal without executing it"
    )
    show.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_trace(args.directory)
    except (ValueError, OSError, PathEscapeError) as exc:
        print(f"cannot inspect trace ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "complete": result.complete,
                "records": len(result.records),
                "interrupted_spans": list(result.interrupted_spans),
                "warnings": list(result.warnings),
                "operations": [
                    {
                        "name": r.name,
                        "span_id": r.span_id,
                        "turn_id": r.turn_id,
                        "status": r.status,
                        "attributes": dict(r.attributes),
                    }
                    for r in result.records
                    if r.kind != "span.start"
                ],
            },
            indent=2,
        )
    )
    return 0 if result.complete else 2
