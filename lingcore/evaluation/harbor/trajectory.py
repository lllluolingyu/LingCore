"""ATIF v1.7 projection of committed, redacted journal evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lingcore.evaluation.identity import PlanIdentity
from lingcore.evaluation.storage import read_bytes
from lingcore.tracing.inspect import inspect_trace


def export_trajectory(
    directory: Path, plan: PlanIdentity, execution_id: str
) -> dict[str, Any] | None:
    if plan.trace.content != "redacted":
        return None
    try:
        inspection = inspect_trace(directory, max_bytes=plan.trace.max_bytes)
        steps: list[dict[str, Any]] = []
        calls: dict[str, dict[str, Any]] = {}
        requests: dict[str, dict[str, Any]] = {}
        events = []
        tool_definitions = None
        attributed = set()
        for record in inspection.records:
            attrs = dict(record.attributes)
            if record.name == "request.usage":
                requests[str(attrs["request_id"])] = attrs
            if (
                record.name
                in {
                    "memory.compact",
                    "turn.rollback",
                    "execution.outcome",
                    "retry.backoff",
                    "llm.request",
                }
                and record.kind != "span.start"
            ):
                events.append(
                    {"name": record.name, "status": record.status, "attributes": attrs}
                )
            if (
                record.name != "content.capture"
                or attrs.get("content.kind") != "turn.committed"
            ):
                continue
            items = json.loads(
                read_bytes(
                    directory / "artifacts" / (str(attrs["artifact_id"]) + ".json"),
                    max_bytes=plan.trace.artifact_max_bytes,
                )
            )
            if not isinstance(items, list):
                continue
            for item in items:
                value = item["value"]
                if item.get("truncated"):
                    events.append({"name": "content.truncated", "kind": item["kind"]})
                    continue
                if item["kind"] == "prompt":
                    tool_definitions = value["tools"]
                    steps.append(
                        {
                            "source": "system",
                            "message": value["system"],
                            "extra": {"lingcore": {"prompt": True}},
                        }
                    )
                elif item["kind"] == "message":
                    role = value["role"]
                    if role in {"user", "assistant"}:
                        step: dict[str, Any] = {
                            "source": "agent" if role == "assistant" else "user",
                            "message": value["content"],
                            "extra": {
                                "lingcore": {
                                    "redacted": True,
                                    "turn_id": record.turn_id,
                                }
                            },
                        }
                        if role == "assistant":
                            request_id = value.get("request_id")
                            usage = requests.get(request_id)
                            if usage is not None:
                                attributed.add(request_id)
                                step["metrics"] = {
                                    "prompt_tokens": usage.get("input_tokens"),
                                    "completion_tokens": usage.get("output_tokens"),
                                    "cached_tokens": usage.get("cached_input_tokens"),
                                    "extra": {
                                        "availability": usage.get("availability")
                                    },
                                }
                                step["model_name"] = usage["model"]
                            step["extra"]["lingcore"].update(
                                request_id=request_id,
                                finish_reason=value.get("finish_reason"),
                            )
                            if value.get("tool_calls"):
                                step["tool_calls"] = [
                                    {
                                        "tool_call_id": c["id"],
                                        "function_name": c["name"],
                                        "arguments": c["arguments"],
                                    }
                                    for c in value["tool_calls"]
                                ]
                                for call in step["tool_calls"]:
                                    calls[call["tool_call_id"]] = step
                        steps.append(step)
                    elif role == "tool" and value["tool_call_id"] in calls:
                        calls[value["tool_call_id"]].setdefault(
                            "observation", {"results": []}
                        )["results"].append(
                            {
                                "source_call_id": value["tool_call_id"],
                                "content": value["content"],
                            }
                        )
                elif item["kind"] == "final" and steps:
                    steps[-1]["extra"]["lingcore"]["guarded_final"] = value
        if not steps:
            return None
        for index, step in enumerate(steps, 1):
            step["step_id"] = index
        return {
            "schema_version": "ATIF-v1.7",
            "session_id": execution_id,
            "agent": {
                "name": "LingCore",
                "version": plan.runtime.lingcore,
                "model_name": plan.candidate.model.id,
                "tool_definitions": tool_definitions,
            },
            "steps": steps,
            "final_metrics": {
                "total_prompt_tokens": sum(
                    r.get("input_tokens") or 0 for r in requests.values()
                ),
                "total_completion_tokens": sum(
                    r.get("output_tokens") or 0 for r in requests.values()
                ),
                "total_cached_tokens": sum(
                    r.get("cached_input_tokens") or 0 for r in requests.values()
                ),
                "total_steps": len(steps),
            },
            "extra": {
                "lingcore": {
                    "execution_id": execution_id,
                    "trace_complete": inspection.complete,
                    "redacted": True,
                    "events": events,
                    "requests": list(requests.values()),
                    "unattributed_request_ids": sorted(set(requests) - attributed),
                }
            },
        }
    except (ValueError, OSError, KeyError, TypeError):
        return None
