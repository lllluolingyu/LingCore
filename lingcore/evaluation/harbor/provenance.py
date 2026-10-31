"""Offline verification against reviewed, bundled publisher metadata.

The catalog links a release to its published Hub packages, not to byte-identical
Git exports (which can use Dockerfiles instead of the Hub's prebuilt images).
Suite authors cannot supply their own trusted manifest or eligibility policy.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from lingcore.evaluation.config import EvaluationConfigError
from lingcore.evaluation.identity import value_digest
from lingcore.evaluation.models import HarborExecutionConfig, HarborSource, Suite

CATALOG = {"terminal-bench/terminal-bench@4.0.0": "terminal-bench-4.0.0.json"}


def release_manifest(dataset: str) -> dict[str, Any] | None:
    filename = CATALOG.get(dataset)
    if filename is None:
        return None
    data: dict[str, Any] = json.loads(
        files(__package__).joinpath("releases", filename).read_text(encoding="utf-8")
    )
    return data


def verify_release(source: HarborSource, description: dict[str, Any]) -> dict[str, Any]:
    manifest = release_manifest(source.dataset)
    if manifest is None:
        return {"status": "unverified", "reason": "unknown_release"}
    if any(
        manifest["release_source"].get(key) != value
        for key, value in source.release_source.items()
    ):
        raise EvaluationConfigError(
            "release_source differs from the bundled publisher metadata",
            "release_source_mismatch",
        )
    expected = manifest["tasks"]
    observed = {task["id"]: task.get("harbor_digest") for task in description["tasks"]}
    if (
        len(observed) != len(description["tasks"])
        or not observed
        or any(t.get("package_complete") is not True for t in description["tasks"])
        or any(
            expected.get(name) != digest or digest is None
            for name, digest in observed.items()
        )
    ):
        raise EvaluationConfigError(
            "local tasks differ from the published Hub packages; export the pinned release or use a distinct local dataset name",
            "release_task_mismatch",
        )
    return {
        "status": "verified" if set(observed) == set(expected) else "verified_subset",
        "manifest_digest": value_digest(manifest),
        "hub_version": manifest["hub_version"],
        "hub_version_id": manifest["hub_version_id"],
        "release_source": manifest["release_source"],
        "release_url": manifest["release_url"],
        "hub_url": manifest["hub_url"],
        "policy_url": manifest["policy_url"],
        "policy": manifest["policy"],
        "published_tasks": len(expected),
        "verified_tasks": sorted(observed),
    }


def check_benchmark(
    suite: Suite, provenance: dict[str, Any], selected: list[dict[str, Any]]
) -> None:
    if suite.track != "benchmark":
        return
    assert isinstance(suite.source, HarborSource)
    assert isinstance(suite.execution, HarborExecutionConfig)
    reasons = []
    if provenance["status"] != "verified":
        reasons.append("verified_complete_release")
    elif {t["id"] for t in selected} != set(provenance["verified_tasks"]):
        reasons.append("full_task_selection")
    policy = provenance.get("policy", {})
    if suite.execution.repetitions != policy.get("repetitions"):
        reasons.append("release_repetitions")
    if suite.grading.model_dump().get("reward") != policy.get("reward"):
        reasons.append("release_reward_policy")
    if any(
        v != 1 for v in suite.execution.harbor.timeout_multipliers.model_dump().values()
    ):
        reasons.append("default_timeout_multipliers")
    limits = suite.execution.limits
    if (
        any(
            getattr(limits, k) is not None
            for k in ("wall_seconds", "model_requests", "tool_calls", "reported_tokens")
        )
        or limits.usage_missing != "ignore"
        or limits.automatic_turns != 0
    ):
        reasons.append("task_defined_budgets")
    if not suite.trace.required:
        reasons.append("required_trace")
    if suite.source.training_use != "prohibited":
        reasons.append("training_prohibited")
    if any(not e["pinned"] for t in selected for e in t["environments"].values() if e):
        reasons.append("pinned_images")
    if reasons:
        raise EvaluationConfigError(
            "benchmark requirements not met: " + ", ".join(reasons), "benchmark_policy"
        )
