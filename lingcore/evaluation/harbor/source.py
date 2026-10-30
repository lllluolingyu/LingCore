"""Offline Harbor selection and split input/verifier identity."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from lingcore.config import AgentProfile
from lingcore.evaluation.config import EvaluationConfigError, ResolvedSuite
from lingcore.evaluation.harbor.binding import RefusingTerminal
from lingcore.evaluation.harbor.process import bridge_environment, find_bridge, invoke
from lingcore.evaluation.harbor.tools import terminal_registry
from lingcore.evaluation.identity import (
    HarborExecutionIdentity,
    HarborGradingIdentity,
    HarborSourceIdentity,
    HarborTaskIdentity,
    PlanIdentity,
    SuiteIdentity,
    assert_secret_free,
    candidate_identity,
    runtime_identity,
    value_digest,
)
from lingcore.evaluation.models import (
    HarborExecutionConfig,
    HarborGradingPolicy,
    HarborSource,
    Suite,
    TrialSpec,
)
from lingcore.evaluation.storage import canonical_json, directory
from lingcore.paths import ConfinedDirectory
from lingcore.plugins.discovery import local_skills
from lingcore.tracing.config import known_secrets

# Probe + baseline (2 x 30), interrupt drain (30), sweep (15), export (30),
# and transport/scheduling reserve (15). Keep in sync with the bridge bounds.
HARNESS_MARGIN_SECONDS = 150


def select(
    source: HarborSource, description: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    required = source.selection.required.model_dump()
    seen = set()
    for task in sorted(description["tasks"], key=lambda t: t["id"]):
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", task["id"])
            or task.get("path", task["id"]) != task["id"]
        ):
            raise EvaluationConfigError("invalid Harbor task locator", "task_id")
        if task["id"] in seen:
            raise EvaluationConfigError("duplicate Harbor task ID", "task_id")
        seen.add(task["id"])
        reasons = []
        if task["status"] != "ok":
            reasons.append("metadata_unreadable")
        else:
            properties = task["properties"]
            for key in ("os", "gpus", "containers", "task_tools"):
                if properties.get(key) is None:
                    reasons.append("metadata_unreadable")
                elif (
                    properties[key] != required[key]
                    if key in {"os", "task_tools"}
                    else properties[key] > required[key]
                ):
                    reasons.append(key)
            if properties.get("steps") != 0:
                reasons.append("steps")
            for key in ("cpus", "memory_mb", "storage_mb"):
                limit = required["max_" + key]
                if limit is not None and (
                    properties.get(key) is None or properties[key] > limit
                ):
                    reasons.append(
                        key
                        if properties.get(key) is not None
                        else "metadata_unreadable"
                    )
        if (
            source.selection.policy == "explicit"
            and task["id"] not in source.selection.tasks
        ):
            reasons.append("not_selected")
        if not reasons and len(included) >= source.selection.count:
            reasons.append("selection_count")
        if reasons:
            excluded.append({"task": task["id"], "reasons": sorted(set(reasons))})
        else:
            included.append(task)
    expected = set(source.selection.tasks)
    if (
        source.selection.policy == "explicit"
        and (
            len(expected) != source.selection.count
            or expected != {t["id"] for t in included}
        )
    ) or len(included) < source.selection.count:
        raise EvaluationConfigError(
            "not enough eligible Harbor tasks, or explicit selection is ineligible",
            "task_selection",
        )
    return included, excluded


def resolve_harbor(
    suite: Suite,
    profile: AgentProfile,
    path: Path,
    *,
    bridge: Path | None,
    require_credentials: bool,
) -> ResolvedSuite:
    assert isinstance(suite.source, HarborSource) and isinstance(
        suite.execution, HarborExecutionConfig
    )
    if (
        set(profile.tools) != set(terminal_registry().names())
        or profile.skills
        or profile.plugins
        or local_skills(profile)
        or profile.sessions.enabled
        or profile.persona.project_instructions
        or profile.loop.parallel_tools
    ):
        raise EvaluationConfigError(
            "Harbor requires only terminal tools, no skills/plugins/sessions/project instructions, and sequential tools",
            "harbor_profile",
        )
    env = bridge_environment(profile, suite.execution.harbor.pass_env)
    if require_credentials:
        for cfg in (profile.llm, profile.media_fallback.image):
            if cfg is not None:
                try:
                    cfg.resolve_api_key(getattr(profile, "_profile_env", {}))
                except Exception:
                    raise EvaluationConfigError(
                        "Harbor candidate credential is unavailable",
                        "model_credentials",
                    ) from None
    executable = find_bridge(bridge)
    dataset = (path.parent / suite.source.path).absolute()
    description = invoke(executable, ["describe", "--dataset", str(dataset)], env)
    if (
        description.get("protocol") != "lingcore.harbor-bridge.v1"
        or description.get("harbor", {}).get("version") != "0.25.0"
    ):
        raise EvaluationConfigError(
            "unsupported Harbor bridge protocol or version", "bridge_version"
        )
    selected, _ = select(suite.source, description)
    wall = suite.execution.limits.wall_seconds
    if wall is None or any(
        wall
        > t["config"]["agent"]["timeout_sec"]
        * suite.execution.harbor.timeout_multipliers.agent
        - HARNESS_MARGIN_SECONDS
        for t in selected
    ):
        raise EvaluationConfigError(
            f"wall_seconds must end at least {HARNESS_MARGIN_SECONDS} seconds before every Harbor agent timeout",
            "harbor_deadline",
        )
    return ResolvedSuite(suite, (), profile, executable, description, dataset)


def task_files(root: Path, secrets: tuple[str, ...]) -> dict[str, str]:
    """Hash every task file with bounded memory and no skipped dotfiles."""
    result = {}
    overlap = max((len(s.encode()) for s in secrets), default=0) + 8

    def walk(current: ConfinedDirectory, prefix: str = "") -> None:
        for name, is_dir, is_file, is_link in sorted(current.iter_entries()):
            if is_link or not (is_file or is_dir):
                raise EvaluationConfigError(
                    "task contains an unsafe filesystem entry", "unsafe_task_tree"
                )
            relative = prefix + name
            if is_dir:
                with current.subdirectory(name) as child:
                    walk(child, relative + "/")
            else:
                hashed = hashlib.sha256()
                tail = b""
                with current.open_regular(name) as file:
                    while chunk := file.read(1024 * 1024):
                        hashed.update(chunk)
                        assert_secret_free(tail + chunk, secrets)
                        tail = (tail + chunk)[-overlap:]
                result[relative] = "sha256:" + hashed.hexdigest()

    with directory(root) as current:
        walk(current)
    return result


def task_identity(
    task: dict[str, Any], dataset: Path, secrets: tuple[str, ...]
) -> HarborTaskIdentity:
    files = task_files(dataset / task["path"], secrets)
    assert_secret_free(canonical_json(task["config"]), secrets)
    inputs: list[tuple[str, str]] = []
    verifier: list[tuple[str, str]] = []
    support: list[tuple[str, str]] = []
    for name, digest in sorted(files.items()):
        if name == "task.toml":
            continue
        destination = (
            inputs
            if name == "instruction.md" or name.startswith("environment/")
            else verifier
            if name.startswith("tests/")
            else support
        )
        destination.append((name, digest))
    config = dict(task["config"])
    verifier_config = config.pop("verifier")
    artifacts = config.pop("artifacts", {})
    return HarborTaskIdentity(
        id=task["id"],
        name=task["name"],
        path=task["path"],
        input_digest=value_digest({"files": inputs, "config": config}),
        verifier_digest=value_digest(
            {"files": verifier, "config": verifier_config, "artifacts": artifacts}
        ),
        support_digest=value_digest(support),
        harbor_digest=task["harbor_digest"],
        instruction_digest=task["instruction_digest"],
        verifier={
            "mode": task["properties"]["verifier_mode"],
            "timeout_sec": verifier_config["timeout_sec"],
            "artifacts": artifacts,
        },
        agent_timeout_sec=config["agent"]["timeout_sec"],
        resources={
            k: task["properties"][k] for k in ("cpus", "memory_mb", "storage_mb")
        },
        images=task["environments"],
    )


async def build_harbor_identity(
    resolved: ResolvedSuite, suite_path: Path, base_dir: Path
) -> PlanIdentity:
    suite = resolved.suite
    assert (
        isinstance(suite.source, HarborSource)
        and isinstance(suite.execution, HarborExecutionConfig)
        and isinstance(suite.grading, HarborGradingPolicy)
    )
    assert resolved.description is not None and resolved.dataset is not None
    selected, excluded = select(suite.source, resolved.description)
    secrets = known_secrets(resolved.profile)
    tasks = [task_identity(t, resolved.dataset, secrets) for t in selected]
    profile_path = (suite_path.parent / suite.agent.profile).resolve()
    if profile_path.is_dir():
        profile_path /= "config.yaml"
    candidate = await candidate_identity(
        resolved.profile,
        suite,
        profile_path=profile_path,
        tool_registry=terminal_registry(),
        environment_binding=RefusingTerminal(resolved.profile.terminal_options),
    )
    candidate = candidate.model_copy(
        update={
            "profile": candidate.profile.model_copy(
                update={"path": os.path.relpath(profile_path, base_dir)}
            )
        }
    )
    runtime = await asyncio.to_thread(runtime_identity)
    selection = suite.source.selection.model_dump(mode="json") | {
        "included": [t.id for t in tasks],
        "excluded": excluded,
    }
    dataset = {
        "ref": suite.source.dataset,
        "path": os.path.relpath(resolved.dataset, base_dir),
        "release_source": suite.source.release_source,
        "training_use": suite.source.training_use,
    }
    source_fields: dict[str, Any] = dict(
        dataset=dataset, selection=selection, tasks=tasks
    )
    source = HarborSourceIdentity(
        digest=value_digest(
            {
                "dataset": {k: v for k, v in dataset.items() if k != "path"},
                "selection": selection,
                "tasks": [t.model_dump(exclude={"path"}) for t in tasks],
            }
        ),
        **source_fields,
    )
    harness = {
        k: v
        for k, v in resolved.description.items()
        if k not in {"tasks", "schema_version"}
    }
    harness.update(
        environment="docker",
        timeout_multipliers=suite.execution.harbor.timeout_multipliers.model_dump(
            mode="json"
        ),
        atif="ATIF-v1.7",
    )
    harness["digest"] = value_digest(harness)
    execution = suite.execution.model_dump(
        mode="json", exclude={"infrastructure_retries", "retry_on", "harbor"}
    )
    execution.update(
        harness=harness,
        pass_env=suite.execution.harbor.pass_env,
        order="declared",
        retry={
            "infrastructure_retries": suite.execution.infrastructure_retries,
            "retry_on": suite.execution.retry_on,
        },
    )
    limitations = ["release_unverified"] if suite.source.release_source else []
    if any(not env["pinned"] for task in tasks for env in task.images.values() if env):
        limitations.append("image_unpinned")
    if any(
        (suite.execution.limits.wall_seconds or 0) + HARNESS_MARGIN_SECONDS
        < t.agent_timeout_sec * suite.execution.harbor.timeout_multipliers.agent
        for t in tasks
    ):
        limitations.append("reduced_limits")
    if any(not c.set for c in candidate.credentials):
        limitations.append("credential_unset")
    grading = suite.grading.model_dump(mode="json")
    plan = PlanIdentity(
        suite=SuiteIdentity(
            name=suite.name,
            track=suite.track,
            path=os.path.relpath(suite_path, base_dir),
            digest=value_digest(suite.model_dump(mode="json")),
        ),
        source=source,
        candidate=candidate,
        runtime=runtime,
        execution=HarborExecutionIdentity(digest=value_digest(execution), **execution),
        trace=suite.trace,
        grading=HarborGradingIdentity(
            digest=value_digest(
                {"policy": grading, "tasks": [t.verifier_digest for t in tasks]}
            ),
            **grading,
        ),
        trials=[
            TrialSpec(trial_id=f"{t.id}--candidate--{i}", case_id=t.id, repetition=i)
            for t in tasks
            for i in range(1, suite.execution.repetitions + 1)
        ],
        limitations=limitations,
    )
    assert_secret_free(canonical_json(plan.model_dump(mode="json")), secrets)
    return plan
