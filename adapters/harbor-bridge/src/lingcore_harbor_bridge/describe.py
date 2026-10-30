"""Offline inspection through the pinned Harbor models, without a container."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any, get_args

from harbor.environments.docker import docker
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import (
    resolve_task_verifier_mode,
    resolve_verifier_environment_definition,
)
from harbor.models.trajectories.trajectory import Trajectory
from harbor.models.trial.error_policy import ERROR_POLICIES, UNKNOWN_ERROR_POLICY
from harbor.publisher.packager import Packager

from lingcore_harbor_bridge import HARBOR_VERSION, PROTOCOL, __version__

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024


class DescriptionError(ValueError):
    """An intentional, value-free diagnostic safe to print as JSON."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_metadata(path: Path, limit: int = MAX_METADATA_BYTES) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise DescriptionError("unsafe_or_oversized_metadata")
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise DescriptionError("unsafe_or_oversized_metadata")
    return data


def validate_tree(root: Path) -> None:
    """Refuse links and special files before Harbor's own path-based readers."""
    if root.is_symlink() or not root.is_dir():
        raise DescriptionError("unsafe_task_tree")
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in [*dirs, *files]:
            info = (Path(current) / name).lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise DescriptionError("unsafe_task_tree")


def package_digest(root: Path) -> str:
    files = []
    for file in sorted(root.rglob("*")):
        if "__pycache__" in file.relative_to(root).parts or file.suffix in {
            ".pyc",
            ".pyo",
        }:
            continue
        if file.is_symlink():
            raise DescriptionError("unsafe_package_tree")
        if file.is_dir():
            continue
        data = read_metadata(file, 16 * MAX_METADATA_BYTES)
        files.append([file.relative_to(root).as_posix(), digest(data)])
    return digest(canonical(files))


def environment(config: EnvironmentConfig, root: Path) -> dict[str, Any]:
    compose = root / "docker-compose.yaml"
    if compose.exists():
        # Use the same YAML parser as Harbor's Docker backend; the bridge owns
        # no YAML dependency and does not invoke `docker compose config`.
        document = docker.yaml.safe_load(read_metadata(compose).decode("utf-8"))
        services = document.get("services") if isinstance(document, dict) else None
        if (
            not isinstance(services, dict)
            or not services
            or not all(
                isinstance(name, str) and isinstance(value, dict)
                for name, value in services.items()
            )
        ):
            raise DescriptionError("metadata_unreadable")
        return {
            "kind": "compose",
            "containers": len(set(services) | {"main"}),
            "pinned": False,
        }
    if config.docker_image:
        return {
            "kind": "image",
            "containers": 1,
            "pinned": bool(
                re.fullmatch(r".+@sha256:[0-9a-fA-F]{64}", config.docker_image)
            ),
        }
    if (root / "Dockerfile").is_file():
        return {"kind": "dockerfile", "containers": 1, "pinned": False}
    raise DescriptionError("metadata_unreadable")


def describe_task(path: Path) -> dict[str, Any]:
    validate_tree(path)
    read_metadata(path / "task.toml")
    read_metadata(path / "instruction.md", MAX_JSON_BYTES)
    task = Task(path)
    config = task.config
    agent_env = environment(config.environment, task.paths.environment_dir)
    verifier = resolve_verifier_environment_definition(config, task.paths)
    verifier_env = (
        environment(verifier.config, verifier.directory) if verifier else None
    )
    gpu_counts = [config.environment.gpus or 0]
    if verifier:
        gpu_counts.append(verifier.config.gpus or 0)
    return {
        "id": path.name,
        "path": path.name,
        "status": "ok",
        "name": task.name,
        "config": config.model_dump(mode="json"),
        "instruction_digest": digest(task.instruction.encode("utf-8")),
        "harbor_digest": "sha256:" + Packager.compute_content_hash(path)[0],
        "environments": {"agent": agent_env, "verifier": verifier_env},
        "properties": {
            "os": config.environment.os.value,
            "gpus": max(gpu_counts),
            "containers": max(
                agent_env["containers"],
                verifier_env["containers"] if verifier_env else 1,
            ),
            "cpus": config.environment.cpus,
            "memory_mb": config.environment.memory_mb,
            "storage_mb": config.environment.storage_mb,
            "task_tools": bool(
                config.environment.mcp_servers or config.environment.skills_dir
            ),
            "steps": len(config.steps or []),
            "verifier_mode": resolve_task_verifier_mode(config).value,
        },
    }


def describe(dataset: Path) -> dict[str, Any]:
    if metadata.version("harbor") != HARBOR_VERSION:
        raise DescriptionError("unsupported_harbor_version")
    if dataset.is_symlink() or not dataset.is_dir():
        raise DescriptionError("dataset_unreadable")
    tasks = []
    for path in sorted(dataset.iterdir()):
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        if path.is_symlink():
            raise DescriptionError("unsafe_task_tree")
        if not path.is_dir():
            continue
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", path.name):
            raise DescriptionError("invalid_task_id")
        try:
            tasks.append(describe_task(path))
        except Exception as exc:
            tasks.append(
                {
                    "id": path.name,
                    "path": path.name,
                    "status": "metadata_unreadable",
                    "error_type": type(exc).__name__,
                    "code": str(exc)
                    if isinstance(exc, DescriptionError)
                    else "metadata_unreadable",
                }
            )
    if not tasks:
        raise DescriptionError("empty_dataset")
    table = {
        name: {**asdict(policy), "reward_valid": policy.reward_valid}
        for name, policy in sorted(ERROR_POLICIES.items())
    }
    unknown = {
        **asdict(UNKNOWN_ERROR_POLICY),
        "reward_valid": UNKNOWN_ERROR_POLICY.reward_valid,
    }
    import harbor

    dependencies = {}
    for requirement in metadata.requires("harbor") or ():
        if "extra ==" in requirement:
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match:
            dependencies[match[0]] = metadata.version(match[0])
    result = {
        "schema_version": "lingcore.harbor-description.v1",
        "protocol": PROTOCOL,
        "bridge": {
            "version": __version__,
            "digest": package_digest(Path(__file__).parent),
        },
        "harbor": {
            "version": HARBOR_VERSION,
            "digest": package_digest(Path(harbor.__file__).parent),
            "dependencies": dependencies,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "error_policy": {
            "digest": digest(canonical({"table": table, "unknown": unknown})),
            "table": table,
            "unknown": unknown,
        },
        "atif_versions": list(
            get_args(Trajectory.model_fields["schema_version"].annotation)
        ),
        "tasks": tasks,
    }
    if len(canonical(result)) + 1 > MAX_JSON_BYTES:
        raise DescriptionError("description_too_large")
    return result
