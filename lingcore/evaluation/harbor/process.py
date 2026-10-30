"""Credential-free bridge discovery and bounded offline invocations."""

from __future__ import annotations

import asyncio
import json
import os
import selectors
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from lingcore.config import AgentProfile
from lingcore.doctor import required_environment_names
from lingcore.evaluation.config import EvaluationConfigError

ALLOW_ENV = frozenset(
    "PATH HOME USER LANG LC_ALL TMPDIR XDG_CONFIG_HOME XDG_CACHE_HOME DOCKER_HOST DOCKER_CONFIG DOCKER_CONTEXT".split()
)


def bridge_environment(profile: AgentProfile, pass_env: list[str]) -> dict[str, str]:
    credentials = required_environment_names(profile)
    if set(pass_env) & credentials or any(not name.isidentifier() for name in pass_env):
        raise EvaluationConfigError(
            "bridge pass_env must contain non-credential environment names",
            "bridge_environment",
        )
    # Profile-local .env is never merged, even for explicitly passed names.
    return {
        name: value
        for name, value in os.environ.items()
        if name in (ALLOW_ENV | set(pass_env)) - credentials
    }


def find_bridge(override: Path | None = None) -> Path:
    command = (
        str(override)
        if override
        else os.environ.get("LINGCORE_HARBOR_BRIDGE", "lingcore-harbor-bridge")
    )
    path = shutil.which(command)
    if path is None:
        raise EvaluationConfigError(
            "Harbor bridge is unavailable; install adapters/harbor-bridge separately or set --bridge",
            "bridge_missing",
        )
    return Path(path).absolute()


def invoke(
    bridge: Path,
    args: list[str],
    env: dict[str, str],
    *,
    timeout: float = 120,
    limit: int = 4 * 1024 * 1024,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    process = subprocess.Popen(
        [str(bridge), *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    data = bytearray()
    assert process.stdout is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if cancel is not None and cancel.is_set():
                    raise EvaluationConfigError(
                        "bridge command cancelled", "bridge_cancelled"
                    )
                if remaining <= 0:
                    raise EvaluationConfigError(
                        "bridge command timed out", "bridge_timeout"
                    )
                if not selector.select(
                    min(remaining, 0.2) if cancel is not None else remaining
                ):
                    continue
                chunk = os.read(
                    process.stdout.fileno(), min(65536, limit + 1 - len(data))
                )
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > limit:
                    raise EvaluationConfigError(
                        "bridge response exceeds its byte limit", "bridge_response"
                    )
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
        value = json.loads(data)
        if isinstance(value, dict) and isinstance(value.get("error"), dict):
            code = value["error"].get("code")
            if code in {
                "unsupported_harbor_version",
                "dataset_unreadable",
                "empty_dataset",
                "invalid_task_id",
                "unsafe_task_tree",
                "unsafe_or_oversized_metadata",
                "description_too_large",
                "missing_or_ambiguous_trial_result",
                "unsafe_harbor_output",
                "harbor_exit",
            }:
                raise EvaluationConfigError(
                    "Harbor bridge: " + code.replace("_", " "), code
                )
        if not isinstance(value, dict) or process.returncode or "error" in value:
            raise EvaluationConfigError(
                "bridge command failed; inspect its installation and task metadata",
                "bridge_failed",
            )
        return value
    except (ValueError, subprocess.TimeoutExpired):
        raise EvaluationConfigError(
            "bridge response is invalid", "bridge_response"
        ) from None
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            except ProcessLookupError:
                process.wait()
        process.stdout.close()


async def invoke_cancellable(
    bridge: Path, args: list[str], env: dict[str, str], *, timeout: float
) -> dict[str, Any]:
    """Join Harbor teardown on cancellation, without waiting for its trial budget."""
    cancel = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(invoke, bridge, args, env, timeout=timeout, cancel=cancel)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise
