"""Resolve private trace destinations against effective runtime state."""

from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError
from lingcore.paths import PathEscapeError, resolve_confined
from lingcore.tracing.local import LocalTraceSink
from lingcore.tracing.recorder import NoopRecorder, TraceRecorder


def known_secrets(profile: AgentProfile) -> tuple[str, ...]:
    """Credentials declared by the profile or named as credentials in the env.

    Model names, endpoints, paths, and other ordinary exported settings are
    configuration, regardless of their length. They must survive fingerprinting.
    """
    environment = {**os.environ, **getattr(profile, "_profile_env", {})}
    declared: set[str] = set()

    def declarations(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {
                    "api_key_env",
                    "token_env",
                    "access_token_env",
                    "password_env",
                    "secret_env",
                } and isinstance(item, str):
                    declared.add(item)
                else:
                    declarations(item)
        elif isinstance(value, list):
            for item in value:
                declarations(item)

    declarations(profile.model_dump())
    credential_name = re.compile(
        r"(?:^|_)(?:API_KEY|API_TOKEN|ACCESS_TOKEN|AUTH_TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|CLIENT_SECRET|PRIVATE_KEY)$"
    )
    return tuple(
        sorted(
            {
                value
                for name, value in environment.items()
                if value and (name in declared or credential_name.search(name.upper()))
            },
            key=lambda value: (-len(value), value),
        )
    )


def build_recorder(
    profile: AgentProfile, workspace: Path, state_dir: Path | None
) -> TraceRecorder:
    cfg = profile.tracing
    if not cfg.enabled:
        return NoopRecorder()
    raw = Path(cfg.path)
    root = state_dir or getattr(profile, "_source_dir", None)
    if raw.is_absolute():
        if not cfg.allow_absolute_path:
            raise ConfigError(
                "absolute tracing.path requires tracing.allow_absolute_path"
            )
        destination = raw.resolve()
    else:
        if root is None:
            raise ConfigError(
                "relative tracing.path requires a profile or state directory"
            )
        try:
            destination = resolve_confined(root, raw)
        except PathEscapeError:
            raise ConfigError("tracing.path escapes private profile state") from None
    package = Path(__file__).resolve().parents[1]
    if state_dir is not None and not destination.is_relative_to(state_dir.resolve()):
        raise ConfigError("tracing.path escapes the explicit state directory")
    workspace = workspace.resolve()
    if destination.is_relative_to(package) or package.is_relative_to(destination):
        raise ConfigError("trace destination overlaps the installed package")
    if destination.is_relative_to(workspace) or workspace.is_relative_to(destination):
        raise ConfigError("trace destination overlaps the effective workspace")
    run_id = str(uuid4())
    secrets = known_secrets(profile)
    manifest = {
        "schema_version": "lingcore.trace.v1",
        "run_id": run_id,
        "capture": cfg.model_dump(exclude={"path", "allow_absolute_path"}),
        "durability": "flushed; no fsync guarantee",
    }
    sink = LocalTraceSink(
        destination / run_id,
        manifest=manifest,
        queue_capacity=cfg.queue_capacity,
        flush_interval_seconds=cfg.flush_interval_seconds,
        max_bytes=cfg.max_bytes,
    )
    return TraceRecorder(
        sink,
        run_id=run_id,
        content=cfg.content,
        secrets=secrets,
        artifact_max_bytes=cfg.artifact_max_bytes,
    )
