"""Strict, self-verifying local plan locks and value-free drift reports."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import model_validator

from lingcore.evaluation.config import EvaluationConfigError, resolve_suite
from lingcore.evaluation.identity import (
    Digest,
    PlanIdentity,
    assert_secret_free,
    build_identity,
    identity_body,
    value_digest,
)
from lingcore.evaluation.models import EvaluationDiagnostic, StrictModel
from lingcore.evaluation.storage import canonical_json, read_bytes, write_json
from lingcore.paths import PathEscapeError
from lingcore.tracing.config import known_secrets


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Lock(PlanIdentity):
    schema_version: Literal["lingcore.eval-lock.v1"] = "lingcore.eval-lock.v1"
    digest: Digest
    plan_digest: Digest
    created_at: str

    @model_validator(mode="after")
    def _locators(self) -> Lock:
        for path in (
            self.suite.path,
            self.candidate.profile.path,
            *(c.path for c in self.source.cases),
        ):
            if (
                not path
                or Path(path).is_absolute()
                or PureWindowsPath(path).drive
                or "\\" in path
                or path.startswith("~")
                or "\0" in path
            ):
                raise ValueError("lock locators must be relative paths")
        return self


class Drift(StrictModel):
    field: str
    kind: Literal["changed", "missing", "added", "unreadable"]
    diagnostic: EvaluationDiagnostic | None = None

    def describe(self) -> str:
        summary = f"{self.kind}: {self.field}"
        if self.diagnostic is not None:
            summary += f" ({self.diagnostic.code}: {self.diagnostic.message})"
        return summary


def make_lock(plan: PlanIdentity, *, created_at: str | None = None) -> Lock:
    lock = Lock(
        **plan.model_dump(mode="json"),
        digest="sha256:" + "0" * 64,
        plan_digest=value_digest(identity_body(plan)),
        created_at=created_at or utc_now(),
    )
    return lock.model_copy(
        update={
            "digest": value_digest(lock.model_dump(mode="json", exclude={"digest"}))
        }
    )


async def resolve(suite_path: Path, lock_path: Path) -> Lock:
    resolved = await asyncio.to_thread(
        resolve_suite,
        suite_path.absolute(),
        require_credentials=False,
        allow_locked=True,
    )
    plan = await build_identity(
        resolved, suite_path.absolute(), lock_path.absolute().parent
    )
    lock = make_lock(plan)
    assert_secret_free(
        canonical_json(lock.model_dump(mode="json")), known_secrets(resolved.profile)
    )
    return lock


def write_lock(lock: Lock, path: Path, *, replace: bool = False) -> None:
    try:
        write_json(path.absolute(), lock.model_dump(mode="json"), replace=replace)
    except FileExistsError:
        raise EvaluationConfigError(
            "lock already exists; use --replace to replace it atomically", "lock_exists"
        ) from None


def parse_lock(data: bytes) -> Lock:
    try:
        raw = json.loads(data)
        if not isinstance(raw, dict) or value_digest(
            {k: v for k, v in raw.items() if k != "digest"}
        ) != raw.get("digest"):
            raise ValueError("raw lock self-digest mismatch")
        lock = Lock.model_validate(raw)
        if (
            value_digest(lock.model_dump(mode="json", exclude={"digest"}))
            != lock.digest
        ):
            raise ValueError("lock self-digest mismatch")
        if value_digest(identity_body(lock)) != lock.plan_digest:
            raise ValueError("plan digest mismatch")
        return lock
    except (ValueError, OSError, PathEscapeError):
        raise EvaluationConfigError(
            "lock is unreadable, corrupt, or uses an unsupported schema", "invalid_lock"
        ) from None


def load_lock(path: Path) -> Lock:
    try:
        return parse_lock(read_bytes(path.absolute()))
    except (OSError, PathEscapeError):
        raise EvaluationConfigError("lock is unreadable", "invalid_lock") from None


def _diff(old: Any, new: Any, prefix: str = "") -> list[Drift]:
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in old:
                result.append(Drift(field=name, kind="added"))
            elif key not in new:
                result.append(Drift(field=name, kind="missing"))
            else:
                result.extend(_diff(old[key], new[key], name))
        return result
    if (
        isinstance(old, list)
        and isinstance(new, list)
        and old
        and new
        and isinstance(old[0], dict)
        and isinstance(new[0], dict)
    ):
        key = next(
            (
                key
                for key in ("id", "name", "trial_id")
                if key in old[0] and key in new[0]
            ),
            None,
        )
        if key is not None:
            result = _diff({v[key]: v for v in old}, {v[key]: v for v in new}, prefix)
            if [v[key] for v in old] != [v[key] for v in new]:
                result.append(Drift(field=prefix + ".order", kind="changed"))
            return result
    return [Drift(field=prefix, kind="changed")]


async def validate(lock: Lock, lock_path: Path) -> list[Drift]:
    base = lock_path.absolute().parent
    for field, path in [
        ("suite", lock.suite.path),
        ("candidate.profile", lock.candidate.profile.path),
        *((f"source.cases.{c.id}", c.path) for c in lock.source.cases),
    ]:
        if not (base / path).is_file():
            return [Drift(field=field, kind="missing")]
    try:
        current = await resolve(base / lock.suite.path, lock_path)
    except EvaluationConfigError as exc:
        return [Drift(field="inputs", kind="unreadable", diagnostic=exc.diagnostic)]
    except (ValueError, OSError, PathEscapeError):
        return [Drift(field="inputs", kind="unreadable")]
    return _diff(identity_body(lock), identity_body(current))
