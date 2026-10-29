"""Private, confined immutable records and atomically replaced derived views."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

from lingcore.paths import ConfinedDirectory, confined_directory

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def finish_io(
    operation: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """Join a bounded filesystem operation before propagating cancellation.

    Cancelling to_thread alone leaves its worker running. In particular, evidence
    collection must finish before the trial removes the workspace it is reading.
    """
    task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            if not cancelled:
                raise
    if cancelled:
        # Retrieve any worker exception, but preserve the caller's cancellation.
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


@contextmanager
def directory(path: Path, *, create: bool = False) -> Iterator[ConfinedDirectory]:
    path = path.absolute()
    with confined_directory(
        Path(path.anchor),
        path.relative_to(path.anchor),
        create=create,
        permissions=0o700,
        durable=create,
    ) as root:
        yield root


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_bytes(path: Path, data: bytes, *, replace: bool = False) -> None:
    with directory(path.parent, create=True) as root:
        name = ".tmp-" + uuid4().hex if replace else path.name
        try:
            with root.open_exclusive(name, 0o600) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if replace:
                root.replace(name, path.name)
            root.sync()
        finally:
            if replace:
                root.unlink(name, missing_ok=True)


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    write_bytes(path, canonical_json(value) + b"\n", replace=replace)


def read_bytes(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    with directory(path.parent) as root:
        return root.read_regular(path.name, max_bytes=max_bytes)


def read_json(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    record = json.loads(read_bytes(path, max_bytes=max_bytes))
    if not isinstance(record, dict):
        raise ValueError("evaluation records must be JSON objects")
    return record


@contextmanager
def job_lock(job: Path) -> Iterator[None]:
    from lingcore.evaluation.config import EvaluationConfigError

    with directory(job) as root:
        with root.open_lock_file(".lock") as lock:
            if not lock.try_acquire():
                raise EvaluationConfigError(
                    "evaluation job is already locked", "job_locked"
                )
            yield
