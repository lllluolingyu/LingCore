"""Remove a LingCore OCI sandbox if its owning process disappears."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_CONTAINER_NAME = re.compile(r"^lingcore-[a-f0-9]{32}$")
_ENV_FILE_PREFIX = "lingcore-sandbox-env-"


def _validated_cleanup_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute() or not path.name.startswith(_ENV_FILE_PREFIX):
        raise ValueError
    try:
        if path.parent.resolve() != Path(tempfile.gettempdir()).resolve():
            raise ValueError
    except OSError:
        raise ValueError from None
    return path


def _discard(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _remove_with_retries(executable: Path, name: str) -> int:
    """Catch a create/start race after owner cancellation or sudden death."""
    deadline = time.monotonic() + 65
    delay = 0.25
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 1
        try:
            completed = subprocess.run(
                [str(executable), "rm", "--force", "--volumes", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=min(5, remaining),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            return 0
        time.sleep(min(delay, max(0, deadline - time.monotonic())))
        delay = min(delay * 1.5, 2)


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        return 2
    executable = Path(sys.argv[1])
    name = sys.argv[2]
    if not executable.is_absolute() or not _CONTAINER_NAME.fullmatch(name):
        return 2
    try:
        cleanup_path = _validated_cleanup_path(
            sys.argv[3] if len(sys.argv) == 4 else None
        )
    except ValueError:
        return 2

    # A deliberate one-byte disarm means normal cleanup already removed the
    # container. EOF means the owner exited/crashed and left cleanup to us.
    disarm = sys.stdin.buffer.read(1)
    _discard(cleanup_path)
    if disarm == b"D":
        return 0
    return _remove_with_retries(executable, name)


if __name__ == "__main__":
    raise SystemExit(main())
