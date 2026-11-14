"""Host ownership and executor process/session lifecycle helpers."""

from __future__ import annotations

import asyncio
import fcntl
import os
import time
from pathlib import Path
from typing import Any

from lingcore.errors import ConfigError
from lingcore.paths import confined_directory


class ServiceLock:
    def __init__(self, state: Path, name: str) -> None:
        with confined_directory(state, ".", create=True) as directory:
            self.fd = os.open(
                name + ".lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory._fd,
            )
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(self.fd)
                raise ConfigError(
                    "a superagent service already owns this state store"
                ) from None

    def close(self) -> None:
        os.close(self.fd)


async def wait_for_lease(store: Any, session_id: str | None) -> None:
    if not store or not session_id:
        return
    while True:
        with store._lock:
            row = store._conn.execute(
                "SELECT owner_token, heartbeat FROM session_leases WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None or row[0] is None or row[1] + 60 <= time.time():
            return
        await asyncio.sleep(min(1, row[1] + 60 - time.time()))


async def hold_processes(
    runtime: Any, child: Any, ticket_id: str, assignment_id: str
) -> None:
    hooks = child.tool_ctx.plugins.get("process")
    if hooks is None or not hasattr(hooks, "processes"):
        return
    deadline = time.monotonic() + runtime.config.executor.background_limit_seconds
    while any(p.state == "running" for p in hooks.processes()):
        if time.monotonic() >= deadline:
            runtime.final_reports[assignment_id] = {
                "status": "waiting",
                "summary": "Background process holding period expired.",
                "question": "The background process reached its holding limit. Should work continue?",
            }
            return
        await asyncio.sleep(0.25)
