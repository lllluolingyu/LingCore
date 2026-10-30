"""Bound capture and deadlines inside the task container, before Docker reads."""

from __future__ import annotations

import asyncio
import base64
import shlex
import time
from typing import Any
from uuid import uuid4

from harbor.environments.base import BaseEnvironment

from lingcore_harbor_bridge.protocol import BridgeProtocolError

# /proc stat's comm may contain spaces/parentheses: strip through the last ')'.
_STAT = 'read -r stat < "$p/stat" 2>/dev/null || continue; rest=${stat##*) }; read -ra fields <<< "$rest";'


class Supervisor:
    def __init__(self, environment: BaseEnvironment):
        self.environment = environment
        self.root = "/tmp/lingcore-" + uuid4().hex
        self.baseline: set[str] = set()
        self.active: str | None = None
        self.supported = False
        self.command_started = 0.0

    async def probe(self) -> list[str]:
        result = await self.environment.exec(
            command="test -d /proc || echo /proc; for c in bash setsid kill head cat tee wc base64 mkfifo mkdir rm sleep touch; do command -v $c >/dev/null || echo $c; done",
            timeout_sec=30,
        )
        missing = (result.stdout or "").split()
        if result.return_code and not missing:
            missing = ["probe_failed"]
        self.supported = not missing
        if self.supported:
            result = await self.environment.exec(
                command=f'for p in /proc/[0-9]*; do {_STAT} echo "${{p##*/}}:${{fields[19]}}"; done',
                timeout_sec=30,
            )
            if result.return_code:
                raise BridgeProtocolError("baseline capture failed")
            self.baseline = set((result.stdout or "").split())
            if not self.baseline:
                raise BridgeProtocolError("empty process baseline")
        return missing

    async def interrupt(self) -> None:
        if self.active is None:
            return
        path = shlex.quote(self.active)
        # The wrapper writes its group before the command starts. Wait briefly
        # for a concurrent exec to publish it, never kill a guessed host PID.
        result = await self.environment.exec(
            command=f"""
for ((i=0;i<100;i++)); do
  if test -f {path}/pid; then read -r pid < {path}/pid; kill -KILL -- -"$pid" 2>/dev/null || true; exit 0; fi
  test -f {path}/done && exit 0
  sleep .02
done
exit 1
""",
            timeout_sec=10,
        )
        if result.return_code:
            raise BridgeProtocolError("command interruption could not be confirmed")

    async def stop_command(
        self, task: asyncio.Task[dict[str, Any]], request: dict[str, Any]
    ) -> dict[str, Any]:
        path = self.active
        await self.interrupt()
        try:
            async with asyncio.timeout(5):
                return await asyncio.shield(task)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if path is None:
                raise BridgeProtocolError("command interruption lost its capture")
            partial = await self.environment.exec(
                command=f"head -c {request['capture_bytes']} {path}/capture | base64",
                timeout_sec=10,
            )
            captured = base64.b64decode(partial.stdout or "", validate=False)
            return dict(
                type="exec_result",
                call_id=request["call_id"],
                exit_code=137,
                output=base64.b64encode(captured[: request["capture_bytes"]]).decode(),
                total_output_bytes=None,
                truncated=True,
                timed_out=False,
                duration=time.monotonic() - self.command_started,
            )

    async def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        path = self.root + "/" + uuid4().hex
        self.active = path
        command = request["command"]
        if request["cwd"] is not None:
            command = "cd -- " + shlex.quote(request["cwd"]) + " && " + command
        # The group leader publishes its PID before running untrusted code.
        child = (
            f"echo $$ > {shlex.quote(path)}/pid; exec bash -c {shlex.quote(command)}"
        )
        script = f"""
mkdir -p {path} || exit 1
mkfifo {path}/pipe {path}/capture-pipe || exit 1
# Count the entire stream independently. Buffered head implementations can
# consume more than they emit; the extra read must not disappear from totals.
(head -c {request["capture_bytes"]} > {path}/capture; cat >/dev/null) < {path}/capture-pipe &
capture=$!
(tee {path}/capture-pipe | wc -c > {path}/total) < {path}/pipe &
drain=$!
setsid bash -c {shlex.quote(child)} > {path}/pipe 2>&1 &
pid=$!
setsid bash -c 'sleep "$1"; echo yes > "$2/timeout"; kill -KILL -- -"$3" 2>/dev/null || true' -- {request["timeout"]} {path} "$pid" &
watch=$!
wait "$pid" 2>/dev/null
rc=$?
# Keep redirected background services alive between calls. If a child keeps
# this command's pipe open, its deadline still bounds the drain.
wait "$drain"
wait "$capture"
kill -KILL -- -"$watch" 2>/dev/null || kill -KILL "$watch" 2>/dev/null || true
wait "$watch" 2>/dev/null || true
read -r total < {path}/total
flag=0; test -f {path}/timeout && flag=1
printf '%s %s %s\\n' "$rc" "$total" "$flag"
base64 < {path}/capture
touch {path}/done
"""
        started = time.monotonic()
        self.command_started = started
        try:
            result = await self.environment.exec(
                command=script, timeout_sec=request["timeout"] + 30
            )
            if result.return_code:
                raise BridgeProtocolError("command supervisor failed")
            header, output = (result.stdout or "").split("\n", 1)
            rc, count, timed_out = map(int, header.split())
            captured = base64.b64decode(output, validate=False)
            if len(captured) > request["capture_bytes"] or count < len(captured):
                raise BridgeProtocolError("invalid supervised output")
            return dict(
                type="exec_result",
                call_id=request["call_id"],
                exit_code=rc,
                output=base64.b64encode(captured).decode(),
                total_output_bytes=count,
                truncated=count > len(captured),
                timed_out=bool(timed_out),
                duration=time.monotonic() - started,
            )
        except TimeoutError:
            await self.interrupt()
            # A detached child can keep the output pipe open after its original
            # group dies. Preserve a bounded prefix and mark the count unknown;
            # the final sweep removes all remaining processes before grading.
            partial = await self.environment.exec(
                command=f"head -c {request['capture_bytes']} {path}/capture | base64",
                timeout_sec=10,
            )
            captured = base64.b64decode(partial.stdout or "", validate=False)
            return dict(
                type="exec_result",
                call_id=request["call_id"],
                exit_code=124,
                output=base64.b64encode(captured[: request["capture_bytes"]]).decode(),
                total_output_bytes=None,
                truncated=True,
                timed_out=True,
                duration=time.monotonic() - started,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self.interrupt())
            raise
        finally:
            self.active = None

    async def sweep(self) -> bool:
        if not self.supported:
            return True
        baseline = shlex.quote(" " + " ".join(sorted(self.baseline)) + " ")
        # Exclude this cleanup exec and its ancestors. Compare start times as
        # well as PIDs so PID reuse cannot hide an agent process. Zombies cannot
        # change artifacts and only the container init can reap them.
        script = f"""
baseline={baseline}
ancestors=" "
pid=$$
while test "$pid" -gt 0 && test -r /proc/$pid/stat; do
 ancestors="$ancestors$pid "
 read -r stat < /proc/$pid/stat; rest=${{stat##*) }}; read -ra fields <<< "$rest"; pid=${{fields[1]}}
done
for ((i=0;i<50;i++)); do
 remaining=0
 for p in /proc/[0-9]*; do
  {_STAT}
  pid=${{p##*/}}; token="$pid:${{fields[19]}}"
  [[ "$ancestors" == *" $pid "* || "$baseline" == *" $token "* || "${{fields[0]}}" == Z ]] && continue
  remaining=1; kill -KILL "$pid" 2>/dev/null || true
 done
 if test "$remaining" = 0; then rm -rf -- {self.root}; exit $?; fi
 sleep .1
done
exit 1
"""
        try:
            result = await self.environment.exec(command=script, timeout_sec=15)
            return bool(result.return_code == 0)
        except Exception:
            return False
