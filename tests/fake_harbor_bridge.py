#!/usr/bin/env python3
"""Synthetic protocol peer, never imports LingCore/Harbor or executes commands."""

import asyncio
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

SECRET = "fixture-secret-never-forwarded"
assert SECRET not in str(dict(os.environ))
args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
kind = sys.argv[1]


def digest(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def describe():
    root = Path(args["--dataset"])
    tasks = []
    for path in sorted(root.iterdir()):
        task = json.loads((path / "description.json").read_text())
        task.update(
            id=path.name,
            path=path.name,
            name="synthetic/" + path.name,
            instruction_digest=digest((path / "instruction.md").read_bytes()),
            harbor_digest=digest(
                b"".join(p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file())
            ),
        )
        tasks.append(task)
    return dict(
        protocol="lingcore.harbor-bridge.v1",
        bridge={"version": "0.1.0", "digest": "sha256:" + "a" * 64},
        harbor={"version": "0.25.0"},
        python={"version": "fake"},
        error_policy={"digest": "sha256:" + "b" * 64},
        atif_versions=["ATIF-v1.7"],
        tasks=tasks,
    )


async def trial():
    task = Path(args["--task"])
    scenario = os.environ.get("FIXTURE_SCENARIO", "")
    output = Path(args["--output"])
    trial_dir = output / "job" / "trial"
    trial_dir.mkdir(parents=True)
    import fcntl

    lock = (output.parent / "harbor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if scenario == "before_hello":
        return
    if (
        scenario == "environment_once"
        and len(list(output.parent.parent.glob("*/trial.json"))) == 1
    ):
        summary = dict(
            failure_cause="sandbox_failed",
            exceptions=[],
            rewards={},
            reward_valid=False,
            task_digest=digest(
                b"".join(p.read_bytes() for p in sorted(task.rglob("*")) if p.is_file())
            ),
        )
        (output / "summary.json").write_text(json.dumps(summary))
        return
    r, w = await asyncio.open_unix_connection(args["--socket"], limit=4 * 1024 * 1024)
    frames = []

    async def send(value):
        data = json.dumps(value).encode() + b"\n"
        assert SECRET.encode() not in data
        w.write(data)
        await w.drain()

    await send(
        dict(
            type="hello",
            execution_id=args["--execution-id"],
            bridge={"version": "0.1.0", "digest": "sha256:" + "a" * 64},
            harbor_version="0.25.0",
            task="synthetic/" + task.name,
            instruction=(task / "instruction.md").read_text(),
            probe={"supported": scenario != "unsupported"},
        )
    )
    pending = None
    sweep = "clean"
    atif = "absent"
    atif_reason = None
    outcome = {}
    reward = 0.0
    if scenario == "abort_idle":
        await asyncio.sleep(0.05)
        await send(dict(type="abort", reason="harness_timeout"))
    else:
        while line := await r.readline():
            assert SECRET.encode() not in line
            frame = json.loads(line)
            frames.append(frame)
            if frame["type"] == "exec":
                if scenario == "malformed":
                    w.write(b'{"type":"nonsense"}\n')
                    await w.drain()
                    break
                pending = frame
                if scenario == "wait_interrupt":
                    continue
                reward = 0.0 if frame["command"] == "fail" else 1.0
            elif frame["type"] == "interrupt":
                assert pending and pending["call_id"] == frame["call_id"]
            elif frame["type"] == "finish":
                outcome = frame["outcome"]
                atif_reason = frame.get("atif_reason")
                if scenario in {"timeout_finish", "timeout_finish_eof"}:
                    if scenario == "timeout_finish":
                        await send(dict(type="abort", reason="harness_timeout"))
                    break
                if frame.get("trajectory") is not None:
                    atif = "written"
                await send(dict(type="finished", sweep=sweep, atif=atif))
                break
            elif frame["type"] == "abort":
                break
            else:
                continue
            raw = b"ok"
            await send(
                dict(
                    type="exec_result",
                    call_id=pending["call_id"],
                    exit_code=0,
                    output=base64.b64encode(raw).decode(),
                    total_output_bytes=2,
                    truncated=False,
                    timed_out=False,
                    duration=0.01,
                )
            )
            pending = None
    summary = dict(
        trial="job/trial",
        rewards={"reward": reward},
        reward_valid=True,
        failure_cause=None,
        exceptions=[],
        timing={},
        task_digest=digest(
            b"".join(p.read_bytes() for p in sorted(task.rglob("*")) if p.is_file())
        ),
        metadata={
            "sweep": sweep,
            "atif": atif,
            "atif_reason": atif_reason,
            "outcome": outcome,
        },
    )
    if scenario in {"abort_idle", "timeout_finish", "timeout_finish_eof"}:
        summary["exceptions"] = [{"type": "AgentTimeoutError"}]
    if scenario == "digest_mismatch":
        summary["task_digest"] = "sha256:" + "f" * 64
    (trial_dir / "result.json").write_text(json.dumps(summary))
    (trial_dir / "frames.json").write_text(json.dumps(frames))
    (output / "summary.json").write_text(json.dumps(summary))
    w.close()
    await w.wait_closed()


if kind == "describe":
    print(json.dumps(describe()))
elif kind == "trial":
    asyncio.run(trial())
elif kind == "summarize":
    print((Path(args["--output"]) / "summary.json").read_text())
elif kind == "regrade":
    source = Path(args["--source"])
    summary = json.loads((source / "result.json").read_text())
    summary["rewards"] = {"reward": 1.0}
    output = Path(args["--output"])
    output.mkdir(parents=True)
    (output / "environment.json").write_text(
        json.dumps({k: v for k, v in os.environ.items() if k.startswith("FIXTURE_")})
    )
    print(json.dumps(summary))
