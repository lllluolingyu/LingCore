from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext

from lingcore_harbor_bridge.agent import BridgeAgent
from lingcore_harbor_bridge.protocol import Channel, validate


class Environment:
    def __init__(self, missing=False):
        self.commands = []
        self.missing = missing

    async def exec(self, command, timeout_sec):
        self.commands.append(command)
        if "command -v" in command:
            output = "setsid\n" if self.missing else ""
        elif 'echo "${p##*/}' in command:
            output = "1:42\n"
        elif "mkfifo" in command:
            output = "0 2 0\n" + base64.b64encode(b"ok").decode() + "\n"
        else:
            output = ""
        return SimpleNamespace(stdout=output, return_code=0)


async def session(tmp_path, exchange, missing=False):
    socket = tmp_path / "s"
    done = asyncio.get_running_loop().create_future()

    async def connected(reader, writer):
        channel = Channel(reader, writer)
        try:
            await exchange(channel)
            done.set_result(None)
        except BaseException as error:
            done.set_exception(error)
        finally:
            await channel.aclose()

    server = await asyncio.start_unix_server(connected, path=socket)
    agent = BridgeAgent(
        logs_dir=tmp_path / "logs",
        socket_path=str(socket),
        execution_id="execution",
        task_name="tiny",
    )
    env = Environment(missing)
    context = AgentContext()
    try:
        await agent.run("instruction", env, context)
        await done
    finally:
        server.close()
        await server.wait_closed()
    return env, context


async def test_relay_usage_and_sweep_before_finished(tmp_path):
    async def exchange(channel):
        assert (await channel.receive()).type == "hello"
        await channel.send(
            validate(
                {
                    "type": "usage",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "coverage": "complete",
                        "per_model": {"model": {"input_tokens": 12}},
                    },
                }
            )
        )
        await channel.send(
            validate(
                dict(
                    type="exec",
                    call_id="call",
                    command="true",
                    timeout=1,
                    capture_bytes=10,
                )
            )
        )
        assert (await channel.receive()).data["output"] == base64.b64encode(
            b"ok"
        ).decode()
        await channel.send(
            validate(
                dict(
                    type="finish",
                    outcome={"status": "completed"},
                    usage={
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "coverage": "complete",
                    },
                    metadata={"trial_id": "trial"},
                )
            )
        )
        assert (await channel.receive()).data == dict(
            type="finished", sweep="clean", atif="absent"
        )

    env, context = await session(tmp_path, exchange)
    assert context.n_input_tokens == 12 and context.cost_usd is None
    assert context.metadata["lingcore"]["trial_id"] == "trial"
    assert "remaining=0" in env.commands[-1]


@pytest.mark.parametrize("mode", ["eof", "abort", "malformed"])
async def test_every_disconnect_sweeps(tmp_path, mode):
    async def exchange(channel):
        await channel.receive()
        if mode == "abort":
            await channel.send(validate(dict(type="abort", reason="cancelled")))
        elif mode == "malformed":
            channel.writer.write(b'{"type":"invalid"}\n')
            await channel.writer.drain()

    env, context = await session(tmp_path, exchange)
    assert "remaining=0" in env.commands[-1]
    assert context.metadata["lingcore"]["sweep"] == "clean"


async def test_missing_prerequisite_is_reported_before_exec(tmp_path):
    async def exchange(channel):
        hello = await channel.receive()
        assert hello.data["probe"] == {"supported": False, "missing": ["setsid"]}
        await channel.send(validate(dict(type="abort", reason="protocol_error")))

    env, _ = await session(tmp_path, exchange, missing=True)
    assert len(env.commands) == 1


@pytest.mark.skipif(
    __import__("os").environ.get("LINGCORE_HARBOR_E2E") != "1",
    reason="opt-in Docker supervision",
)
async def test_real_container_capture_timeout_interrupt_and_detached_sweep():
    from uuid import uuid4

    from test_contract import IMAGE

    from lingcore_harbor_bridge.supervision import Supervisor

    name = "lingcore-supervision-" + uuid4().hex[:10]

    async def docker(*args):
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return proc.returncode, out.decode()

    class DockerEnvironment:
        async def exec(self, command, timeout_sec):
            async with asyncio.timeout(timeout_sec):
                code, output = await docker("exec", name, "bash", "-c", command)
                return SimpleNamespace(return_code=code, stdout=output)

    code, output = await docker("run", "-d", "--name", name, IMAGE, "sleep", "infinity")
    assert code == 0, output
    try:
        env = DockerEnvironment()
        supervisor = Supervisor(env)
        assert await supervisor.probe() == []
        request = dict(
            type="exec",
            call_id="call",
            command="yes",
            cwd=None,
            timeout=0.3,
            capture_bytes=128,
        )
        result = await supervisor.execute(request)
        assert result["timed_out"] and result["truncated"]
        assert (
            len(base64.b64decode(result["output"])) == 128
            and result["total_output_bytes"] > 128
        )
        task = asyncio.create_task(
            supervisor.execute(request | {"command": "sleep 1000", "timeout": 1000})
        )
        await asyncio.sleep(0.2)
        await supervisor.interrupt()
        result = await asyncio.wait_for(task, 5)
        assert result["exit_code"] != 0
        background = await supervisor.execute(
            request | {"command": "sleep 1000 >/dev/null 2>&1 & echo $!", "timeout": 2}
        )
        background_pid = base64.b64decode(background["output"]).decode().strip()
        assert (await env.exec("kill -0 " + background_pid, 5)).return_code == 0
        result = await supervisor.execute(
            request
            | {"command": "setsid sleep 1000 >/dev/null 2>&1 & echo $!", "timeout": 2}
        )
        pid = base64.b64decode(result["output"]).decode().strip()
        assert (await env.exec("kill -0 " + pid, 5)).return_code == 0
        assert await supervisor.sweep()
        check = await env.exec(
            f'test ! -d /proc/{pid} || test "$(cut -d " " -f 3 /proc/{pid}/stat)" = Z',
            5,
        )
        assert check.return_code == 0
        check = await env.exec(
            f'test ! -d /proc/{background_pid} || test "$(cut -d " " -f 3 /proc/{background_pid}/stat)" = Z',
            5,
        )
        assert check.return_code == 0
        assert (
            await env.exec("mv /usr/bin/setsid /usr/bin/setsid.disabled", 5)
        ).return_code == 0
        assert await Supervisor(env).probe() == ["setsid"]
    finally:
        await docker("rm", "-f", name)


@pytest.mark.skipif(
    __import__("os").environ.get("LINGCORE_HARBOR_E2E") != "1",
    reason="opt-in Docker harness timeout",
)
@pytest.mark.parametrize("during_finish", [False, True])
async def test_harbor_agent_timeout_keeps_usage_and_valid_reward(
    tmp_path, during_finish
):
    from harbor.models.trial.result import TrialResult
    from test_contract import command, environment, task

    root = task(tmp_path)
    config = root / "task.toml"
    config.write_text(
        config.read_text().replace("timeout_sec = 180", "timeout_sec = 5")
    )
    socket = tmp_path / "s"
    frames = []

    async def connected(reader, writer):
        channel = Channel(reader, writer)
        try:
            frames.append((await channel.receive()).data)
            await channel.send(
                validate(
                    {
                        "type": "usage",
                        "usage": {"input_tokens": 12, "coverage": "complete"},
                    }
                )
            )
            if during_finish:
                await channel.send(
                    validate(
                        dict(
                            type="finish",
                            outcome={"status": "completed"},
                            usage={"input_tokens": 12, "coverage": "complete"},
                            metadata={},
                        )
                    )
                )
            frames.append((await channel.receive()).data)
        finally:
            await channel.aclose()

    server = await asyncio.start_unix_server(connected, path=socket)
    process = await asyncio.create_subprocess_exec(
        *command(
            [
                "run",
                "-p",
                str(root),
                "-a",
                "probe_agent:FinishingTimeoutAgent"
                if during_finish
                else "lingcore_harbor_bridge.agent:BridgeAgent",
                "-k",
                "1",
                "-n",
                "1",
                "--max-retries",
                "0",
                "--agent-kwarg",
                "socket_path=" + str(socket),
                "--agent-kwarg",
                "execution_id=timeout",
                "--agent-kwarg",
                "task_name=synthetic",
                "--jobs-dir",
                str(tmp_path / "output"),
                "--job-name",
                "timeout",
                "--quiet",
            ]
        ),
        env=environment(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async with asyncio.timeout(90):
            output, _ = await process.communicate()
        assert process.returncode == 0, output.decode()[-4000:]
        result = TrialResult.model_validate_json(
            next((tmp_path / "output").glob("*/*/result.json")).read_bytes()
        )
        assert frames[-1] == {"type": "abort", "reason": "harness_timeout"}
        assert result.exception_info.exception_type == "AgentTimeoutError"
        assert result.reward_valid and result.verifier_result.rewards == {"reward": 0.0}
        assert result.agent_result.n_input_tokens == 12
        assert result.agent_result.metadata["lingcore"]["sweep"] == "clean"
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        server.close()
        await server.wait_closed()


async def test_harness_cancellation_during_finish_joins_sweep(tmp_path, monkeypatch):
    from lingcore_harbor_bridge.supervision import Supervisor

    entered, release, swept = asyncio.Event(), asyncio.Event(), asyncio.Event()
    replies = []

    async def sweep(self):
        entered.set()
        await release.wait()
        swept.set()
        return True

    async def exchange(channel):
        await channel.receive()
        await channel.send(
            validate(
                dict(
                    type="finish",
                    outcome={"status": "completed"},
                    usage={"input_tokens": 7},
                    metadata={},
                )
            )
        )
        replies.append((await channel.receive()).data)
        assert swept.is_set()

    monkeypatch.setattr(Supervisor, "sweep", sweep)
    task = asyncio.create_task(session(tmp_path, exchange))
    async with asyncio.timeout(5):
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not swept.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert swept.is_set()
    assert replies == [{"type": "abort", "reason": "harness_timeout"}]


@pytest.mark.parametrize("cap", [1, 128, 4097])
async def test_capture_counts_bytes_consumed_by_buffered_head(tmp_path, cap):
    import os
    import sys

    from lingcore_harbor_bridge.supervision import Supervisor

    shim = tmp_path / "bin"
    shim.mkdir()
    head = shim / "head"
    # Like BusyBox's getc-based implementation, this reads ahead internally.
    head.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stdout.buffer.write(sys.stdin.buffer.read(int(sys.argv[2])))\n"
    )
    head.chmod(0o700)

    class LocalEnvironment:
        async def exec(self, command, timeout_sec):
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                env=os.environ | {"PATH": str(shim) + ":" + os.environ["PATH"]},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(timeout_sec):
                output, error = await process.communicate()
            assert process.returncode == 0, error.decode()
            return SimpleNamespace(stdout=output.decode(), return_code=0)

    supervisor = Supervisor(LocalEnvironment())
    supervisor.root = str(tmp_path / "capture")
    result = await supervisor.execute(
        dict(
            command="printf '%10000s' x",
            cwd=None,
            timeout=5,
            capture_bytes=cap,
            call_id="buffered",
        )
    )
    assert result["total_output_bytes"] == 10000
    assert base64.b64decode(result["output"]) == b" " * cap
    assert result["truncated"] and not result["timed_out"]
