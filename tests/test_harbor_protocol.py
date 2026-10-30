from __future__ import annotations

import asyncio
import base64
import os
import sys
from pathlib import Path

import pytest

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.evaluation.harbor.binding import (
    HarborTerminal,
    RefusingTerminal,
    TerminalOptions,
)
from lingcore.evaluation.harbor.listener import Listener
from lingcore.evaluation.harbor.protocol import (
    MAX_FRAME_BYTES,
    Abort,
    BridgeInfo,
    Channel,
    Exec,
    ExecResult,
    Finish,
    Finished,
    Hello,
    Interrupt,
    Probe,
    ProtocolFailure,
    UsageTotals,
    decode,
    encode,
)
from lingcore.evaluation.harbor.tools import ExecArgs, terminal_registry
from lingcore.evaluation.identity import candidate_identity
from lingcore.evaluation.models import Suite
from lingcore.execution import ExecutionBudget
from lingcore.tools import REGISTRY, ToolContext
from tests.fakes import FakeLLMClient

FIXTURES = Path(__file__).resolve().parents[1] / "adapters/harbor-bridge/protocol/v1"


def hello(execution_id="execution"):
    return Hello(
        execution_id=execution_id,
        bridge=BridgeInfo(version="0.1.0", digest="sha256:" + "0" * 64),
        harbor_version="0.25.0",
        task="synthetic",
        instruction="Write a file.",
        probe=Probe(supported=True),
    )


@pytest.mark.parametrize(
    "file", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.stem
)
def test_protocol_golden_frames(file):
    value = file.read_bytes()
    assert encode(decode(value)) == value


@pytest.mark.parametrize(
    "value",
    [
        b"{}\n",
        b'{"type":"unknown"}\n',
        b'{"type":"abort","reason":"cancelled","extra":1}\n',
        b'{"type":"abort","reason":"secret-bad-value"}\n',
        b'{"type":"usage","usage":{"requests":true}}\n',
        b'{"type":"finish","outcome":{"x":NaN},"usage":{},"metadata":{}}\n',
        b"\xff\n",
        b"x" * MAX_FRAME_BYTES + b"\n",
    ],
)
def test_malformed_frames_are_safe_runner_signals(value):
    with pytest.raises(ProtocolFailure) as error:
        decode(value)
    assert "secret-bad-value" not in str(error.value)
    assert not isinstance(error.value, Exception)


async def test_private_listener_accepts_once_and_cleans_up():
    listener = Listener("execution")
    try:
        await listener.start()
        assert len(str(listener.path).encode()) < 104
        assert listener.directory.stat().st_mode & 0o777 == 0o700
        reader, writer = await asyncio.open_unix_connection(listener.path)
        peer = Channel(reader, writer)
        await peer.send(hello())
        core, accepted = await listener.accept(1)
        assert accepted.execution_id == "execution"
        with pytest.raises(OSError):
            await asyncio.open_unix_connection(listener.path)
        await core.send(Abort(reason="cancelled"))
        assert isinstance(await peer.receive(), Abort)
        await peer.aclose()
    finally:
        await listener.aclose()
    assert not listener.directory.exists()


async def test_wrong_execution_id_is_refused():
    listener = Listener("execution")
    try:
        await listener.start()
        reader, writer = await asyncio.open_unix_connection(listener.path)
        peer = Channel(reader, writer)
        await peer.send(hello("foreign"))
        with pytest.raises(ProtocolFailure, match="does not match"):
            await listener.accept(1)
        await peer.aclose()
    finally:
        await listener.aclose()


class FakeChannel:
    def __init__(self):
        self.sent = []
        self.data = b"0123456789"
        self.entered = asyncio.Event()
        self.block = False
        self.queue = asyncio.Queue()
        self.request = None

    async def send(self, frame):
        self.sent.append(frame)
        if isinstance(frame, Exec):
            self.request = frame
            self.entered.set()
            if self.block:
                return
        elif isinstance(frame, Interrupt):
            pass
        elif isinstance(frame, Finish):
            await self.queue.put(Finished(sweep="clean", atif="absent"))
            return
        else:
            return
        request = self.request
        data = self.data[: request.capture_bytes]
        await self.queue.put(
            ExecResult(
                call_id=request.call_id,
                exit_code=0,
                output=base64.b64encode(data).decode(),
                total_output_bytes=len(self.data),
                truncated=len(data) < len(self.data),
                timed_out=False,
                duration=0.1,
            )
        )

    async def receive(self):
        return await self.queue.get()

    async def aclose(self):
        pass


async def test_output_caps_timeout_clamp_eviction_and_foreign_handles(tmp_path):
    channel = FakeChannel()
    terminal = HarborTerminal(
        channel,
        TerminalOptions(max_timeout=5, capture_bytes=8, display_bytes=3),
        store_bytes=8,
    )
    value = await terminal.execute("synthetic", "/app", 1000)
    assert channel.sent[-1].timeout == 5
    assert value["output"] == "012" and value["total_output_bytes"] == 10
    assert value["truncated"] and value["handle"]
    assert terminal.read_output(value["handle"], 3, 1000)["output"] == "345"
    other = HarborTerminal(FakeChannel())
    with pytest.raises(ToolError, match="unknown"):
        other.read_output(value["handle"], 0, 3)
    await terminal.execute("second", None, None)
    with pytest.raises(ToolError, match="evicted"):
        terminal.read_output(value["handle"], 0, 3)
    await terminal.peer.aclose()
    await other.peer.aclose()


async def test_cancellation_interrupts_command_then_can_finish():
    channel = FakeChannel()
    channel.block = True
    terminal = HarborTerminal(channel)
    task = asyncio.create_task(terminal.execute("wait", None, None))
    await channel.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert isinstance(channel.sent[-1], Interrupt)
    reply = await terminal.peer.finish(
        Finish(outcome={}, usage=UsageTotals(), metadata={})
    )
    assert reply.sweep == "clean"
    await terminal.peer.aclose()


async def test_tools_without_binding_never_touch_workspace_or_global_registry(tmp_path):
    before = REGISTRY.names()
    registry = terminal_registry()
    assert REGISTRY.names() == before
    with pytest.raises(ToolError, match="bound Harbor"):
        await registry.get("terminal_exec").run(
            ExecArgs(command="touch forbidden"), ToolContext(workspace=tmp_path)
        )
    assert not list(tmp_path.iterdir())


async def test_offline_identity_assembles_terminal_schemas_without_execution(tmp_path):
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "fake"},
            "tools": ["terminal_exec", "terminal_read_output"],
            "plugins": [],
            "skills": [],
            "sessions": {"enabled": False},
            "persona": {"project_instructions": []},
        }
    )
    suite = Suite.model_validate(
        {
            "schema_version": "lingcore.eval.suite.v1",
            "name": "offline",
            "source": {"cases": ["unused.yaml"]},
            "agent": {"profile": "unused.yaml"},
        }
    )
    binding = RefusingTerminal()
    result = await candidate_identity(
        profile, suite, tool_registry=terminal_registry(), environment_binding=binding
    )
    assert [tool.name for tool in result.tools.ceiling] == [
        "terminal_exec",
        "terminal_read_output",
    ]
    agent = Agent.from_profile(
        profile,
        llm=FakeLLMClient([]),
        tool_registry=terminal_registry(),
        environment_binding=binding,
        state_dir=tmp_path,
    )
    try:
        assert agent.tool_ctx.environment_binding is binding
    finally:
        await agent.aclose()


async def test_terminal_talks_to_independent_stdlib_bridge(monkeypatch):
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret-not-forwarded")
    listener = Listener("execution")
    await listener.start()
    source = FIXTURES.parents[1] / "src"
    code = """
import asyncio, base64, os, sys
sys.path.insert(0, sys.argv[1])
from lingcore_harbor_bridge.protocol import Channel, validate
assert 'FIXTURE_API_KEY' not in os.environ
async def run():
    r,w = await asyncio.open_unix_connection(sys.argv[2], limit=4*1024*1024)
    channel = Channel(r,w)
    await channel.send(validate({'type':'hello','execution_id':'execution','bridge':{'version':'0.1.0','digest':'sha256:'+'0'*64},'harbor_version':'0.25.0','task':'synthetic','instruction':'fixture','probe':{'supported':True}}))
    request = await channel.receive()
    assert request.type == 'exec'
    data = b'fake environment output'
    await channel.send(validate({'type':'exec_result','call_id':request.data['call_id'],'exit_code':0,'output':base64.b64encode(data).decode(),'total_output_bytes':len(data),'truncated':False,'timed_out':False,'duration':0.1}))
    await channel.aclose()
asyncio.run(run())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        str(source),
        str(listener.path),
        env={"PATH": os.environ.get("PATH", "")},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        channel, _ = await listener.accept(5)
        value = await HarborTerminal(channel).execute("synthetic", None, None)
        assert value["output"] == "fake environment output"
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == 0, stderr.decode()
        assert b"fixture-secret-not-forwarded" not in stdout + stderr
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await listener.aclose()


async def test_binding_serializes_concurrent_calls():
    class SlowChannel(FakeChannel):
        active = 0
        maximum = 0

        async def receive(self):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0.01)
                return await super().receive()
            finally:
                self.active -= 1

    channel = SlowChannel()
    terminal = HarborTerminal(channel)
    await asyncio.gather(
        terminal.execute("one", None, None), terminal.execute("two", None, None)
    )
    assert channel.maximum == 1 and len(channel.sent) == 2
    await terminal.peer.aclose()


def test_oversized_trajectory_is_omitted_without_runner_failure():
    frame = Finish(
        outcome={"status": "completed"},
        usage=UsageTotals(),
        metadata={},
        trajectory={"text": "x" * MAX_FRAME_BYTES},
    )
    wire = decode(encode(frame))
    assert wire.trajectory is None and wire.atif_reason == "oversized"
    assert wire.outcome == {"status": "completed"}


async def test_profile_validates_options_without_overwriting_injected_binding(tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentProfile.model_validate(
            {
                "llm": {"model": "fake"},
                "tool_options": {"terminal_exec": {"capture_bytes": 0}},
            }
        )
    profile = AgentProfile.model_validate(
        {
            "llm": {"model": "fake"},
            "tool_options": {"terminal_exec": {"capture_bytes": 17}},
            "tools": [],
            "sessions": {"enabled": False},
        }
    )
    options = TerminalOptions(capture_bytes=23)
    binding = RefusingTerminal(options)
    agent = Agent.from_profile(
        profile,
        llm=FakeLLMClient([]),
        state_dir=tmp_path,
        environment_binding=binding,
        tool_options={"terminal_exec": {"capture_bytes": 23}},
    )
    assert binding.options is options
    assert binding.options.capture_bytes == 23
    assert profile.terminal_options.capture_bytes == 17
    await agent.aclose()


async def test_repeated_cancellation_drains_interrupt_before_forgetting_command():
    interrupted, release = asyncio.Event(), asyncio.Event()

    class DelayedChannel(FakeChannel):
        async def send(self, frame):
            if isinstance(frame, Interrupt):
                interrupted.set()
                await release.wait()
            await super().send(frame)

    channel = DelayedChannel()
    channel.block = True
    terminal = HarborTerminal(channel)
    task = asyncio.create_task(terminal.execute("wait", None, None))
    try:
        await channel.entered.wait()
        task.cancel()
        await interrupted.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and terminal.peer._pending
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert not terminal.peer.signal.done()
        reply = await terminal.peer.finish(
            Finish(outcome={}, usage=UsageTotals(), metadata={})
        )
        assert reply.sweep == "clean" and not terminal.peer._pending
    finally:
        release.set()
        await terminal.peer.aclose()


@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_protocol_failure_returns_full_headless_result(
    tmp_path, cleanup_failure, monkeypatch
):
    from lingcore.execution import run_headless
    from lingcore.message import ToolCall
    from tests.fakes import ScriptedTurn

    class BrokenChannel(FakeChannel):
        async def receive(self):
            await self.entered.wait()
            raise ProtocolFailure("malformed bridge frame")

    binding = HarborTerminal(BrokenChannel())
    profile = AgentProfile.model_validate(
        {
            "tools": ["terminal_exec", "terminal_read_output"],
            "llm": {"model": "fake"},
            "loop": {"parallel_tools": False},
            "sessions": {"enabled": False},
        }
    )
    client = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="call", name="terminal_exec", arguments={"command": "true"}
                    )
                ]
            )
        ]
    )
    agent = Agent.from_profile(
        profile,
        llm=client,
        state_dir=tmp_path,
        tool_registry=terminal_registry(),
        environment_binding=binding,
        execution_budget=ExecutionBudget(),
    )
    if cleanup_failure:
        original_close = agent.aclose

        async def failed_close():
            await original_close()
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(agent, "aclose", failed_close)
    result = await run_headless(agent, ["run"])
    assert result.outcome.status == "failed" and result.outcome.cause == "runner"
    assert len(result.usage) == 1
    assert result.cleanup_failed == cleanup_failure
    await binding.peer.aclose()
