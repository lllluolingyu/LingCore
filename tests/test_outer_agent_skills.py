"""Bundled Codex and Claude Code collaboration skills."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

import lingcore.outer_agents as outer_agents
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.skills import DEFAULT_HIGH_RISK_TOOLS, load_skill_tools, load_skills
from lingcore.tools import REGISTRY, ToolContext
from tests.fakes import FakeLLMClient

REPO_ROOT = Path(__file__).parent.parent


def _load_outer_skills() -> dict[str, Any]:
    skills = load_skills([REPO_ROOT / "lingcore" / "skills"])
    selected = {name: skills[name] for name in ("codex", "claude-code")}
    load_skill_tools(selected)
    return selected


class _FakeStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    def __init__(self, data: bytes) -> None:
        self._chunks = [data, b""]

    async def read(self, size: int) -> bytes:
        return self._chunks.pop(0)


class _FakeProcess:
    def __init__(self, output: bytes = b"external answer\n", code: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(output)
        self.returncode: int | None = code
        self.pid = 12345

    async def wait(self) -> int:
        assert self.returncode is not None
        return self.returncode


def _executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _ctx(
    tmp_path: Path,
    option_key: str,
    executable: Path,
    *,
    confirm: Any = None,
    session_id: str = "a" * 32,
) -> ToolContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir(exist_ok=True)
    return ToolContext(
        workspace=workspace,
        confirm=confirm,
        options={option_key: {"executable": str(executable)}},
        profile_dir=profile_dir,
        session_id=session_id,
    )


def _codex_output(session_id: str, message: str = "codex finding") -> bytes:
    events = [
        {"type": "thread.started", "thread_id": session_id},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": message},
        },
        {"type": "turn.completed"},
    ]
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode()


def test_bundled_outer_skills_declare_their_tools() -> None:
    skills = _load_outer_skills()

    assert skills["codex"].requested_tools == ("codex_agent",)
    assert skills["codex"].provides == ("codex_agent",)
    assert skills["claude-code"].requested_tools == ("claude_code_agent",)
    assert skills["claude-code"].provides == ("claude_code_agent",)
    assert {"codex_agent", "claude_code_agent"} <= DEFAULT_HIGH_RISK_TOOLS


async def test_codex_consult_starts_persistent_read_only_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "codex")
    external_id = str(uuid.uuid4())
    process = _FakeProcess(_codex_output(external_id))
    launched: dict[str, Any] = {}

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launched["argv"] = argv
        launched["kwargs"] = kwargs
        return process

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("codex_agent")
    result = await tool(
        tool.args_model(prompt="Review the parser", mode="consult"),
        _ctx(tmp_path, "codex_agent", executable),
    )

    argv = launched["argv"]
    assert argv[0] == str(executable.resolve())
    assert argv[1] == "exec"
    assert "--ephemeral" not in argv
    assert "--json" in argv
    assert "resume" not in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[-1] == "-"
    assert launched["kwargs"]["cwd"].endswith("workspace")
    assert b"Review the parser" in process.stdin.data
    assert process.stdin.closed is True
    assert "codex finding" in result
    assert "conversation 'default' (started)" in result


async def test_codex_follow_up_resumes_same_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "codex")
    external_id = str(uuid.uuid4())
    launches: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launches.append(argv)
        message = "first" if len(launches) == 1 else "follow-up"
        return _FakeProcess(_codex_output(external_id, message))

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("codex_agent")
    ctx = _ctx(tmp_path, "codex_agent", executable)
    await tool(tool.args_model(prompt="Review it"), ctx)
    result = await tool(tool.args_model(prompt="What about errors?"), ctx)

    assert "resume" in launches[1]
    assert launches[1][launches[1].index("resume") + 1] == external_id
    assert "conversation 'default' (resumed)" in result
    assert "follow-up" in result


async def test_claude_consult_uses_restricted_plan_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")
    process = _FakeProcess()
    launched: dict[str, Any] = {}

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launched["argv"] = argv
        return process

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    await tool(
        tool.args_model(prompt="Explain the failure", mode="consult"),
        _ctx(tmp_path, "claude_code_agent", executable),
    )

    argv = launched["argv"]
    assert "--print" in argv
    assert "--no-session-persistence" not in argv
    assert "--session-id" in argv
    normalize = str(uuid.UUID(argv[argv.index("--session-id") + 1]))
    assert normalize == argv[argv.index("--session-id") + 1]
    assert argv[argv.index("--permission-mode") + 1] == "plan"
    assert "--restricted" in argv


async def test_claude_follow_up_resumes_same_named_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")
    launches: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launches.append(argv)
        return _FakeProcess(b"answer\n")

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    ctx = _ctx(tmp_path, "claude_code_agent", executable)
    await tool(tool.args_model(prompt="Analyze it", conversation="review"), ctx)
    first_id = launches[0][launches[0].index("--session-id") + 1]
    result = await tool(tool.args_model(prompt="Go deeper", conversation="review"), ctx)

    assert "--resume" in launches[1]
    assert launches[1][launches[1].index("--resume") + 1] == first_id
    assert "--session-id" not in launches[1]
    assert "conversation 'review' (resumed)" in result


async def test_failed_restart_preserves_conversation_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")
    launches: list[tuple[str, ...]] = []
    codes = iter((0, 7, 0))

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launches.append(argv)
        return _FakeProcess(b"answer\n", next(codes))

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    ctx = _ctx(tmp_path, "claude_code_agent", executable)
    await tool(tool.args_model(prompt="Start"), ctx)
    original_id = launches[0][launches[0].index("--session-id") + 1]

    with pytest.raises(ToolError, match="exited with code 7"):
        await tool(tool.args_model(prompt="Restart", restart=True), ctx)
    failed_id = launches[1][launches[1].index("--session-id") + 1]
    assert failed_id != original_id

    result = await tool(tool.args_model(prompt="Continue"), ctx)
    assert launches[2][launches[2].index("--resume") + 1] == original_id
    assert "(resumed)" in result


async def test_successful_restart_replaces_conversation_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")
    launches: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launches.append(argv)
        return _FakeProcess()

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    ctx = _ctx(tmp_path, "claude_code_agent", executable)
    await tool(tool.args_model(prompt="Start"), ctx)
    original_id = launches[0][launches[0].index("--session-id") + 1]
    restarted = await tool(tool.args_model(prompt="Start over", restart=True), ctx)
    replacement_id = launches[1][launches[1].index("--session-id") + 1]
    await tool(tool.args_model(prompt="Continue replacement"), ctx)

    assert replacement_id != original_id
    assert "(restarted)" in restarted
    assert launches[2][launches[2].index("--resume") + 1] == replacement_id


async def test_conversations_are_scoped_to_lingcore_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")
    launches: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launches.append(argv)
        return _FakeProcess()

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    first = _ctx(tmp_path, "claude_code_agent", executable, session_id="a" * 32)
    second = _ctx(tmp_path, "claude_code_agent", executable, session_id="b" * 32)
    await tool(tool.args_model(prompt="First chat"), first)
    await tool(tool.args_model(prompt="Second chat"), second)

    assert "--session-id" in launches[0]
    assert "--session-id" in launches[1]
    first_id = launches[0][launches[0].index("--session-id") + 1]
    second_id = launches[1][launches[1].index("--session-id") + 1]
    assert first_id != second_id


async def test_parallel_follow_ups_to_same_alias_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")
    launches: list[tuple[str, ...]] = []
    active = 0
    maximum_active = 0

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        nonlocal active, maximum_active
        launches.append(argv)
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _FakeProcess()

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    ctx = _ctx(tmp_path, "claude_code_agent", executable)
    await asyncio.gather(
        tool(tool.args_model(prompt="First"), ctx),
        tool(tool.args_model(prompt="Second"), ctx),
    )

    assert maximum_active == 1
    assert "--session-id" in launches[0]
    first_id = launches[0][launches[0].index("--session-id") + 1]
    assert launches[1][launches[1].index("--resume") + 1] == first_id


async def test_claude_implementation_requires_fresh_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")
    process = _FakeProcess()
    launched: dict[str, Any] = {}
    confirmations: list[str] = []

    async def confirm(prompt: str) -> bool:
        confirmations.append(prompt)
        return True

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        launched["argv"] = argv
        return process

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    await tool(
        tool.args_model(prompt="Fix the test", mode="implement"),
        _ctx(tmp_path, "claude_code_agent", executable, confirm=confirm),
    )

    assert confirmations and "modify files" in confirmations[0]
    argv = launched["argv"]
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert "--restricted" not in argv


async def test_implementation_refuses_frontend_without_confirmation(
    tmp_path: Path,
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "codex")
    tool = REGISTRY.get("codex_agent")

    with pytest.raises(ToolError, match="requires confirmation"):
        await tool(
            tool.args_model(prompt="Change it", mode="implement"),
            _ctx(tmp_path, "codex_agent", executable),
        )


def test_coding_profile_gates_outer_tools_behind_skills(tmp_path: Path) -> None:
    profile = AgentProfile.load(REPO_ROOT / "profiles" / "coding")
    profile.workspace = str(tmp_path / "workspace")
    agent = Agent.from_profile(profile, llm=FakeLLMClient([]))

    assert agent.skill_state is not None
    assert {"codex", "claude-code"} <= set(agent.skill_state.skills)
    assert "activate_skill" in agent.initial_tools
    assert "codex_agent" not in agent.initial_tools
    assert "claude_code_agent" not in agent.initial_tools
    assert {"codex_agent", "claude_code_agent"} <= set(agent.tools.names())
