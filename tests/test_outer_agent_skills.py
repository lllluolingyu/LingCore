"""Bundled Codex and Claude Code collaboration skills."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

import lingcore.outer_agents as outer_agents
import lingcore.tools.builtin  # noqa: F401  (registration side effect)
from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.doctor import diagnose_profile
from lingcore.errors import ConfigError, ToolError
from lingcore.skills import DEFAULT_HIGH_RISK_TOOLS
from lingcore.tools import REGISTRY, ToolContext
from tests.fakes import FakeLLMClient

REPO_ROOT = Path(__file__).parent.parent


def _load_outer_skills() -> dict[str, Any]:
    from lingcore.plugins.discovery import discover_plugins, plugin_skills
    from lingcore.skills import _load_tool_module

    selected = {}
    for name in ("codex", "claude-code"):
        plugin = discover_plugins(None)[name]
        selected.update(plugin_skills(plugin))
        _load_tool_module(
            name,
            plugin.root / plugin.manifest.module,
            plugin.manifest.provides,
            prefix=plugin.manifest.prefix,
        )
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
    session_id: str | None = "a" * 32,
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
    assert skills["codex"].provides == ()
    assert skills["claude-code"].requested_tools == ("claude_code_agent",)
    assert skills["claude-code"].provides == ()
    # Risk is declared on the tool itself; the core baseline stays builtin-only.
    assert REGISTRY.get("codex_agent").high_risk is True
    assert REGISTRY.get("claude_code_agent").high_risk is True
    assert REGISTRY.get("read_file").high_risk is False
    assert not ({"codex_agent", "claude_code_agent"} & DEFAULT_HIGH_RISK_TOOLS)


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
    reviewer_override = argv.index("-c")
    assert argv[reviewer_override + 1] == 'approvals_reviewer="user"'
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
    reviewer_override = launches[1].index("-c")
    assert launches[1][reviewer_override + 1] == 'approvals_reviewer="user"'
    assert reviewer_override < launches[1].index("resume")
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
    assert "--safe-mode" in argv
    assert "--restricted" in argv
    assert "--strict-mcp-config" in argv


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
    assert {"--safe-mode", "--restricted", "--strict-mcp-config"} <= set(launches[1])
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
    assert "--safe-mode" in argv
    assert "--restricted" in argv
    assert "--strict-mcp-config" in argv


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
    # skill_gated_tools hides exactly the two outer-agent tools; every other
    # ceiling tool stays initially enabled without being re-listed.
    assert agent.initial_tools == frozenset(profile.tools) - {
        "codex_agent",
        "claude_code_agent",
    }
    assert {"codex_agent", "claude_code_agent"} <= set(agent.tools.names())
    # Their self-declared risk reaches the activation gate alongside the
    # builtin baseline, without skills.py naming them.
    assert {"codex_agent", "claude_code_agent", "run_shell"} <= (
        agent.skill_state.high_risk_tools
    )
    # The ToolContext is built whole with the session id (no post-hoc patching).
    assert agent.tool_ctx.session_id == agent._session_id


def test_declared_risk_is_a_floor_under_an_operator_override(tmp_path: Path) -> None:
    profile = AgentProfile.load(REPO_ROOT / "profiles" / "coding")
    profile.workspace = str(tmp_path / "workspace")
    profile.tool_options["activate_skill"] = {
        **profile.tool_options.get("activate_skill", {}),
        "high_risk_tools": [],
    }
    agent = Agent.from_profile(profile, llm=FakeLLMClient([]))

    assert agent.skill_state is not None
    assert "run_shell" not in agent.skill_state.high_risk_tools
    assert {"codex_agent", "claude_code_agent"} <= agent.skill_state.high_risk_tools


def _gated_profile(tmp_path: Path, body: str) -> Path:
    root = tmp_path / "gated-profile"
    root.mkdir()
    (root / "config.yaml").write_text(body, encoding="utf-8")
    return root


def test_skill_gated_tools_must_be_in_the_ceiling(tmp_path: Path) -> None:
    root = _gated_profile(
        tmp_path,
        "name: gated\nllm:\n  model: m\n  api_key_env: ''\n"
        "tools: [read_file]\nskill_gated_tools: [codex_agent]\n",
    )
    with pytest.raises(ConfigError, match="skill_gated_tools .* not listed in tools"):
        AgentProfile.load(root)


def test_skill_gated_tools_and_initial_tools_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    root = _gated_profile(
        tmp_path,
        "name: gated\nllm:\n  model: m\n  api_key_env: ''\n"
        "tools: [read_file, list_dir]\ninitial_tools: [read_file]\n"
        "skill_gated_tools: [list_dir]\n",
    )
    with pytest.raises(ConfigError, match="declare one or the other"):
        AgentProfile.load(root)


def test_initial_tool_set_resolves_either_form() -> None:
    llm = {"model": "m", "api_key_env": ""}
    both_unset = AgentProfile(llm=llm, tools=["a", "b"])
    inclusion = AgentProfile(llm=llm, tools=["a", "b"], initial_tools=["a"])
    exclusion = AgentProfile(llm=llm, tools=["a", "b"], skill_gated_tools=["b"])

    assert both_unset.initial_tool_set() == {"a", "b"}
    assert inclusion.initial_tool_set() == {"a"}
    assert exclusion.initial_tool_set() == {"a"}


def test_configured_executable_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    executable = _executable(tmp_path, "codex")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    resolved = outer_agents.resolve_outer_agent_executable(
        outer_agents.CODEX, "~/bin/codex", workspace
    )

    assert resolved == executable.resolve()


@pytest.mark.parametrize("configured", ["", "   "])
def test_blank_executable_is_unset(configured: str) -> None:
    options = outer_agents.parse_outer_agent_options({"executable": configured})

    assert options.executable is None


def test_executable_inside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = _executable(workspace, "codex")

    with pytest.raises(ToolError, match="inside the writable workspace"):
        outer_agents.resolve_outer_agent_executable(
            outer_agents.CODEX, str(inside), workspace
        )


async def test_restart_of_unknown_alias_reports_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load_outer_skills()
    executable = _executable(tmp_path, "claude")

    async def fake_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess()

    monkeypatch.setattr(outer_agents.asyncio, "create_subprocess_exec", fake_exec)
    tool = REGISTRY.get("claude_code_agent")
    result = await tool(
        tool.args_model(prompt="Begin", restart=True),
        _ctx(tmp_path, "claude_code_agent", executable),
    )

    assert "conversation 'default' (started)" in result


async def test_sessionless_contexts_share_one_workspace_scope(
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
    first = _ctx(tmp_path, "claude_code_agent", executable, session_id=None)
    second = _ctx(tmp_path, "claude_code_agent", executable, session_id=None)
    await tool(tool.args_model(prompt="First run"), first)
    await tool(tool.args_model(prompt="Second run"), second)

    minted = launches[0][launches[0].index("--session-id") + 1]
    assert launches[1][launches[1].index("--resume") + 1] == minted


def test_doctor_validates_outer_agent_options_and_locates_clis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codex = _executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    root = _gated_profile(
        tmp_path,
        "name: gated\nllm:\n  model: m\n  api_key_env: ''\n"
        "tools: [activate_skill, codex_agent, claude_code_agent]\n"
        "skill_gated_tools: [codex_agent, claude_code_agent]\n"
        "tool_options:\n  codex_agent:\n    executable: ''\n"
        "  claude_code_agent:\n    timout: 5\n",
    )
    report = diagnose_profile(AgentProfile.load(root))
    messages = {finding.level: [] for finding in report.findings}
    for finding in report.findings:
        messages[finding.level].append(finding.message)

    assert any(
        m == f"Codex agent executable: {codex.resolve()}" for m in messages["ok"]
    )
    assert any("tool_options.claude_code_agent" in m for m in messages["error"])
    assert not (root / "workspace").exists()

    # A valid but uninstalled CLI is only a warning: the skill is optional.
    (root / "config.yaml").write_text(
        "name: gated\nllm:\n  model: m\n  api_key_env: ''\n"
        "tools: [activate_skill, claude_code_agent]\n",
        encoding="utf-8",
    )
    report = diagnose_profile(AgentProfile.load(root))
    assert not report.errors
    assert any("claude CLI is not installed" in w.message for w in report.warnings)
