"""Environment initialization, durable snapshots, and profile confinement."""

from __future__ import annotations

import os

import pytest

from lingcore.agent import Agent
from lingcore.composer import ComposeContext
from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.message import ToolCall
from lingcore.profiles import initialize_profile
from lingcore.tools.builtin.world import WorldArgs, world
from lingcore.world import SNAPSHOT_MAX_BYTES, WORLD_MAX_BYTES, WorldFile
from tests.fakes import FakeLLMClient, ScriptedTurn


async def test_snapshot_refresh_preserves_authored_notes(tmp_path):
    path = tmp_path / "world.md"
    authored = "# Environment\nKeep these instructions exactly.\n"
    path.write_text(authored)
    state = WorldFile(tmp_path)
    await state.update("Python 3.13; 8 CPUs", tmp_path / "workspace")
    path.write_text(path.read_text() + "\n# Personal notes\nKeep this too.\n")
    await state.update("Python 3.14; 4 CPUs", tmp_path / "new-workspace")

    result = state.read()
    assert result.startswith(authored)
    assert result.endswith("\n# Personal notes\nKeep this too.\n")
    assert result.count("## Runtime environment") == 1
    assert "Python 3.14; 4 CPUs" in result
    assert "Python 3.13" not in result
    assert f"Workspace: {tmp_path / 'new-workspace'}" in result
    assert "Observed at:" in result
    assert not list(tmp_path.glob(".world-*.tmp"))


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
async def test_world_refuses_nonregular_files(tmp_path, kind):
    profile = tmp_path / "profile"
    profile.mkdir()
    path = profile / "world.md"
    outside = tmp_path / "outside.md"
    outside.write_text("untouched")
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    state = WorldFile(profile)
    with pytest.raises(ToolError, match="cannot safely read"):
        state.read()
    with pytest.raises(ToolError, match="cannot safely update"):
        await state.update("observed", tmp_path)
    assert outside.read_text() == "untouched"


async def test_world_refuses_symlinked_profile_and_installed_package(
    tmp_path, monkeypatch
):
    import lingcore.world as module

    real = tmp_path / "real"
    real.mkdir()
    (real / "world.md").write_text("untouched")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ToolError, match="cannot safely update"):
        await WorldFile(link).update("observed", tmp_path)
    monkeypatch.setattr(module, "_PACKAGE_DIR", real)
    with pytest.raises(ToolError, match="installed package"):
        await WorldFile(real).update("observed", tmp_path)
    assert (real / "world.md").read_text() == "untouched"


@pytest.mark.parametrize(
    "existing, content",
    [
        ("authored", ""),
        ("authored", "é" * SNAPSHOT_MAX_BYTES),
        ("authored", "<!-- lingcore:environment:end -->"),
        ("<!-- lingcore:environment:start -->", "observed"),
        ("<!-- lingcore:environment:end --><!-- lingcore:environment:start -->", "x"),
        ("x" * WORLD_MAX_BYTES, "observed"),
        ("x" * (WORLD_MAX_BYTES + 1), "observed"),
    ],
)
async def test_invalid_update_never_changes_existing_world(tmp_path, existing, content):
    path = tmp_path / "world.md"
    path.write_text(existing)
    with pytest.raises(ToolError):
        await WorldFile(tmp_path).update(content, tmp_path)
    assert path.read_text() == existing


async def test_failed_replace_keeps_existing_file_and_cleans_temporary(
    tmp_path, monkeypatch
):
    from lingcore.paths import ConfinedDirectory

    path = tmp_path / "world.md"
    path.write_text("authored")

    def fail(*args):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(ConfinedDirectory, "replace", fail)
    with pytest.raises(ToolError, match="cannot safely update"):
        await WorldFile(tmp_path).update("observed", tmp_path)
    assert path.read_text() == "authored"
    assert not list(tmp_path.glob(".world-*.tmp"))


async def test_coding_init_inspects_then_updates_live_prompt(tmp_path, monkeypatch):
    from lingcore.io.cli import CLIFrontend

    monkeypatch.delenv("LINGCORE_WORKSPACE", raising=False)
    destination = initialize_profile("coding", destination=tmp_path / "coding")
    profile = AgentProfile.load(destination)
    profile.workspace = tmp_path / "project"
    original = (destination / "world.md").read_text()
    probe = "uname -s"
    snapshot = "## System\nVerified test runtime\n## Packages\nPython available"
    llm = FakeLLMClient(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="read", name="world", arguments={"action": "read"})
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="probe", name="run_shell", arguments={"command": probe})
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="save",
                        name="world",
                        arguments={"action": "update", "content": snapshot},
                    )
                ]
            ),
            ScriptedTurn(text="Environment initialized."),
        ]
    )
    confirmations = []

    async def confirm(command):
        confirmations.append(command)
        return True

    # Exercise the real shell tool with a portable, read-only host probe. The
    # shipped profile remains Bubblewrap; only this test selects the host runner.
    options = dict(profile.tool_options)
    options["run_shell"] = {"require_confirmation": True}
    async with Agent.from_profile(
        profile, llm=llm, confirm=confirm, tool_options=options
    ) as agent:
        assert "/init" in {c.qualified_name for c in agent.commands.commands}
        frontend = CLIFrontend()
        frontend.set_commands(agent.commands)

        async def typed_command():
            return "/init Python"

        monkeypatch.setattr(frontend, "_read_message", typed_command)
        incoming = await frontend.read_input()
        assert incoming is not None
        assert incoming.display_text == "/init Python"
        assert "Optional focus: Python" in incoming.text
        events = [event async for event in agent.run(incoming)]
        assert events[-1].content == "Environment initialized."
        assert len(confirmations) == 2
        assert confirmations[0] == probe
        assert snapshot in confirmations[1]
        assert str(destination / ".lingcore/world.md") in confirmations[1]
        assert any(
            "exit code: 0" in m.content for m in llm.calls[2] if m.role == "tool"
        )
        assert snapshot not in llm.calls[0][0].content
        assert snapshot in llm.calls[-1][0].content
        assert llm.calls[-1][0].content.count("# Environment") == 1
        assert "# Role" in llm.calls[-1][0].content
    assert (destination / "world.md").read_text() == original
    assert snapshot in (destination / ".lingcore/world.md").read_text()
    assert not (profile.workspace / "world.md").exists()
    async with Agent.from_profile(profile, llm=FakeLLMClient([])) as reopened:
        prompt = await reopened.composer.compose(ComposeContext("hi", 0))
        assert snapshot in prompt


async def test_world_uses_isolated_execution_state(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGCORE_WORKSPACE", raising=False)
    destination = initialize_profile("coding", destination=tmp_path / "coding")
    profile = AgentProfile.load(destination)
    original = (destination / "world.md").read_bytes()
    state_dir = tmp_path / "execution"
    async with Agent.from_profile(
        profile, llm=FakeLLMClient([]), state_dir=state_dir, confirm=_approve
    ) as agent:
        assert (
            await world(WorldArgs(action="read"), agent.tool_ctx) == original.decode()
        )
        await world(
            WorldArgs(action="update", content="Isolated runtime"), agent.tool_ctx
        )
        prompt = await agent.composer.compose(ComposeContext("hi", 0))
        assert "Isolated runtime" in prompt
        assert prompt.count("# Environment") == 1
    assert (destination / "world.md").read_bytes() == original
    assert "Isolated runtime" in (state_dir / ".lingcore/world.md").read_text()
    assert not (destination / ".lingcore/world.md").exists()


async def _approve(request):
    return True


async def test_world_created_for_profile_without_world_file(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "name: test\nllm: {model: m}\ntools: [world]\npersona: {system_prompt: Inline persona}\n"
    )
    async with Agent.from_profile(
        AgentProfile.load(cfg), llm=FakeLLMClient([]), confirm=_approve
    ) as agent:
        before = await agent.composer.compose(ComposeContext("hi", 0))
        assert before == "Inline persona"
        await world(WorldArgs(action="update", content="New runtime"), agent.tool_ctx)
        after = await agent.composer.compose(ComposeContext("hi", 0))
        assert "Inline persona" in after and "New runtime" in after
    assert not (tmp_path / "world.md").exists()
    assert "New runtime" in (tmp_path / ".lingcore/world.md").read_text()
    async with Agent.from_profile(
        AgentProfile.load(cfg), llm=FakeLLMClient([])
    ) as reopened:
        prompt = await reopened.composer.compose(ComposeContext("hi", 0))
        assert "Inline persona" in prompt and "New runtime" in prompt
