"""Approval, authored-file compatibility, isolation, and concurrent saves."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from lingcore.agent import Agent
from lingcore.composer import ComposeContext
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, ToolError
from lingcore.integrations._profiles import scoped_profile
from lingcore.tools import ToolContext
from lingcore.tools.builtin.world import WorldArgs, world
from lingcore.world import WORLD_MAX_BYTES, WORLD_STATE_KEY, WorldFile
from tests.fakes import FakeLLMClient


def _profile(root):
    root.mkdir(exist_ok=True)
    (root / "config.yaml").write_text("llm: {model: test}\ntools: [world]\n")
    return AgentProfile.load(root)


@pytest.mark.parametrize("decision", [None, False, "cancel"])
async def test_world_save_requires_fresh_explicit_approval(tmp_path, decision):
    requests = []

    async def confirm(request):
        requests.append(request)
        if decision == "cancel":
            raise asyncio.CancelledError
        return decision

    profile = _profile(tmp_path)
    async with Agent.from_profile(
        profile,
        llm=FakeLLMClient([]),
        confirm=None if decision is None else confirm,
    ) as agent:
        assert (
            await world(WorldArgs(action="read"), agent.tool_ctx)
            == "(world.md is empty)"
        )
        with pytest.raises(
            asyncio.CancelledError if decision == "cancel" else ToolError
        ):
            await world(
                WorldArgs(action="update", content="Untrusted text"), agent.tool_ctx
            )
    assert not (tmp_path / ".lingcore").exists()
    assert not (tmp_path / "world.md").exists()
    if requests:
        assert "Untrusted text" in requests[0]
        assert "system-prompt context in later sessions" in requests[0]


@pytest.mark.parametrize("kind", ["large", "symlink"])
async def test_authored_world_keeps_legacy_loading_and_is_never_modified(
    tmp_path, kind
):
    profile = _profile(tmp_path)
    authored = "Authored environment\n" + "x" * (WORLD_MAX_BYTES + 1)
    path = tmp_path / "world.md"
    if kind == "symlink":
        target = tmp_path / "authored.md"
        target.write_text(authored)
        path.symlink_to(target)
    else:
        path.write_text(authored)

    async def confirm(request):
        assert "Observed runtime" in request
        return True

    async with Agent.from_profile(
        profile, llm=FakeLLMClient([]), confirm=confirm
    ) as agent:
        assert authored in await agent.initial_system_prompt()
        await world(
            WorldArgs(action="update", content="Observed runtime"), agent.tool_ctx
        )
        prompt = await agent.composer.compose(ComposeContext("next", 1))
        assert authored in prompt and "Observed runtime" in prompt
    assert path.read_text() == authored
    assert path.is_symlink() is (kind == "symlink")
    assert "Authored environment" not in (tmp_path / ".lingcore/world.md").read_text()


async def test_unsafe_runtime_state_does_not_prevent_startup(tmp_path):
    profile = _profile(tmp_path)
    (tmp_path / "world.md").write_text("Authored context")
    runtime = tmp_path / ".lingcore"
    runtime.mkdir()
    (runtime / "world.md").symlink_to(tmp_path / "world.md")
    async with Agent.from_profile(profile, llm=FakeLLMClient([])) as agent:
        assert await agent.initial_system_prompt() == "Authored context"
        with pytest.raises(ToolError, match="cannot safely read"):
            await world(WorldArgs(action="read"), agent.tool_ctx)


@pytest.mark.parametrize("absolute", [False, True])
async def test_channel_world_state_is_scoped_per_user(tmp_path, absolute):
    profile = _profile(tmp_path)
    (tmp_path / "world.md").write_text("Shared authored context")
    state_dir = str(tmp_path / "channel") if absolute else "channel"
    user_root = tmp_path / "channel/users"

    async def confirm(request):
        return True

    for user in ("alice", "bob"):
        user_dir = user_root / user
        user_dir.mkdir(parents=True)
        scoped = scoped_profile(profile, state_dir=state_dir, user_dir=user_dir)
        async with Agent.from_profile(
            scoped, llm=FakeLLMClient([]), confirm=confirm
        ) as agent:
            before = await agent.initial_system_prompt()
            assert "Shared authored context" in before
            assert "alice-only" not in before
            await world(
                WorldArgs(action="update", content=f"{user}-only runtime"),
                agent.tool_ctx,
            )
            saved = user_dir / ".lingcore/world.md"
            assert f"{user}-only runtime" in saved.read_text()
    assert not (tmp_path / ".lingcore/world.md").exists()
    assert profile.tool_options == {}


async def test_two_approvals_of_same_revision_cannot_silently_overwrite(tmp_path):
    state = WorldFile(tmp_path / ".lingcore")
    requests = []
    ready = asyncio.Event()
    release = asyncio.Event()

    async def confirm(request):
        requests.append(request)
        if len(requests) == 2:
            ready.set()
        await release.wait()
        return True

    def save(content):
        ctx = ToolContext(
            workspace=tmp_path,
            options={WORLD_STATE_KEY: state},
            confirm=confirm,
        )
        return world(WorldArgs(action="update", content=content), ctx)

    tasks = [asyncio.create_task(save(content)) for content in ("first", "second")]
    await asyncio.wait_for(ready.wait(), 2)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(isinstance(result, str) for result in results) == 1
    rejected = next(result for result in results if isinstance(result, ToolError))
    assert "changed while awaiting approval" in str(rejected)
    assert len(requests) == 2
    assert state.read().count("## Runtime environment") == 1
    assert state.path.with_suffix(".md.lock").is_file()


async def test_cross_process_lock_wait_is_cancellable_and_reusable(tmp_path):
    state = WorldFile(tmp_path)
    await state.update("initial", tmp_path)
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import fcntl, sys; f=open(sys.argv[1], 'r+b'); "
        "fcntl.flock(f, fcntl.LOCK_EX); print('ready', flush=True); sys.stdin.read(1)",
        str(tmp_path / "world.md.lock"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(child.stdout.readline(), 3) == b"ready\n"
        task = asyncio.create_task(state.update("blocked", tmp_path))
        await asyncio.sleep(0.1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "initial" in state.read() and "blocked" not in state.read()
    finally:
        await asyncio.wait_for(child.communicate(b"x"), 3)
    await state.update("after release", tmp_path)
    assert "after release" in state.read()


async def test_replaced_lock_is_refused(tmp_path, monkeypatch):
    from lingcore.paths import ConfinedLockFile

    acquire = ConfinedLockFile.try_acquire

    def replace_lock(lock):
        acquired = acquire(lock)
        (tmp_path / "world.md.lock").unlink()
        (tmp_path / "world.md.lock").write_text("")
        return acquired

    monkeypatch.setattr(ConfinedLockFile, "try_acquire", replace_lock)
    state = WorldFile(tmp_path)
    with pytest.raises(ToolError, match="cannot safely update"):
        await state.update("must not publish", tmp_path)
    assert not state.path.exists()


async def test_runtime_path_cannot_target_authored_world_or_escape_execution(tmp_path):
    profile = _profile(tmp_path)
    profile.tool_options["world"] = {"path": "world.md"}
    with pytest.raises(ConfigError, match="must differ"):
        Agent.from_profile(profile, llm=FakeLLMClient([]))
    profile.tool_options["world"] = {"path": "../outside.md"}
    with pytest.raises(ConfigError, match="escapes|confined"):
        Agent.from_profile(
            profile, llm=FakeLLMClient([]), state_dir=tmp_path / "execution"
        )


async def test_checkout_snapshot_is_ignored_and_authored_file_stays_clean(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / ".gitignore").write_text("/profiles/*/.lingcore/\n")
    root = repo / "profiles/coding"
    root.mkdir(parents=True)
    profile = _profile(root)
    profile.workspace = str(tmp_path / "workspace")
    (root / "world.md").write_text("Tracked author notes\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)

    def status():
        return subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"]
        )

    before = status()

    async def confirm(request):
        return True

    async with Agent.from_profile(
        profile, llm=FakeLLMClient([]), confirm=confirm
    ) as agent:
        await world(
            WorldArgs(action="update", content="Private host details"), agent.tool_ctx
        )
    assert status() == before
    assert (root / "world.md").read_text() == "Tracked author notes\n"
    assert (root / ".lingcore/world.md").stat().st_mode & 0o777 == 0o600
