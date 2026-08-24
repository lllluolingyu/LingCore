"""Tests for the structured, read-only Git builtin."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.tools import REGISTRY, ToolContext
from lingcore.tools.builtin.git import GitArgs, git


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _run(tmp_path, "init", "--quiet")
    _run(tmp_path, "config", "user.name", "LingCore Tests")
    _run(tmp_path, "config", "user.email", "lingcore@example.invalid")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    _run(tmp_path, "add", "--", "tracked.txt")
    _run(tmp_path, "commit", "--quiet", "-m", "initial")
    return tmp_path


def _ctx(repo: Path, *, confirm=None) -> ToolContext:
    return ToolContext(workspace=repo, confirm=confirm)


async def test_status_is_structured_and_does_not_confirm(repo):
    (repo / "tracked.txt").write_text("after\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")

    async def unexpected_confirmation(prompt: str) -> bool:
        raise AssertionError(f"read-only git requested confirmation: {prompt}")

    out = await git(
        GitArgs(action="status"), _ctx(repo, confirm=unexpected_confirmation)
    )

    assert "(read-only)" in out
    assert " M tracked.txt" in out
    assert "?? new.txt" in out


async def test_diff_supports_worktree_staged_revision_and_literal_path(repo):
    odd_name = ":(exclude)*.txt"
    (repo / odd_name).write_text("odd before\n", encoding="utf-8")
    _run(repo, "--literal-pathspecs", "add", "--", odd_name)
    _run(repo, "commit", "--quiet", "-m", "add odd path")
    (repo / odd_name).write_text("odd after\n", encoding="utf-8")

    worktree = await git(GitArgs(action="diff", paths=[odd_name]), _ctx(repo))
    assert "+odd after" in worktree
    assert "-odd before" in worktree

    _run(repo, "--literal-pathspecs", "add", "--", odd_name)
    staged = await git(
        GitArgs(action="diff", staged=True, revision="HEAD", paths=[odd_name]),
        _ctx(repo),
    )
    assert "+odd after" in staged


async def test_log_show_and_branches(repo):
    log = await git(GitArgs(action="log", max_count=1), _ctx(repo))
    assert "initial" in log

    show = await git(GitArgs(action="show", revision="HEAD"), _ctx(repo))
    assert "initial" in show
    assert "+before" in show

    branches = await git(GitArgs(action="branches"), _ctx(repo))
    assert "* " in branches


async def test_empty_diff_has_explicit_output(repo):
    out = await git(GitArgs(action="diff"), _ctx(repo))
    assert out.endswith("(no output)")


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd"])
async def test_paths_cannot_escape_workspace(repo, path):
    with pytest.raises(ToolError, match="escapes workspace"):
        await git(GitArgs(action="diff", paths=[path]), _ctx(repo))


@pytest.mark.parametrize("revision", ["--output=/tmp/leak", "HEAD\n--all", ""])
async def test_revisions_cannot_inject_options(repo, revision):
    with pytest.raises(ToolError, match="invalid Git revision"):
        await git(GitArgs(action="show", revision=revision), _ctx(repo))


async def test_irrelevant_operation_arguments_are_rejected(repo):
    with pytest.raises(ToolError, match="staged is supported only"):
        await git(GitArgs(action="status", staged=True), _ctx(repo))
    with pytest.raises(ToolError, match="paths are not supported"):
        await git(GitArgs(action="branches", paths=["tracked.txt"]), _ctx(repo))


async def test_parent_repository_is_not_discovered(repo):
    nested = repo / "nested"
    nested.mkdir()
    with pytest.raises(ToolError, match="workspace root"):
        await git(GitArgs(action="status"), _ctx(nested))


async def test_linked_or_separate_git_directory_is_refused(tmp_path):
    actual_git = tmp_path / "metadata"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _run(checkout, "init", "--quiet", f"--separate-git-dir={actual_git}")

    with pytest.raises(ToolError, match="linked/separate worktrees"):
        await git(GitArgs(action="status"), _ctx(checkout))


async def test_object_alternates_are_refused(repo):
    info = repo / ".git" / "objects" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "alternates").write_text("/tmp/other-objects\n", encoding="utf-8")

    with pytest.raises(ToolError, match="object alternates"):
        await git(GitArgs(action="show"), _ctx(repo))


async def test_symlinked_git_metadata_cannot_escape(repo):
    outside = repo.parent / f"{repo.name}-outside-index"
    outside.write_bytes((repo / ".git" / "index").read_bytes())
    (repo / ".git" / "index").unlink()
    try:
        (repo / ".git" / "index").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this platform")

    with pytest.raises(ToolError, match="metadata path escapes"):
        await git(GitArgs(action="status"), _ctx(repo))


async def test_external_diff_and_fsmonitor_helpers_are_disabled(repo):
    marker = repo / "helper-ran"
    helper = repo / "helper.sh"
    helper.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    helper.chmod(helper.stat().st_mode | 0o111)
    (repo / ".gitattributes").write_text("*.txt diff=unsafe\n", encoding="utf-8")
    _run(repo, "add", "--", ".gitattributes")
    _run(repo, "commit", "--quiet", "-m", "attributes")
    _run(repo, "config", "diff.unsafe.command", str(helper))
    _run(repo, "config", "core.fsmonitor", str(helper))
    (repo / "tracked.txt").write_text("after\n", encoding="utf-8")

    await git(GitArgs(action="status"), _ctx(repo))
    out = await git(GitArgs(action="diff"), _ctx(repo))

    assert "+after" in out
    assert not marker.exists()


async def test_large_output_is_truncated_without_writing_runtime_files(repo):
    (repo / "tracked.txt").write_text("x" * 40_000 + "\n", encoding="utf-8")
    out = await git(GitArgs(action="diff"), _ctx(repo))

    assert "truncated" in out
    assert "narrow paths or revision" in out
    assert not (repo / ".lingcore").exists()


def test_git_is_registered_and_mutating_actions_are_not_in_schema():
    assert REGISTRY.get("git") is git
    with pytest.raises(ValidationError):
        GitArgs.model_validate({"action": "commit"})
    actions = git.json_schema()["function"]["parameters"]["properties"]["action"][
        "enum"
    ]
    assert actions == ["status", "diff", "log", "show", "branches"]


def test_git_environment_does_not_inherit_ambient_git_overrides(repo, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/tmp/hostile")
    from lingcore.tools.builtin.git import _git_environment

    environment = _git_environment(repo)
    assert "GIT_DIR" not in environment
    assert environment["GIT_ALLOW_PROTOCOL"] == ""
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    if os.name != "nt":
        assert "HOME" not in environment


@pytest.mark.parametrize("profile_name", ["coding", "coding_ollama"])
def test_bundled_coding_profiles_enable_git(profile_name):
    profile_dir = Path(__file__).parents[1] / "profiles" / profile_name
    assert "git" in AgentProfile.load(profile_dir).tools
