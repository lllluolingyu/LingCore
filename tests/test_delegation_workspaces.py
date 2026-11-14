from __future__ import annotations

import shutil

import pytest

from lingcore.delegation.config import WorkspaceConfig
from lingcore.delegation.workspaces import WorkspaceManager, git
from lingcore.errors import ToolError


def checkout(path):
    path.mkdir()
    git(path, "init")
    (path / "tracked").write_text("base\n")
    git(path, "add", "-A")
    git(path, "commit", "-m", "base")


def test_snapshot_carries_dirty_input_but_patch_only_contains_worker_edits(tmp_path):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    (workspace / "tracked").write_text("parent dirty\n")
    (workspace / "untracked").write_text("local\n")
    manager = WorkspaceManager(workspace, WorkspaceConfig())
    target, _, skipped = manager.capture("r-test", "w-test")
    assert not skipped
    assert (target / "tracked").read_text() == "parent dirty\n"
    assert (target / "untracked").read_text() == "local\n"
    assert not (target / ".git/objects/info/alternates").exists()
    original_objects = [
        p for p in (workspace / ".git/objects").rglob("*") if p.is_file()
    ]
    for original in original_objects:
        copied = target / original.relative_to(workspace)
        assert original.stat().st_ino != copied.stat().st_ino
    (target / "tracked").write_text("worker edit\n")
    patch = manager.read(manager.patch("r-test", "w-test"))
    assert b"-parent dirty" in patch and b"+worker edit" in patch
    assert b"+local" not in patch and b"-base" not in patch
    assert (workspace / "tracked").read_text() == "parent dirty\n"


def test_frozen_manifest_tracks_content_modes_and_is_removable(tmp_path):
    (tmp_path / "input").write_text("value")
    manager = WorkspaceManager(tmp_path, WorkspaceConfig())
    target, revision, skipped = manager.capture("r-test", "input", frozen=True)
    assert not target.stat().st_mode & 0o222
    assert not (target / "input").stat().st_mode & 0o222
    assert manager.revision(target, skipped) == revision
    manager.remove("r-test")
    assert not target.exists()


def test_non_git_copy_is_bounded_and_does_not_follow_links(tmp_path):
    (tmp_path / "a").write_text("123")
    (tmp_path / "b").write_text("456")
    (tmp_path / "link").symlink_to("/etc/passwd")
    manager = WorkspaceManager(tmp_path, WorkspaceConfig(snapshot_max_bytes=4))
    target, _, skipped = manager.capture("r-test", "input", frozen=True)
    assert (target / "a").read_text() == "123"
    assert not (target / "b").exists() and not (target / "link").exists()
    assert "b (cap)" in skipped and "link (symlink)" in skipped
    manager.remove("r-test")


@pytest.mark.parametrize("bad", ["../escape", "/tmp", "r/child", "..", "R-upper"])
def test_untrusted_run_ids_never_escape(tmp_path, bad):
    with pytest.raises(ToolError):
        WorkspaceManager(tmp_path, WorkspaceConfig()).path(bad)


def test_alternates_and_linked_metadata_refused_before_clone(tmp_path):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    (workspace / ".git/objects/info/alternates").write_text("/tmp")
    with pytest.raises(ToolError, match="alternates"):
        WorkspaceManager(workspace, WorkspaceConfig()).capture("r-test", "w-test")


def test_snapshot_diff_never_executes_parent_filters_and_keeps_staged_additions(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    (workspace / "staged-new").write_text("staged addition")
    git(workspace, "add", "staged-new")
    (workspace / "tracked").write_text("dirty text")
    (workspace / ".gitattributes").write_text("tracked filter=evil\n")
    marker = tmp_path / "filter-ran"
    git(workspace, "config", "filter.evil.clean", f"touch {marker}")
    manager = WorkspaceManager(workspace, WorkspaceConfig())
    target, _, _ = manager.capture("r-safe", "w-safe")
    assert not marker.exists()
    assert (target / "tracked").read_text() == "dirty text"
    assert (target / "staged-new").read_text() == "staged addition"


def test_worker_git_config_cannot_execute_during_host_patch_creation(tmp_path):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    manager = WorkspaceManager(workspace, WorkspaceConfig())
    target, _, _ = manager.capture("r-escape", "w-escape")
    marker = tmp_path / "escaped"
    (target / ".git/config").write_text(
        f'[core]\nrepositoryformatversion=0\nbare=false\n[filter "escape"]\nclean=touch {marker}\n'
    )
    (target / ".gitattributes").write_text("* filter=escape\n")
    (target / "tracked").write_text("worker change\n")
    patch = manager.read(manager.patch("r-escape", "w-escape"))
    assert not marker.exists()
    assert b"+worker change" in patch


@pytest.mark.parametrize("frozen", [False, True])
def test_tracked_symlinks_are_skipped_without_preventing_capture(tmp_path, frozen):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    (workspace / "link").symlink_to("/etc/passwd")
    git(workspace, "add", "link")
    git(workspace, "commit", "-m", "link")
    manager = WorkspaceManager(workspace, WorkspaceConfig())
    target, revision, skipped = manager.capture("r-link", "w-link", frozen=frozen)
    assert not (target / "link").exists()
    assert "link (symlink)" in skipped
    assert manager.revision(target, skipped) == revision


def test_patch_preserves_tracked_files_in_normally_excluded_directories(tmp_path):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    dependency = workspace / "node_modules/package"
    dependency.mkdir(parents=True)
    (dependency / "tracked.js").write_text("tracked dependency\n")
    git(workspace, "add", "-A")
    git(workspace, "commit", "-m", "tracked dependency")
    manager = WorkspaceManager(workspace, WorkspaceConfig())
    target, _, _ = manager.capture("r-tracked", "w-tracked")
    (target / "tracked").write_text("worker change\n")
    patch = manager.read(manager.patch("r-tracked", "w-tracked"))
    assert b"+worker change" in patch
    assert b"node_modules" not in patch


def test_patch_ignores_replaced_and_nested_git_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    manager = WorkspaceManager(workspace, WorkspaceConfig())
    target, _, _ = manager.capture("r-metadata", "w-metadata")
    shutil.rmtree(target / ".git")
    (target / ".git").write_text(f"gitdir: {workspace / '.git'}\n")
    nested = target / "nested"
    nested.mkdir()
    (nested / ".git").write_text(f"gitdir: {workspace / '.git'}\n")
    (nested / "result").write_text("worker result\n")
    patch = manager.read(manager.patch("r-metadata", "w-metadata"))
    assert b"+worker result" in patch
    assert b"gitdir:" not in patch
    assert git(workspace, "status", "--porcelain") == b"?? .lingcore/\n"


@pytest.mark.parametrize("frozen", [False, True])
def test_unborn_git_workspaces_can_be_captured(tmp_path, frozen):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    git(workspace, "init")
    (workspace / "new").write_text("initial")
    git(workspace, "add", "new")
    manager = WorkspaceManager(workspace, WorkspaceConfig())
    target, _, _ = manager.capture("r-unborn", "w-unborn", frozen=frozen)
    assert (target / "new").read_text() == "initial"
    if not frozen:
        (target / "new").write_text("worker")
        assert b"+worker" in manager.read(manager.patch("r-unborn", "w-unborn"))


def test_worker_symlinks_fail_patch_creation_explicitly(tmp_path):
    manager = WorkspaceManager(tmp_path, WorkspaceConfig())
    target, _, _ = manager.capture("r-link", "w-link")
    (target / "output").symlink_to("/etc/passwd")
    with pytest.raises(ToolError, match="symlinks"):
        manager.patch("r-link", "w-link")


def test_streamed_artifact_hash_keeps_existing_journal_format_and_caps(tmp_path):
    from lingcore.delegation.models import digest

    manager = WorkspaceManager(tmp_path, WorkspaceConfig())
    data = b"chunk" * 100000
    (tmp_path / "source").write_bytes(data)
    assert manager.retain_artifact(
        tmp_path, "source", "r-art", "output", len(data)
    ) == len(data)
    relative = ".lingcore/runs/r-art/output"
    assert manager.artifact_digest(relative) == digest(data.hex())
    assert manager.verify_artifacts({relative: digest(data.hex())})
    with pytest.raises(ToolError, match="byte cap"):
        manager.retain_artifact(tmp_path, "source", "r-art", "too-big", 2)
    assert not (tmp_path / ".lingcore/runs/r-art/too-big").exists()
    assert not list((tmp_path / ".lingcore/runs/r-art").glob("*.part"))
