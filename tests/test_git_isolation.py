from __future__ import annotations

import pytest

from lingcore.tools import ToolContext
from lingcore.tools.builtin.git import GitArgs, git
from tests.test_delegation_workspaces import checkout


@pytest.mark.parametrize("action", ["status", "diff"])
async def test_read_only_git_never_runs_worktree_clean_filters(tmp_path, action):
    workspace = tmp_path / "workspace"
    checkout(workspace)
    marker = tmp_path / "escaped"
    (workspace / ".git/config").write_text(
        f'[core]\nrepositoryformatversion=0\nbare=false\n[filter "escape"]\nclean=touch {marker}\n'
    )
    (workspace / ".gitattributes").write_text("* filter=escape\n")
    (workspace / "tracked").write_text("dirty change\n")
    await git(GitArgs(action=action), ToolContext(workspace))
    assert not marker.exists()
