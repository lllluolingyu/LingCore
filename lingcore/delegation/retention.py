"""Explicit/automatic retention of finished journals and their owned artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from lingcore._sessions.runs import RunJournal
from lingcore.delegation.config import RetentionConfig
from lingcore.delegation.workspaces import WorkspaceManager
from lingcore.errors import SessionError


def prune(
    journal: RunJournal, workspace: WorkspaceManager, config: RetentionConfig
) -> dict[str, Any]:
    records = sorted(journal.list(), key=lambda r: (r.updated_at, r.id))
    local = [r for r in records if r.workspace == str(workspace.workspace.resolve())]
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.max_age_days)
    finished = [
        r for r in local if r.status not in {"pending", "running", "interrupted"}
    ]
    excess = max(0, len(local) - config.max_runs)
    selected = {
        r.id
        for i, r in enumerate(finished)
        if i < excess or datetime.fromisoformat(r.updated_at) < cutoff
    }
    workers = {r.id: {n.worker_id for n in journal.nodes(r.id)} for r in records}
    sessions = {
        r.id: {n.session_id for n in journal.nodes(r.id) if n.session_id}
        for r in records
    }
    # Continued sessions/scratch workspaces stay until every referencing run can
    # be pruned. Never destroy the workspace of a live or interrupted child.
    changed = True
    while changed:
        changed = False
        for record in records:
            if record.id in selected and any(
                (workers[record.id] & workers[r.id])
                or r.parent_session_id in sessions[record.id]
                for r in records
                if r.id not in selected
            ):
                selected.remove(record.id)
                changed = True
    removed, retained = [], []
    for record in finished:
        if record.id not in selected:
            continue
        try:
            journal.delete(record.id)
        except SessionError:
            retained.append(record.id)
            continue
        try:
            workspace.remove(record.id)
        except FileNotFoundError:
            pass
        removed.append(record.id)
    return {"removed": removed, "leased": retained}


def retained_size(workspace: WorkspaceManager) -> dict[str, int]:
    from lingcore.paths import confined_directory

    count, size = 0, 0

    def walk(directory: Any) -> None:
        nonlocal count, size
        for name, is_dir, regular, symlink in directory.iter_entries():
            if symlink:
                continue
            if is_dir:
                with directory.subdirectory(name) as child:
                    walk(child)
            elif regular:
                count += 1
                size += directory.regular_size(name) or 0

    try:
        with confined_directory(workspace.workspace, ".lingcore/runs") as directory:
            walk(directory)
    except FileNotFoundError:
        pass
    return {"files": count, "bytes": size}
