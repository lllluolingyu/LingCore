"""Durable, non-branch-scoped run journals sharing the session transaction lock."""

from __future__ import annotations

import builtins
import json
from typing import TYPE_CHECKING, Any

from lingcore.delegation.models import NodeRecord, RunRecord, now
from lingcore.errors import SessionError

if TYPE_CHECKING:
    from lingcore.sessions import SessionStore

RUN_DDL = [
    "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, parent_session_id TEXT, anchor_seq INTEGER NOT NULL, status TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS run_nodes (run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, identity TEXT NOT NULL, worker_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(run_id, identity))",
    "CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_session_id, anchor_seq)",
    "CREATE INDEX IF NOT EXISTS idx_run_workers ON run_nodes(worker_id)",
]


def detach_locked(conn: Any, session_id: str, seq: int, reason: str) -> None:
    rows = conn.execute(
        "SELECT id,payload FROM runs WHERE parent_session_id=? AND anchor_seq>?",
        (session_id, seq),
    ).fetchall()
    for run_id, payload in rows:
        run = RunRecord.model_validate_json(payload)
        if run.kind == "ticket":
            continue
        run.status, run.reason, run.updated_at = "detached", reason, now()
        conn.execute(
            "UPDATE runs SET status=?,updated_at=?,payload=? WHERE id=?",
            (run.status, run.updated_at, run.model_dump_json(), run_id),
        )


class RunJournal:
    """Same API for in-memory and durable runs; writes commit before execution."""

    def __init__(self, store: SessionStore | None) -> None:
        self.store = store
        self._runs: dict[str, RunRecord] = {}
        self._nodes: dict[tuple[str, str], NodeRecord] = {}

    def save(self, run: RunRecord) -> None:
        run.updated_at = now()
        if self.store is None:
            self._runs[run.id] = run.model_copy(deep=True)
            return
        with self.store._lock, self.store._conn:
            old = self.store._conn.execute(
                "SELECT status FROM runs WHERE id=?", (run.id,)
            ).fetchone()
            if (
                old
                and old[0] == "detached"
                and run.status in {"running", "pending", "completed"}
            ):
                run.status, run.reason = "detached", "starting turn was removed"
            self.store._conn.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,updated_at=excluded.updated_at,payload=excluded.payload",
                (
                    run.id,
                    run.parent_session_id,
                    run.anchor_seq,
                    run.status,
                    run.updated_at,
                    run.model_dump_json(),
                ),
            )

    def get(self, run_id: str) -> RunRecord:
        if self.store is None:
            run = self._runs.get(run_id)
            if run is None:
                raise SessionError("unknown run")
            return run.model_copy(deep=True)
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT payload,status FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise SessionError("unknown run")
        run = RunRecord.model_validate_json(row[0])
        run.status = row[1]
        return run

    def list(self, parent: str | None = None) -> list[RunRecord]:
        if self.store is None:
            values = [v.model_copy(deep=True) for v in self._runs.values()]
        else:
            with self.store._lock:
                rows = self.store._conn.execute(
                    "SELECT payload FROM runs ORDER BY updated_at DESC,id"
                ).fetchall()
            values = [RunRecord.model_validate_json(r[0]) for r in rows]
        return [r for r in values if parent is None or r.parent_session_id == parent]

    def node(self, node: NodeRecord) -> None:
        if self.store is None:
            self._nodes[(node.run_id, node.identity)] = node.model_copy(deep=True)
            return
        with self.store._lock, self.store._conn:
            self.store._conn.execute(
                "INSERT INTO run_nodes VALUES (?,?,?,?) ON CONFLICT(run_id,identity) DO UPDATE SET worker_id=excluded.worker_id,payload=excluded.payload",
                (
                    node.run_id,
                    node.identity,
                    node.worker_id,
                    node.model_dump_json(by_alias=True),
                ),
            )

    def nodes(self, run_id: str) -> builtins.list[NodeRecord]:
        if self.store is None:
            return [
                v.model_copy(deep=True)
                for (rid, _), v in self._nodes.items()
                if rid == run_id
            ]
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT payload FROM run_nodes WHERE run_id=? ORDER BY identity",
                (run_id,),
            ).fetchall()
        return [NodeRecord.model_validate_json(r[0]) for r in rows]

    def worker(self, worker_id: str) -> NodeRecord:
        if self.store is not None:
            with self.store._lock:
                row = self.store._conn.execute(
                    "SELECT n.payload FROM run_nodes n JOIN runs r ON r.id=n.run_id "
                    "WHERE n.worker_id=? ORDER BY r.updated_at DESC,n.identity LIMIT 1",
                    (worker_id,),
                ).fetchone()
            if row is None:
                raise SessionError("unknown worker")
            return NodeRecord.model_validate_json(row[0])
        for run in self.list():
            for node in self.nodes(run.id):
                if node.worker_id == worker_id:
                    return node
        raise SessionError("unknown worker")

    def lineage(self, session_id: str, value: dict[str, Any]) -> None:
        if self.store is None:
            return
        with self.store._lock, self.store._conn:
            self.store._conn.execute(
                "INSERT OR IGNORE INTO sessions(id,created_at,updated_at) VALUES (?,?,?)",
                (session_id, now(), now()),
            )
            row = self.store._conn.execute(
                "SELECT meta FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            meta = json.loads(row[0]) | {"delegation": value}
            self.store._conn.execute(
                "UPDATE sessions SET meta=? WHERE id=?", (json.dumps(meta), session_id)
            )

    def delete(self, run_id: str) -> None:
        run = self.get(run_id)
        if run.status in {"pending", "running", "interrupted"}:
            raise SessionError("cannot prune live or interrupted runs")
        nodes = self.nodes(run_id)
        if self.store is not None:
            with self.store._lock, self.store._conn:
                self.store._conn.execute("BEGIN IMMEDIATE")
                for node in nodes:
                    if node.session_id:
                        shared = self.store._conn.execute(
                            "SELECT 1 FROM run_nodes WHERE run_id<>? AND json_extract(payload,'$.session_id')=? LIMIT 1",
                            (run_id, node.session_id),
                        ).fetchone()
                        if shared:
                            continue
                        if self.store._conn.execute(
                            "SELECT 1 FROM runs WHERE id<>? AND parent_session_id=? LIMIT 1",
                            (run_id, node.session_id),
                        ).fetchone():
                            continue
                        self.store._leases.check_locked(node.session_id, None)
                        self.store._conn.execute(
                            "DELETE FROM sessions WHERE id=?", (node.session_id,)
                        )
                self.store._conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        else:
            self._runs.pop(run_id, None)
            self._nodes = {k: v for k, v in self._nodes.items() if k[0] != run_id}

    def revive(self, run: RunRecord) -> None:
        """Explicit confirmed-resume transition, separate from background writes."""
        run.status, run.reason = "pending", None
        if self.store is not None:
            with self.store._lock, self.store._conn:
                self.store._conn.execute(
                    "UPDATE runs SET status='pending' WHERE id=?", (run.id,)
                )
        self.save(run)
