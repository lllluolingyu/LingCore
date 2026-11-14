"""Confined SQLite transactions; the pure engine alone decides state changes."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from lingcore.paths import confined_directory
from lingcore.superagent.core.models import State

T = TypeVar("T")
U = TypeVar("U")


class SuperagentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._directory_context = confined_directory(path.parent, ".", create=True)
        self._directory = self._directory_context.__enter__()
        for name in (path.name, path.name + "-wal", path.name + "-shm"):
            if (
                self._directory.entry_exists(name)
                and self._directory.regular_size(name) is None
            ):
                raise ValueError(
                    "superagent database and sidecars must be regular files"
                )
        if not self._directory.entry_exists(path.name):
            with self._directory.open_exclusive(path.name, mode=0o600):
                pass
        self.conn = sqlite3.connect(
            f"/proc/self/fd/{self._directory._fd}/{path.name}", check_same_thread=False
        )
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL)"
        )
        for table in (
            "triggers",
            "threads",
            "tickets",
            "decisions",
            "effects",
            "outbox",
        ):
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY,payload TEXT NOT NULL)"
            )
        self.conn.commit()
        self._cached: State | None = None
        self._version: int | None = None

    def _load(self) -> State:
        version = self.conn.execute("PRAGMA data_version").fetchone()[0]
        if self._cached is None or self._version != version:
            row = self.conn.execute("SELECT payload FROM state WHERE id=1").fetchone()
            self._cached = State.model_validate_json(row[0]) if row else State()
            self._version = version
        return self._cached

    def read(self) -> State:
        with self.lock:
            self._directory.ensure_anchored()
            return self._load().model_copy(deep=True)

    def transact(self, factory: Callable[[State], U], command: Callable[[U], T]) -> T:
        with self.lock, self.conn:
            self._directory.ensure_anchored()
            self.conn.execute("BEGIN IMMEDIATE")
            before = self._load()
            state = before.model_copy(deep=True)
            result = command(factory(state))
            if state == before:
                return result
            self.conn.execute(
                "INSERT INTO state VALUES(1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (state.model_dump_json(),),
            )
            previous, current = _tables(before), _tables(state)
            for table, records in current.items():
                for key in previous[table].keys() - records.keys():
                    self.conn.execute(f"DELETE FROM {table} WHERE id=?", (key,))
                for key, value in records.items():
                    if previous[table].get(key) == value:
                        continue
                    payload = (
                        value.model_dump_json()
                        if hasattr(value, "model_dump_json")
                        else json.dumps(value)
                    )
                    self.conn.execute(
                        f"INSERT INTO {table} VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                        (key, payload),
                    )
            # Commit before replacing the cache; rollback never exposes a
            # command's partial mutations, including on another connection.
            self.conn.commit()
            self._cached = state.model_copy(deep=True)
            return result

    def close(self) -> None:
        self.conn.close()
        self._directory_context.__exit__(None, None, None)


def _tables(state: State) -> dict[str, dict[str, Any]]:
    return {
        "triggers": state.triggers,
        "tickets": state.tickets,
        "decisions": state.evaluations,
        "effects": state.effects,
        "threads": state.thread_queues,
        "outbox": {k: v for k, v in state.effects.items() if v.kind == "reply"},
    }
