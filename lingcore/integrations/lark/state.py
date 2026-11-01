"""Durable active-session selection, keyed by conversation and sender."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lingcore.sessions import is_session_id


class LarkStateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        try:
            with self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS active_sessions "
                    "(scope TEXT PRIMARY KEY, session_id TEXT NOT NULL)"
                )
        except BaseException:
            self._conn.close()
            raise

    def get(self, scope: str) -> str | None:
        row = self._conn.execute(
            "SELECT session_id FROM active_sessions WHERE scope = ?", (scope,)
        ).fetchone()
        return str(row[0]) if row and is_session_id(str(row[0])) else None

    def set(self, scope: str, session_id: str) -> None:
        if not is_session_id(session_id):
            raise ValueError("invalid session id")
        with self._conn:
            self._conn.execute(
                "INSERT INTO active_sessions VALUES (?, ?) "
                "ON CONFLICT(scope) DO UPDATE SET session_id = excluded.session_id",
                (scope, session_id),
            )

    def close(self) -> None:
        self._conn.close()
