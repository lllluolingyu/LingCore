"""Durable active-session selection for Telegram users."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from lingcore.errors import ConfigError
from lingcore.sessions import is_session_id

_LOG = logging.getLogger(__name__)


class TelegramStateStore:
    """Small bridge registry separate from every user's transcript database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._closed = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            with self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA busy_timeout=5000")
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS active_sessions (
                      user_id INTEGER PRIMARY KEY,
                      session_id TEXT NOT NULL,
                      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
        except (OSError, sqlite3.DatabaseError) as exc:
            connection = getattr(self, "_conn", None)
            if connection is not None:
                connection.close()
            raise ConfigError(
                f"cannot open Telegram bridge state at {path}: {exc}"
            ) from None

    def active_session(self, user_id: int) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id FROM active_sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        session_id = str(row[0])
        if not is_session_id(session_id):
            # The selection is only a cursor into the per-user session store.
            # Treat corruption as no selection so runtime construction can mint
            # and atomically persist a fresh session instead of bricking the user.
            _LOG.warning(
                "Ignoring invalid Telegram active-session selection for user %s",
                user_id,
            )
            return None
        return session_id

    def set_active_session(self, user_id: int, session_id: str) -> None:
        if user_id <= 0:
            raise ConfigError("Telegram user ids must be positive")
        if not is_session_id(session_id):
            raise ConfigError("cannot store an invalid Telegram session id")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO active_sessions (user_id, session_id, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                      session_id = excluded.session_id,
                      updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, session_id),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True
