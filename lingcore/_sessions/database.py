"""Session database implementation."""

from __future__ import annotations

import builtins
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

from lingcore._sessions.models import (
    _PREFIX_RE,
    SessionMeta,
    StoredMessage,
    _message_title,
    _row_to_meta,
    _stored_message,
    _trim_dangling_records,
    is_session_id,
)
from lingcore.errors import SessionError
from lingcore.message import Message
from lingcore.session_leases import LEASE_DDL, SessionLease, SessionLeases

_SCHEMA_VERSION = 3

_DDL = [
    LEASE_DDL,
    """
    CREATE TABLE IF NOT EXISTS sessions (
      id           TEXT PRIMARY KEY,
      title        TEXT NOT NULL DEFAULT '',
      profile_name TEXT NOT NULL DEFAULT '',
      created_at   TEXT NOT NULL,
      updated_at   TEXT NOT NULL,
      meta         TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
      session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
      seq         INTEGER NOT NULL,
      role        TEXT    NOT NULL,
      created_at  TEXT    NOT NULL,
      payload     TEXT    NOT NULL,
      PRIMARY KEY (session_id, seq)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_events (
      event_seq    INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
      message_seq INTEGER NOT NULL,
      kind        TEXT    NOT NULL,
      created_at  TEXT    NOT NULL,
      payload     TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_session_events_cursor "
    "ON session_events(session_id, event_seq)",
    "CREATE INDEX IF NOT EXISTS idx_session_events_anchor "
    "ON session_events(session_id, message_seq)",
]

_META_QUERY = """
SELECT s.id, s.title, s.profile_name, s.created_at, s.updated_at, s.meta,
       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n
FROM sessions s
"""


class SessionDatabase:
    """Connection ownership, message CRUD, and lease-fenced transactions."""

    def __init__(self, db_path: Path, *, profile_name: str = "") -> None:
        self.db_path = db_path
        self._profile_name = profile_name
        self._lock = threading.Lock()
        self._closed = False
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._leases = SessionLeases(self._conn, self._lock)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > _SCHEMA_VERSION:
                raise SessionError(
                    f"session db {db_path} has schema version {version}, newer than "
                    f"this lingcore understands ({_SCHEMA_VERSION}); upgrade lingcore"
                )
            if version < _SCHEMA_VERSION:
                with self._conn:
                    self._conn.execute("BEGIN IMMEDIATE")
                    for stmt in _DDL:
                        self._conn.execute(stmt)
                    self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        except BaseException:
            self._conn.close()
            raise

    @contextmanager
    def _write(
        self,
        session_id: str,
        lease: SessionLease | None = None,
        *,
        detach: bool = False,
    ) -> Iterator[None]:
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                self._leases.check_locked(session_id, lease)
                yield
                if detach:
                    self._conn.execute(
                        "UPDATE session_leases SET owner_token=NULL, heartbeat=0 WHERE session_id=?",
                        (session_id,),
                    )
            if detach:
                self._leases.owned.pop(session_id, None)

    def acquire_session(self, session_id: str) -> SessionLease:
        """Acquire before hydration; release after the agent has stopped."""
        if not is_session_id(session_id):
            raise SessionError(f"invalid session id: {session_id!r}")
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                lease = self._leases.acquire_locked(session_id)
            self._leases.owned[session_id] = lease
            self._leases.start_heartbeat()
        return lease

    def heartbeat_session(self, lease: SessionLease) -> None:
        """Validate ownership, including expiry, and renew its heartbeat."""
        with self._write(lease.session_id, lease):
            pass

    def release_session(self, lease: SessionLease) -> None:
        with self._lock:
            try:
                if not self._closed:
                    with self._conn:
                        self._conn.execute("BEGIN IMMEDIATE")
                        self._leases.release_locked(lease)
            finally:
                # Even if SQLite cannot release it now, a closed owner must
                # stop renewing so the durable lease can expire normally.
                if self._leases.owned.get(lease.session_id) == lease:
                    del self._leases.owned[lease.session_id]

    def append(
        self, session_id: str, message: Message, *, lease: SessionLease | None = None
    ) -> int:
        """Append one message and return its exact committed sequence."""
        if not is_session_id(session_id):
            raise SessionError(f"invalid session id: {session_id!r}")
        now = self._now()
        with self._write(session_id, lease):
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title, profile_name, created_at, updated_at)"
                " VALUES (?, '', ?, ?, ?)",
                (session_id, self._profile_name, now, now),
            )
            (seq,) = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self._conn.execute(
                "INSERT INTO messages (session_id, seq, role, created_at, payload)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, seq, message.role, now, message.model_dump_json()),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
            if message.role == "user":
                title = _message_title(message)
                if title:
                    self._conn.execute(
                        "UPDATE sessions SET title = ? WHERE id = ? AND title = ''",
                        (title, session_id),
                    )
        return int(seq)

    def rename(
        self, session_id: str, title: str, *, lease: SessionLease | None = None
    ) -> SessionMeta:
        title = " ".join(title.split())[:200]
        with self._write(session_id, lease):
            cur = self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )
            if cur.rowcount == 0:
                raise SessionError(f"no session with id {session_id!r}")
        meta = self.get(session_id)
        assert meta is not None
        return meta

    def delete(self, session_id: str, *, lease: SessionLease | None = None) -> bool:
        """Delete a session and (via cascade) its messages."""
        with self._write(session_id, lease, detach=True):
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def get(self, session_id: str) -> SessionMeta | None:
        with self._lock:
            row = self._conn.execute(
                _META_QUERY + "WHERE s.id = ?", (session_id,)
            ).fetchone()
        return _row_to_meta(row) if row else None

    def latest(self) -> SessionMeta | None:
        sessions = self.list(limit=1)
        return sessions[0] if sessions else None

    def list(self, *, limit: int = 200) -> list[SessionMeta]:
        """All sessions, most recently active first."""
        with self._lock:
            rows = self._conn.execute(
                _META_QUERY + "ORDER BY s.updated_at DESC, s.id LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_meta(r) for r in rows]

    def resolve_prefix(self, prefix: str) -> SessionMeta:
        """Resolve a unique id prefix (like a short git hash) to its session."""
        if not _PREFIX_RE.fullmatch(prefix):
            raise SessionError(f"invalid session id prefix: {prefix!r}")
        with self._lock:
            rows = self._conn.execute(
                _META_QUERY + "WHERE s.id LIKE ? || '%' LIMIT 6", (prefix,)
            ).fetchall()
        if not rows:
            raise SessionError(f"no session matching {prefix!r}")
        if len(rows) > 1:
            listing = "\n".join(f"  {r[0][:8]}  {r[1] or '(untitled)'}" for r in rows)
            raise SessionError(f"session id prefix {prefix!r} is ambiguous:\n{listing}")
        return _row_to_meta(rows[0])

    def message_records(
        self,
        session_id: str,
        *,
        after_seq: int = -1,
        through_seq: int | None = None,
    ) -> builtins.list[StoredMessage]:
        """Stored messages in a stable inclusive/exclusive sequence window."""
        with self._lock:
            return self._message_records_locked(
                session_id, after_seq=after_seq, through_seq=through_seq
            )

    def _message_records_locked(
        self,
        session_id: str,
        *,
        after_seq: int = -1,
        through_seq: int | None = None,
    ) -> builtins.list[StoredMessage]:
        """Stored messages in a stable inclusive/exclusive sequence window."""
        if after_seq < -1:
            raise SessionError("message replay cursor must be -1 or greater")
        if through_seq is not None and through_seq < 0:
            raise SessionError("message replay boundary must be non-negative")
        sql = "SELECT seq, payload FROM messages WHERE session_id = ? AND seq > ?"
        params: list[Any] = [session_id, after_seq]
        if through_seq is not None:
            sql += " AND seq <= ?"
            params.append(through_seq)
        sql += " ORDER BY seq"
        rows = self._conn.execute(sql, params).fetchall()
        out: list[StoredMessage] = []
        for seq, payload in rows:
            message = _stored_message(session_id, seq, payload)
            out.append(StoredMessage(seq=seq, message=message))
        return out

    def _canonical_message_suffix(
        self, session_id: str, *, through_seq: int, count: int
    ) -> builtins.list[Message]:
        with self._lock:
            return self._canonical_message_suffix_locked(
                session_id, through_seq=through_seq, count=count
            )

    def _canonical_message_suffix_locked(
        self, session_id: str, *, through_seq: int, count: int
    ) -> builtins.list[Message]:
        """Read only enough canonical rows to validate a working-set suffix.

        Rows are fetched newest-first in bounded batches. Once the validated
        history contains ``count`` messages, prepending still-earlier blocks
        cannot change that suffix, so large pre-compaction transcripts are not
        materialized during resume.
        """
        if count <= 0 or through_seq < 0:
            return []
        upper_seq = through_seq
        expected_seq = through_seq
        records: list[StoredMessage] = []
        batch_size = max(32, min(count * 2, 256))
        while True:
            rows = self._conn.execute(
                "SELECT seq, payload FROM messages "
                "WHERE session_id = ? AND seq <= ? "
                "ORDER BY seq DESC LIMIT ?",
                (session_id, upper_seq, batch_size),
            ).fetchall()
            if not rows:
                return []
            parsed_desc: list[StoredMessage] = []
            for seq, payload in rows:
                seq = int(seq)
                if seq != expected_seq:
                    return []  # a damaged sequence is not a canonical suffix
                expected_seq -= 1
                parsed_desc.append(
                    StoredMessage(
                        seq=seq,
                        message=_stored_message(session_id, seq, payload),
                    )
                )
            records = list(reversed(parsed_desc)) + records
            validated = _trim_dangling_records(records)
            if len(validated) >= count:
                return [record.message for record in validated[-count:]]
            upper_seq = int(rows[-1][0]) - 1

    def messages(self, session_id: str) -> builtins.list[Message]:
        """Stored messages in order; empty list when the session has no row."""
        return [record.message for record in self.message_records(session_id)]

    def next_sequence(self, session_id: str) -> int:
        """Sequence the next append will receive (0 for a fresh session)."""
        with self._lock:
            return self._next_sequence_locked(session_id)

    def _next_sequence_locked(self, session_id: str) -> int:
        """Sequence the next append will receive (0 for a fresh session)."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])

    def truncate_after(
        self, session_id: str, seq: int, *, lease: SessionLease | None = None
    ) -> int:
        """Delete messages after ``seq`` and return the number removed.

        This is the stop/cancel cleanup primitive: the current user message is
        kept while any partially committed assistant/tool tail is discarded.
        """
        with self._write(session_id, lease):
            cur = self._conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND seq > ?",
                (session_id, seq),
            )
            event_cur = self._conn.execute(
                "DELETE FROM session_events WHERE session_id = ? AND message_seq >= ?",
                (session_id, seq),
            )
            if cur.rowcount or event_cur.rowcount:
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (self._now(), session_id),
                )
            return cur.rowcount

    def rewind_to_user_message(
        self, session_id: str, seq: int, *, lease: SessionLease | None = None
    ) -> Message:
        """Delete ``seq`` and the later tail, returning the edited user message.

        Only an ordinary user message is a valid branch point. Synthetic user
        messages (tool-media hoists and compaction summaries) cannot be edited.
        The returned attachments let a frontend preserve them on regeneration.
        Success releases the lease: the old working set is invalid and the
        frontend must attach a new agent before regenerating the edited turn.
        """
        if seq < 0:
            raise SessionError("message sequence must be non-negative")
        with self._write(session_id, lease, detach=True):
            row = self._conn.execute(
                "SELECT payload FROM messages WHERE session_id = ? AND seq = ?",
                (session_id, seq),
            ).fetchone()
            if row is None:
                raise SessionError(f"no message {seq} in session {session_id!r}")
            try:
                target = Message.model_validate_json(row[0])
            except Exception as exc:
                raise SessionError(
                    f"cannot edit corrupt message row {session_id[:8]}/{seq}: {exc}"
                ) from None
            if target.role != "user" or target.name is not None:
                raise SessionError("only an ordinary user message can be edited")

            meta = self._conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            current_title = str(meta[0]) if meta is not None else ""
            earlier_user = self._conn.execute(
                "SELECT 1 FROM messages "
                "WHERE session_id = ? AND seq < ? AND role = 'user' LIMIT 1",
                (session_id, seq),
            ).fetchone()
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND seq >= ?",
                (session_id, seq),
            )
            self._conn.execute(
                "DELETE FROM session_events WHERE session_id = ? AND message_seq >= ?",
                (session_id, seq),
            )
            # If this was the auto-titled first prompt, clear it so appending the
            # edited prompt derives a fresh title. Preserve explicit renames.
            title = current_title
            if earlier_user is None and current_title == _message_title(target):
                title = ""
            self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, self._now(), session_id),
            )
        return target

    def close(self) -> None:
        self._leases.stop.set()
        if self._leases.thread is not None:
            self._leases.thread.join()
        with self._lock:
            if self._closed:
                return
            try:
                with self._conn:
                    self._conn.execute("BEGIN IMMEDIATE")
                    for lease in self._leases.owned.values():
                        self._leases.release_locked(lease)
            finally:
                self._closed = True
                self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _now() -> str:
        # The public module owns the clock so callers can inject deterministic
        # timestamps without depending on storage internals.
        from lingcore.sessions import _utcnow

        return _utcnow()
