"""SQLite session ownership; all *_locked operations require BEGIN IMMEDIATE."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
import weakref
from dataclasses import dataclass

from lingcore.errors import SessionError

LEASE_SECONDS = 60.0
HEARTBEAT_SECONDS = 15.0
LEASE_DDL = """
CREATE TABLE IF NOT EXISTS session_leases (
    session_id TEXT PRIMARY KEY,
    owner_token TEXT,
    epoch INTEGER NOT NULL,
    heartbeat REAL NOT NULL
)
"""


@dataclass(frozen=True)
class SessionLease:
    session_id: str
    owner_token: str
    epoch: int


class SessionLeases:
    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock) -> None:
        self.connection = connection
        self.lock = lock
        self.owned: dict[str, SessionLease] = {}
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def acquire_locked(self, session_id: str) -> SessionLease:
        now = time.time()
        row = self.connection.execute(
            "SELECT owner_token, epoch, heartbeat FROM session_leases WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is not None and row[0] is not None and row[2] > now - LEASE_SECONDS:
            raise SessionError(
                f"session {session_id[:8]} is already attached to another agent; "
                "close its frontend or use fork_session to continue in a new session"
            )
        lease = SessionLease(session_id, uuid.uuid4().hex, row[1] + 1 if row else 1)
        self.connection.execute(
            "INSERT INTO session_leases VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET owner_token=excluded.owner_token, "
            "epoch=excluded.epoch, heartbeat=excluded.heartbeat",
            (session_id, lease.owner_token, lease.epoch, now),
        )
        return lease

    def check_locked(self, session_id: str, lease: SessionLease | None) -> None:
        lease = lease if lease is not None else self.owned.get(session_id)
        if lease is not None:
            if lease.session_id != session_id or not self.renew_locked(lease):
                raise SessionError(
                    f"session {session_id[:8]} lease was lost; close this agent and "
                    "reopen the session, or use fork_session"
                )
            return
        # One-off administrative writes are allowed only without a live owner.
        # The surrounding write transaction prevents an attach until commit.
        now = time.time()
        row = self.connection.execute(
            "SELECT owner_token, heartbeat FROM session_leases WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is not None and row[0] is not None:
            if row[1] > now - LEASE_SECONDS:
                raise SessionError(
                    f"session {session_id[:8]} is owned by another agent; "
                    "close its frontend or use fork_session"
                )
            self.connection.execute(
                "UPDATE session_leases SET owner_token=NULL, heartbeat=0 "
                "WHERE session_id = ?",
                (session_id,),
            )

    def renew_locked(self, lease: SessionLease) -> bool:
        now = time.time()
        return (
            self.connection.execute(
                "UPDATE session_leases SET heartbeat=? WHERE session_id=? "
                "AND owner_token=? AND epoch=? AND heartbeat>?",
                (
                    now,
                    lease.session_id,
                    lease.owner_token,
                    lease.epoch,
                    now - LEASE_SECONDS,
                ),
            ).rowcount
            == 1
        )

    def release_locked(self, lease: SessionLease) -> None:
        # Keep the epoch tombstone, including after session deletion. A stale
        # close must never release a replacement owner's lease.
        self.connection.execute(
            "UPDATE session_leases SET owner_token=NULL, heartbeat=0 "
            "WHERE session_id=? AND owner_token=? AND epoch=?",
            (lease.session_id, lease.owner_token, lease.epoch),
        )

    def start_heartbeat(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(
                target=self._heartbeat,
                args=(weakref.ref(self), self.stop),
                name="lingcore-session-heartbeat",
                daemon=True,
            )
            self.thread.start()

    @staticmethod
    def _heartbeat(
        reference: weakref.ReferenceType[SessionLeases], stop: threading.Event
    ) -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            manager = reference()
            if manager is None:
                return
            try:
                with manager.lock, manager.connection:
                    if stop.is_set():
                        return
                    manager.connection.execute("BEGIN IMMEDIATE")
                    for lease in manager.owned.values():
                        # Expired ownership is never revived. Its captured token
                        # remains available so subsequent writes fail closed.
                        manager.renew_locked(lease)
            except sqlite3.Error:
                # A temporary database outage may recover within the lease.
                # Every write independently checks expiry and ownership.
                pass
            finally:
                del manager
