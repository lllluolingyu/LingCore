"""Session events implementation."""

from __future__ import annotations

import builtins
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from lingcore._sessions.database import SessionDatabase
from lingcore._sessions.models import (
    _SESSION_EVENT_KINDS,
    CompactionSnapshot,
    PersistedSkillState,
    SessionEvent,
    _compaction_tail,
    _skill_state_payload,
    _trim_dangling_records,
    is_session_id,
)
from lingcore.errors import SessionError
from lingcore.message import Message
from lingcore.session_leases import SessionLease
from lingcore.todos import TodoItem, todos_from_payload, todos_payload
from lingcore.usage import TurnStats, turn_stats_from_payload, turn_stats_payload


class SessionEvents(SessionDatabase):
    """Durable runtime event persistence and validated state replay."""

    def append_event(
        self,
        session_id: str,
        *,
        message_seq: int,
        kind: str,
        payload: dict[str, Any],
        lease: SessionLease | None = None,
    ) -> SessionEvent:
        """Append one replayable runtime event and return its monotonic cursor."""
        with self._write(session_id, lease):
            return self._append_event_locked(
                session_id, message_seq=message_seq, kind=kind, payload=payload
            )

    def _append_event_locked(
        self,
        session_id: str,
        *,
        message_seq: int,
        kind: str,
        payload: dict[str, Any],
    ) -> SessionEvent:
        """Append one replayable runtime event and return its monotonic cursor."""
        if not is_session_id(session_id):
            raise SessionError(f"invalid session id: {session_id!r}")
        if kind not in _SESSION_EVENT_KINDS:
            raise SessionError(f"unsupported session event kind: {kind!r}")
        if (
            isinstance(message_seq, bool)
            or not isinstance(message_seq, int)
            or message_seq < 0
        ):
            raise SessionError("session event message sequence must be non-negative")
        try:
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            normalized_payload = json.loads(payload_json)
        except (TypeError, ValueError) as exc:
            raise SessionError(
                f"session event payload is not JSON-compatible: {exc}"
            ) from None
        if not isinstance(normalized_payload, dict):
            raise SessionError("session event payload must be an object")

        now = self._now()
        anchor = self._conn.execute(
            "SELECT 1 FROM messages WHERE session_id = ? AND seq = ?",
            (session_id, message_seq),
        ).fetchone()
        if anchor is None:
            raise SessionError(
                f"cannot anchor session event to missing message {message_seq}"
            )
        cur = self._conn.execute(
            "INSERT INTO session_events "
            "(session_id, message_seq, kind, created_at, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, message_seq, kind, now, payload_json),
        )
        if cur.lastrowid is None:
            raise SessionError("session event insert did not return a cursor")
        event_seq = int(cur.lastrowid)
        if kind == "compaction":
            self._prune_old_compaction_snapshots_locked(
                session_id, current_event_seq=event_seq
            )
        return SessionEvent(
            event_seq=event_seq,
            message_seq=message_seq,
            kind=kind,
            created_at=datetime.fromisoformat(now),
            payload=normalized_payload,
        )

    def _prune_old_compaction_snapshots_locked(
        self, session_id: str, *, current_event_seq: int
    ) -> None:
        """Retain full bodies for only the two newest compaction snapshots.

        Historical compaction rows remain replayable (counts and cursor stay
        intact), but their copied working sets may contain large base64 payloads
        and are no longer needed for resume once two newer fallbacks exist.
        Called inside the append/fork lock and transaction.
        """
        retained = self._conn.execute(
            "SELECT event_seq FROM session_events "
            "WHERE session_id = ? AND kind = 'compaction' AND event_seq <= ? "
            "AND payload LIKE '%\"messages\":%' "
            "ORDER BY event_seq DESC LIMIT 2",
            (session_id, current_event_seq),
        ).fetchall()
        if len(retained) < 2:
            return
        oldest_retained_event_seq = int(retained[-1][0])
        cursor = -1
        while True:
            row = self._conn.execute(
                "SELECT event_seq, payload FROM session_events "
                "WHERE session_id = ? AND kind = 'compaction' "
                "AND event_seq > ? AND event_seq < ? "
                "AND payload LIKE '%\"messages\":%' "
                "ORDER BY event_seq LIMIT 1",
                (session_id, cursor, oldest_retained_event_seq),
            ).fetchone()
            if row is None:
                return
            event_seq, payload_json = int(row[0]), row[1]
            cursor = event_seq
            try:
                payload = json.loads(payload_json)
                if not isinstance(payload, dict):
                    raise ValueError("payload is not an object")
                payload.pop("messages", None)
                payload["snapshot_superseded"] = True
            except Exception:
                # Derived state is replaceable. A corrupt obsolete body should
                # not retain arbitrary disk usage or become a resume candidate.
                payload = {"snapshot_superseded": True}
            self._conn.execute(
                "UPDATE session_events SET payload = ? WHERE event_seq = ?",
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    event_seq,
                ),
            )

    def save_compaction(
        self,
        session_id: str,
        *,
        messages: list[Message],
        summarized_messages: int,
        before_tokens: int,
        after_tokens: int,
        turn_index: int | None = None,
        lease: SessionLease | None = None,
    ) -> SessionEvent:
        """Persist a compacted working set at the current transcript watermark."""
        with self._write(session_id, lease):
            message_seq = self._next_sequence_locked(session_id) - 1
            if message_seq < 0:
                raise SessionError("cannot save a compaction before the first message")
            tail = _compaction_tail(messages)
            if tail is None:
                raise SessionError(
                    "compaction snapshot must contain one synthetic summary followed "
                    "by a complete canonical message tail"
                )
            canonical_suffix = self._canonical_message_suffix_locked(
                session_id,
                through_seq=message_seq,
                count=len(tail),
            )
            if canonical_suffix != tail:
                raise SessionError(
                    "compaction snapshot tail does not match the canonical transcript"
                )
            if turn_index is None:
                # Public/manual callers do not have Agent._turn_index. Derive it from
                # the same validated history resume uses, never from raw dangling
                # assistant rows left by a crashed turn.
                records = self._message_records_locked(
                    session_id, through_seq=message_seq
                )
                turn_index = sum(
                    record.message.role == "assistant"
                    for record in _trim_dangling_records(records)
                )
            if (
                isinstance(turn_index, bool)
                or not isinstance(turn_index, int)
                or turn_index < 0
            ):
                raise SessionError("compaction turn index must be non-negative")
            for label, value, minimum in (
                ("summarized message count", summarized_messages, 1),
                ("before token count", before_tokens, 0),
                ("after token count", after_tokens, 0),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < minimum
                ):
                    raise SessionError(f"compaction {label} must be at least {minimum}")
            return self._append_event_locked(
                session_id,
                message_seq=message_seq,
                kind="compaction",
                payload={
                    "messages": [
                        message.model_dump(mode="json") for message in messages
                    ],
                    "turn_index": turn_index,
                    "summarized_messages": summarized_messages,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                },
            )

    def save_skill_state(
        self,
        session_id: str,
        *,
        active: list[str],
        activated: list[str],
        deactivated: list[str],
        approved_high_risk: dict[str, list[str]] | None = None,
        lease: SessionLease | None = None,
    ) -> SessionEvent:
        """Persist the complete dynamic-skill state plus its visible transition."""
        with self._write(session_id, lease):
            message_seq = self._next_sequence_locked(session_id) - 1
            if message_seq < 0:
                raise SessionError("cannot save skill state before the first message")
            for label, names in (
                ("active", active),
                ("activated", activated),
                ("deactivated", deactivated),
            ):
                if not isinstance(names, list) or not all(
                    isinstance(name, str) and name for name in names
                ):
                    raise SessionError(
                        f"{label} skill names must be a list of non-empty strings"
                    )
            active_names = list(dict.fromkeys(active))
            if approved_high_risk is not None and not isinstance(
                approved_high_risk, dict
            ):
                raise SessionError("approved high-risk skill grants must be an object")
            normalized_approvals: dict[str, list[str]] = {}
            for name, tools in (approved_high_risk or {}).items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(tools, list)
                    or not all(isinstance(tool, str) and tool for tool in tools)
                ):
                    raise SessionError(
                        "approved high-risk skill grants must be string lists"
                    )
                if name in active_names:
                    normalized_approvals[name] = sorted(set(tools))
            return self._append_event_locked(
                session_id,
                message_seq=message_seq,
                kind="skill_state",
                payload={
                    "active": active_names,
                    "activated": list(dict.fromkeys(activated)),
                    "deactivated": list(dict.fromkeys(deactivated)),
                    "approved_high_risk": normalized_approvals,
                },
            )

    def save_todo_state(
        self,
        session_id: str,
        todos: Sequence[TodoItem],
        *,
        lease: SessionLease | None = None,
    ) -> SessionEvent:
        """Persist the complete todo checklist, anchored to the latest message.

        Rows are complete snapshots (each bounded by the tool's item and
        content caps), so rewinding a branch restores the previous list simply
        by removing the later rows.
        """
        with self._write(session_id, lease):
            message_seq = self._next_sequence_locked(session_id) - 1
            if message_seq < 0:
                raise SessionError("cannot save todo state before the first message")
            return self._append_event_locked(
                session_id,
                message_seq=message_seq,
                kind="todo_state",
                payload=todos_payload(todos),
            )

    def save_turn_stats(
        self,
        session_id: str,
        message_seq: int,
        stats: TurnStats,
        *,
        lease: SessionLease | None = None,
    ) -> SessionEvent:
        """Persist one turn's usage and timing, anchored to its user message.

        The anchor is the user row the turn kept, so Stop's truncation (which
        runs first) leaves it in place, while Edit removes it with the row.
        """
        return self.append_event(
            session_id,
            message_seq=message_seq,
            kind="turn_stats",
            lease=lease,
            payload=turn_stats_payload(stats),
        )

    def turn_stats(self, session_id: str) -> dict[int, TurnStats]:
        """Valid persisted turn stats by user-message sequence (newest wins)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.message_seq, e.payload FROM session_events e "
                "WHERE e.session_id = ? AND e.kind = 'turn_stats' "
                "AND EXISTS(SELECT 1 FROM messages m "
                "WHERE m.session_id = e.session_id AND m.seq = e.message_seq) "
                "ORDER BY e.event_seq",
                (session_id,),
            ).fetchall()
        out: dict[int, TurnStats] = {}
        for message_seq, payload_json in rows:
            try:
                stats = turn_stats_from_payload(json.loads(payload_json))
            except ValueError:
                continue
            if stats is not None:
                out[int(message_seq)] = stats
        return out

    def latest_todos(self, session_id: str) -> tuple[TodoItem, ...]:
        """Newest persisted todo checklist; empty when absent or corrupt.

        Unlike skill state this grants nothing, so a corrupt latest row simply
        yields an empty list rather than needing a fail-closed tombstone.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT e.payload FROM session_events e "
                "WHERE e.session_id = ? AND e.kind = 'todo_state' "
                "AND EXISTS(SELECT 1 FROM messages m "
                "WHERE m.session_id = e.session_id AND m.seq = e.message_seq) "
                "ORDER BY e.event_seq DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return ()
        try:
            return todos_from_payload(json.loads(row[0])) or ()
        except ValueError:
            return ()

    def events(
        self,
        session_id: str,
        *,
        after_seq: int = -1,
        kind: str | None = None,
    ) -> builtins.list[SessionEvent]:
        """Durable runtime events after a monotonic replay cursor."""
        if after_seq < -1:
            raise SessionError("session event replay cursor must be -1 or greater")
        if kind is not None and kind not in _SESSION_EVENT_KINDS:
            raise SessionError(f"unsupported session event kind: {kind!r}")
        sql = (
            "SELECT event_seq, message_seq, kind, created_at, payload "
            "FROM session_events WHERE session_id = ? AND event_seq > ?"
        )
        params: list[Any] = [session_id, after_seq]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY event_seq"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: list[SessionEvent] = []
        for event_seq, message_seq, event_kind, created_at, payload_json in rows:
            try:
                payload = json.loads(payload_json)
                if not isinstance(payload, dict):
                    raise ValueError("payload is not an object")
                if event_kind not in _SESSION_EVENT_KINDS:
                    raise ValueError(f"unsupported kind {event_kind!r}")
                out.append(
                    SessionEvent(
                        event_seq=event_seq,
                        message_seq=message_seq,
                        kind=event_kind,
                        created_at=created_at,
                        payload=payload,
                    )
                )
            except Exception:
                # Runtime events are derived state, not the canonical
                # transcript. A damaged row must not make session history (or
                # later valid snapshots) unusable; omit it and let cursors
                # advance past it via ``event_cursor``.
                continue
        return out

    def event_cursor(self, session_id: str) -> int:
        """Latest surviving event cursor, or ``-1`` when there are no events."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(event_seq), -1) FROM session_events "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row[0])

    def latest_compaction(self, session_id: str) -> CompactionSnapshot | None:
        """Newest valid compaction snapshot, falling back past corrupt snapshots."""
        before_event_seq: int | None = None
        while True:
            sql = (
                "SELECT event_seq, message_seq, created_at, payload "
                "FROM session_events "
                "WHERE session_id = ? AND kind = 'compaction' "
                "AND payload LIKE '%\"messages\":%'"
            )
            params: list[Any] = [session_id]
            if before_event_seq is not None:
                sql += " AND event_seq < ?"
                params.append(before_event_seq)
            sql += " ORDER BY event_seq DESC LIMIT 1"
            with self._lock:
                row = self._conn.execute(sql, params).fetchone()
            if row is None:
                return None
            event_seq, message_seq, _created_at, payload_json = row
            before_event_seq = int(event_seq)
            try:
                payload = json.loads(payload_json)
                if not isinstance(payload, dict):
                    continue
                snapshot = CompactionSnapshot.model_validate(
                    {
                        **payload,
                        "event_seq": event_seq,
                        "message_seq": message_seq,
                    }
                )
                if snapshot.snapshot_superseded:
                    continue
                tail = _compaction_tail(snapshot.messages)
                if tail is None:
                    continue
                canonical_suffix = self._canonical_message_suffix(
                    session_id,
                    through_seq=snapshot.message_seq,
                    count=len(tail),
                )
                if canonical_suffix == tail:
                    return snapshot
            except Exception:
                continue

    def active_skill_state(self, session_id: str) -> PersistedSkillState:
        """Latest complete skill state, failing closed if that row is corrupt."""
        with self._lock:
            row = self._conn.execute(
                "SELECT e.payload, EXISTS("
                "SELECT 1 FROM messages m "
                "WHERE m.session_id = e.session_id AND m.seq = e.message_seq"
                ") FROM session_events e "
                "WHERE e.session_id = ? AND e.kind = 'skill_state' "
                "ORDER BY e.event_seq DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None or not row[1]:
            return PersistedSkillState()
        try:
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                return PersistedSkillState()
            return _skill_state_payload(payload) or PersistedSkillState()
        except Exception:
            # A corrupt latest deactivation must never revive an older active
            # skill (which could re-authorize high-risk tools).
            return PersistedSkillState()

    def active_skills(self, session_id: str) -> tuple[str, ...]:
        """Latest persisted dynamic-skill names, preserving activation order."""
        return self.active_skill_state(session_id).active

    def discard_events_from_message_seq(
        self, session_id: str, seq: int, *, lease: SessionLease | None = None
    ) -> int:
        """Discard runtime events anchored at or after a removed branch point."""
        with self._write(session_id, lease):
            cur = self._conn.execute(
                "DELETE FROM session_events WHERE session_id = ? AND message_seq >= ?",
                (session_id, seq),
            )
            return cur.rowcount
