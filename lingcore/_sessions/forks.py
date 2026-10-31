"""Session forks implementation."""

from __future__ import annotations

import json

from lingcore._sessions.events import SessionEvents
from lingcore._sessions.models import (
    _SESSION_EVENT_KINDS,
    CompactionEventPayload,
    CompactionSnapshot,
    SessionEvent,
    SessionMeta,
    _compaction_matches_canonical,
    _normalized_skill_event_payload,
    _session_meta,
    _stored_message,
    is_session_id,
    new_session_id,
    trim_dangling,
)
from lingcore.errors import SessionError
from lingcore.message import Message
from lingcore.todos import todos_from_payload, todos_payload
from lingcore.usage import turn_stats_from_payload, turn_stats_payload


def fork_session(
    self: SessionEvents,
    session_id: str,
    *,
    through_seq: int | None = None,
    title: str | None = None,
) -> SessionMeta:
    """Atomically copy a valid session prefix into a fresh session.

    ``through_seq`` is inclusive and remains the same stable coordinate in
    the destination. ``None`` copies the complete stored transcript. Only
    runtime events anchored inside the prefix and valid against its copied
    messages survive; their AUTOINCREMENT cursors are deliberately reminted.
    The source session is read under the same SQLite write transaction as
    the copy, so an in-process or second-process writer cannot move the
    selected boundary halfway through the operation.
    """
    if not is_session_id(session_id):
        raise SessionError(f"invalid session id: {session_id!r}")
    if through_seq is not None and (
        isinstance(through_seq, bool)
        or not isinstance(through_seq, int)
        or through_seq < 0
    ):
        raise SessionError("fork message sequence must be non-negative")
    requested_title: str | None = None
    if title is not None:
        requested_title = " ".join(title.split())[:200]
        if not requested_title:
            raise SessionError("fork title must not be empty")

    destination_id = new_session_id()
    now = self._now()
    with self._lock:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            source = self._conn.execute(
                "SELECT title, profile_name, meta FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if source is None:
                raise SessionError(f"no session with id {session_id!r}")

            latest = self._conn.execute(
                "SELECT MAX(seq) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            if latest is None:
                raise SessionError("cannot fork an empty session")
            boundary = int(latest) if through_seq is None else through_seq

            rows = self._conn.execute(
                "SELECT seq, created_at, payload FROM messages "
                "WHERE session_id = ? AND seq <= ? ORDER BY seq",
                (session_id, boundary),
            ).fetchall()
            if not rows or int(rows[-1][0]) != boundary:
                raise SessionError(f"no message {boundary} in session {session_id!r}")
            sequences = [int(row[0]) for row in rows]
            if sequences != list(range(boundary + 1)):
                raise SessionError(
                    f"cannot fork through message {boundary}: the source "
                    "message sequence is not contiguous"
                )

            copied: list[tuple[int, str, Message]] = [
                (
                    int(seq),
                    created_at,
                    _stored_message(session_id, int(seq), payload),
                )
                for seq, created_at, payload in rows
            ]
            messages = [message for _, _, message in copied]
            if trim_dangling(messages) != messages:
                raise SessionError(
                    f"cannot fork through message {boundary}: the prefix "
                    "contains an incomplete tool-call block"
                )
            by_seq = {seq: message for seq, _, message in copied}

            source_title, profile_name, raw_meta = source
            suffix = " (fork)"
            default_base = str(source_title) or f"Session {session_id[:8]}"
            fork_title = requested_title or (default_base[: 200 - len(suffix)] + suffix)
            source_meta = _session_meta(raw_meta)
            parent_fork = source_meta.get("fork")
            candidate_root = (
                parent_fork.get("root_session_id")
                if isinstance(parent_fork, dict)
                else None
            )
            root_id = (
                candidate_root
                if isinstance(candidate_root, str) and is_session_id(candidate_root)
                else session_id
            )
            destination_meta = dict(source_meta)
            destination_meta["fork"] = {
                "parent_session_id": session_id,
                "root_session_id": root_id,
                "through_seq": boundary,
            }
            meta_json = json.dumps(
                destination_meta,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._conn.execute(
                "INSERT INTO sessions "
                "(id, title, profile_name, created_at, updated_at, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    destination_id,
                    fork_title,
                    profile_name,
                    now,
                    now,
                    meta_json,
                ),
            )
            self._conn.executemany(
                "INSERT INTO messages "
                "(session_id, seq, role, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        destination_id,
                        seq,
                        message.role,
                        created_at,
                        message.model_dump_json(),
                    )
                    for seq, created_at, message in copied
                ],
            )

            event_rows = self._conn.execute(
                "SELECT event_seq, message_seq, kind, created_at, payload "
                "FROM session_events "
                "WHERE session_id = ? ORDER BY event_seq",
                (session_id,),
            ).fetchall()
            latest_compaction_cursor: int | None = None
            for (
                event_seq,
                message_seq,
                kind,
                created_at,
                payload_json,
            ) in event_rows:
                anchor = by_seq.get(message_seq)
                if kind not in _SESSION_EVENT_KINDS:
                    continue
                if anchor is None:
                    if kind != "skill_state":
                        continue
                    # A valid skill transition after the copied boundary is
                    # future branch state and must be ignored. An event whose
                    # claimed anchor does not exist anywhere is corrupt; carry
                    # an inactive tombstone so omitting it cannot revive an
                    # older authorization in the fork.
                    source_anchor = self._conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? AND seq = ?",
                        (session_id, message_seq),
                    ).fetchone()
                    if source_anchor is not None:
                        continue
                    normalized_payload = json.dumps(
                        {
                            "active": [],
                            "activated": [],
                            "deactivated": [],
                            "approved_high_risk": {},
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    copied_message_seq = boundary
                    copied_kind = "skill_state"
                    copied_created_at = now
                else:
                    copied_message_seq = int(message_seq)
                    copied_kind = str(kind)
                    copied_created_at = now
                    try:
                        payload = json.loads(payload_json)
                        if not isinstance(payload, dict):
                            raise ValueError("payload is not an object")
                        event = SessionEvent(
                            event_seq=event_seq,
                            message_seq=message_seq,
                            kind=kind,
                            created_at=created_at,
                            payload=payload,
                        )
                        copied_message_seq = event.message_seq
                        copied_kind = event.kind
                        copied_created_at = event.created_at.isoformat()
                        if event.kind == "compaction":
                            if "messages" in event.payload:
                                snapshot = CompactionSnapshot.model_validate(
                                    {
                                        **event.payload,
                                        "event_seq": event.event_seq,
                                        "message_seq": event.message_seq,
                                    }
                                )
                                if snapshot.snapshot_superseded:
                                    continue
                                canonical_prefix = messages[: event.message_seq + 1]
                                if not _compaction_matches_canonical(
                                    snapshot.messages, canonical_prefix
                                ):
                                    continue
                            else:
                                metadata = CompactionEventPayload.model_validate(
                                    event.payload
                                )
                                if not metadata.snapshot_superseded:
                                    continue
                        elif event.kind == "todo_state":
                            todos = todos_from_payload(event.payload)
                            if todos is None:
                                raise ValueError("invalid todo-state payload")
                            event.payload = todos_payload(todos)
                        elif event.kind == "turn_stats":
                            stats = turn_stats_from_payload(event.payload)
                            if stats is None:
                                raise ValueError("invalid turn-stats payload")
                            event.payload = turn_stats_payload(stats)
                        else:
                            normalized = _normalized_skill_event_payload(event.payload)
                            if normalized is None:
                                raise ValueError("invalid skill-state payload")
                            event.payload = normalized
                        normalized_payload = json.dumps(
                            event.payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    except Exception:
                        if kind == "skill_state":
                            # Skill rows are complete state snapshots. Dropping a
                            # corrupt deactivation would revive an older active
                            # skill in the fork, so carry an inactive tombstone
                            # instead (fail closed on authorization ambiguity).
                            normalized_payload = json.dumps(
                                {
                                    "active": [],
                                    "activated": [],
                                    "deactivated": [],
                                    "approved_high_risk": {},
                                },
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            copied_message_seq = int(message_seq)
                            copied_kind = "skill_state"
                            copied_created_at = now
                        else:
                            # A fork carries valid derived state only. The canonical
                            # messages remain copyable even if one source event is
                            # corrupt or from an incompatible older release.
                            continue
                inserted = self._conn.execute(
                    "INSERT INTO session_events "
                    "(session_id, message_seq, kind, created_at, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        destination_id,
                        copied_message_seq,
                        copied_kind,
                        copied_created_at,
                        normalized_payload,
                    ),
                )
                if inserted.lastrowid is None:
                    raise SessionError("forked event insert did not return a cursor")
                if copied_kind == "compaction":
                    latest_compaction_cursor = int(inserted.lastrowid)
            if latest_compaction_cursor is not None:
                # A source created before snapshot pruning may contain many
                # full bodies. Forking must not duplicate that unbounded
                # derived state into the destination.
                self._prune_old_compaction_snapshots_locked(
                    destination_id,
                    current_event_seq=latest_compaction_cursor,
                )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    forked = self.get(destination_id)
    assert forked is not None
    return forked
