"""Session history storage — persist conversations per profile and resume them.

A ``SessionStore`` is a small SQLite database (stdlib ``sqlite3``, WAL mode)
that lives *in the profile directory* by default, so every profile keeps its
own separate history — exactly like the memory tool's ``memory.md``. The same
confinement rules apply (invariant 12): a relative ``sessions.path`` may not
escape the profile directory, an absolute one requires
``allow_absolute_path: true``, and a path inside the installed package tree
disables persistence gracefully rather than erroring.

Design notes:

- **Rows are lazy.** ``append`` creates the session row in the same
  transaction as the first message, so an opened-but-never-used session id
  leaves nothing behind (no empty rows from reconnects or instant exits).
- **Messages are stored verbatim** as ``Message.model_dump_json()`` plus a
  denormalized ``role`` column. ``ToolResult.ok`` is not part of ``Message``;
  the loop encodes failures as a ``"ERROR: "`` content prefix (agent.py), and
  display layers may use that same convention.
- **Loading is block-aware.** ``trim_dangling`` drops any assistant message
  whose ``tool_calls`` were never fully answered (a crashed turn) — anywhere
  in the list, not just the tail — because OpenAI rejects unanswered
  ``tool_calls`` and orphaned ``tool`` results alike. A trailing lone user
  message is kept: it is a valid prefix and real user input.
- **Runtime events are durable.** Compaction snapshots and dynamic skill-state
  transitions are anchored to the latest stored message. Resume restores the
  newest valid compaction snapshot, replays its message tail, and restores the
  latest compatible skill state. Rewind/Stop discard events on the removed
  branch. Event ids are monotonic cursors suitable for frontend replay.
- **Forks are explicit copies.** A fork atomically copies one valid transcript
  prefix and the runtime events anchored inside it into a fresh session. Event
  cursors are reminted (never shared between sessions), while provenance in
  ``sessions.meta`` records the immediate parent, root, and source boundary.
- **Still deliberately ephemeral:** per-session "allow always" shell patterns
  are not persisted (re-approval after resume is the conservative default),
  and attach-exclusivity is enforced per process, not across processes.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lingcore.errors import ConfigError, SessionError
from lingcore.message import Message

if TYPE_CHECKING:
    from lingcore.config import AgentProfile
    from lingcore.memory import ShortTermMemory, WindowMemory

# The installed package root — a sessions.db inside it is refused (gracefully).
_PACKAGE_DIR = Path(__file__).parent.resolve()

_SCHEMA_VERSION = 2
_TITLE_LIMIT = 60
_ID_RE = re.compile(r"[0-9a-f]{32}")
_PREFIX_RE = re.compile(r"[0-9a-f]{1,32}")
_SESSION_EVENT_KINDS = frozenset({"compaction", "skill_state"})

_DDL = [
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


def new_session_id() -> str:
    return uuid.uuid4().hex


def is_session_id(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _derive_title(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    title = " ".join(first_line.split())
    return title if len(title) <= _TITLE_LIMIT else title[: _TITLE_LIMIT - 1] + "…"


def _message_title(message: Message) -> str:
    if message.name:
        return ""
    title_source = (
        message.input_text if message.input_text is not None else message.content
    )
    title = _derive_title(title_source)
    if title:
        return title
    names = [a.name for a in message.attachments if a.name]
    if not names:
        return ""
    joined = ", ".join(names[:3])
    if len(names) > 3:
        joined += f", +{len(names) - 3} more"
    return _derive_title(joined)


def _strip_invalid_attachments(payload: str) -> Message | None:
    """Re-parse a stored row without its attachments, marking the loss.

    Returns None when the row is broken beyond its attachments (truly corrupt
    JSON or invalid core fields) — the caller then raises SessionError as
    before.
    """
    try:
        raw = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    attachments = raw.get("attachments")
    if not isinstance(attachments, list) or not attachments:
        return None
    dropped = len(attachments)
    raw["attachments"] = []
    try:
        message = Message.model_validate(raw)
    except Exception:
        return None
    suffix = f" [{dropped} stored attachment(s) no longer pass validation; dropped on load]"
    message.content = (message.content + suffix).strip()
    if message.input_text is not None:
        message.input_text = (message.input_text + suffix).strip()
    return message


def _stored_message(session_id: str, seq: int, payload: str) -> Message:
    """Validate one canonical message row, salvaging stale attachments only."""
    try:
        return Message.model_validate_json(payload)
    except Exception as exc:
        # Attachment validation may tighten between releases. Preserve the text
        # while dropping only media that no longer validates; malformed core
        # message data still fails loudly.
        salvaged = _strip_invalid_attachments(payload)
        if salvaged is not None:
            return salvaged
        raise SessionError(
            f"corrupt message row {session_id[:8]}/{seq}: {exc}"
        ) from None


def _session_meta(payload: str) -> dict[str, Any]:
    """Decode optional session metadata without bricking canonical history."""
    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class SessionForkMeta(BaseModel):
    """Public, validated lineage for an explicitly forked session."""

    parent_session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    root_session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    through_seq: int = Field(ge=0)


class SessionMeta(BaseModel):
    """Summary row for one stored session."""

    id: str
    title: str = ""
    profile_name: str = ""
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    fork: SessionForkMeta | None = None


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """One stored message together with its stable sequence in the session."""

    seq: int
    message: Message


class SessionEvent(BaseModel):
    """A durable runtime event anchored to a stored transcript message."""

    event_seq: int
    message_seq: int
    kind: str
    created_at: datetime
    payload: dict[str, Any]


class CompactionEventPayload(BaseModel):
    """Replayable compaction metadata, with or without a retained snapshot."""

    turn_index: int = Field(ge=0, strict=True)
    summarized_messages: int = Field(ge=1, strict=True)
    before_tokens: int = Field(ge=0, strict=True)
    after_tokens: int = Field(ge=0, strict=True)
    snapshot_superseded: bool = False


class CompactionSnapshot(CompactionEventPayload):
    """Validated working-set snapshot used as the base for resume replay."""

    event_seq: int = Field(ge=1, strict=True)
    message_seq: int = Field(ge=0, strict=True)
    messages: list[Message] = Field(min_length=2)


@dataclass(frozen=True, slots=True)
class PersistedSkillState:
    """Fail-closed dynamic-skill state reconstructed from one durable event."""

    active: tuple[str, ...] = ()
    approved_high_risk: dict[str, frozenset[str]] = field(default_factory=dict)

    def approvals_for(self, name: str) -> frozenset[str]:
        return self.approved_high_risk.get(name, frozenset())


def _skill_state_payload(payload: dict[str, Any]) -> PersistedSkillState | None:
    """Validate a complete skill-state payload.

    Missing approval metadata is the schema-v2 legacy case. It is represented
    as an empty map so restoration can still allow low-risk skills while
    refusing any high-risk grant that lacks recorded consent.
    """
    active = payload.get("active")
    if not isinstance(active, list) or not all(
        isinstance(name, str) and name for name in active
    ):
        return None
    active_names = tuple(dict.fromkeys(active))
    raw_approvals = payload.get("approved_high_risk", {})
    approvals: dict[str, frozenset[str]] = {}
    if isinstance(raw_approvals, dict):
        valid = True
        for name, tools in raw_approvals.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(tools, list)
                or not all(isinstance(tool, str) and tool for tool in tools)
            ):
                valid = False
                break
            if name in active_names:
                approvals[name] = frozenset(tools)
        if not valid:
            # Corrupt consent metadata must never authorize a risky restore.
            approvals = {}
    # A non-object approval field is likewise fail-closed, while the complete
    # active list remains useful for restoring low-risk skills.
    return PersistedSkillState(
        active=active_names,
        approved_high_risk=approvals,
    )


def _normalized_skill_event_payload(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate both the complete state and its display-only transition."""
    state = _skill_state_payload(payload)
    if state is None:
        return None
    normalized: dict[str, Any] = {
        "active": list(state.active),
        "approved_high_risk": {
            name: sorted(tools)
            for name, tools in state.approved_high_risk.items()
        },
    }
    for key in ("activated", "deactivated"):
        names = payload.get(key)
        if not isinstance(names, list) or not all(
            isinstance(name, str) and name for name in names
        ):
            return None
        normalized[key] = list(dict.fromkeys(names))
    return normalized


def _compaction_tail(messages: list[Message]) -> list[Message] | None:
    """Return a structurally valid raw tail after the synthetic summary."""
    if len(messages) < 2 or trim_dangling(messages) != messages:
        return None
    summary = messages[0]
    if (
        summary.role != "user"
        or summary.name != "summary"
        or not summary.content.strip()
        or summary.input_text is not None
        or summary.tool_calls
        or summary.tool_call_id is not None
        or summary.attachments
    ):
        return None
    return messages[1:]


def _compaction_matches_canonical(
    snapshot_messages: list[Message], canonical_messages: list[Message]
) -> bool:
    """Require the non-summary portion to be an exact canonical suffix."""
    tail = _compaction_tail(snapshot_messages)
    canonical = trim_dangling(canonical_messages)
    return (
        tail is not None
        and len(canonical) >= len(tail)
        and canonical[-len(tail) :] == tail
    )


_META_QUERY = """
SELECT s.id, s.title, s.profile_name, s.created_at, s.updated_at, s.meta,
       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n
FROM sessions s
"""


class SessionStore:
    """SQLite-backed session history for one profile.

    A single connection guarded by a lock; safe to share across the
    connections of one process (e.g. all LingChat WebSockets). WAL +
    busy_timeout keep a CLI and a web server on the same file consistent.
    All methods are synchronous on purpose: single-row writes are
    microsecond-scale, matching the blocking file writes async tools
    already do.
    """

    def __init__(self, db_path: Path, *, profile_name: str = "") -> None:
        self.db_path = db_path
        self._profile_name = profile_name
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
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
                    for stmt in _DDL:
                        self._conn.execute(stmt)
                    self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        except BaseException:
            self._conn.close()
            raise

    # -- write ----------------------------------------------------------
    def append(self, session_id: str, message: Message) -> int:
        """Append one message and return its exact committed sequence."""
        if not is_session_id(session_id):
            raise SessionError(f"invalid session id: {session_id!r}")
        now = _utcnow()
        with self._lock, self._conn:
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

    def append_event(
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
            raise SessionError(f"session event payload is not JSON-compatible: {exc}") from None
        if not isinstance(normalized_payload, dict):
            raise SessionError("session event payload must be an object")

        now = _utcnow()
        with self._lock, self._conn:
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
            event_seq = int(cur.lastrowid)
            if kind == "compaction":
                self._prune_old_compaction_snapshots_locked(
                    session_id, current_event_seq=event_seq
                )
        return SessionEvent(
            event_seq=event_seq,
            message_seq=message_seq,
            kind=kind,
            created_at=now,
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
    ) -> SessionEvent:
        """Persist a compacted working set at the current transcript watermark."""
        message_seq = self.next_sequence(session_id) - 1
        if message_seq < 0:
            raise SessionError("cannot save a compaction before the first message")
        tail = _compaction_tail(messages)
        if tail is None:
            raise SessionError(
                "compaction snapshot must contain one synthetic summary followed "
                "by a complete canonical message tail"
            )
        canonical_suffix = self._canonical_message_suffix(
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
            records = self.message_records(session_id, through_seq=message_seq)
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
        return self.append_event(
            session_id,
            message_seq=message_seq,
            kind="compaction",
            payload={
                "messages": [message.model_dump(mode="json") for message in messages],
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
    ) -> SessionEvent:
        """Persist the complete dynamic-skill state plus its visible transition."""
        message_seq = self.next_sequence(session_id) - 1
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
        return self.append_event(
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

    def fork_session(
        self,
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
        now = _utcnow()
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
                    raise SessionError(
                        f"no message {boundary} in session {session_id!r}"
                    )
                sequences = [int(row[0]) for row in rows]
                if sequences != list(range(boundary + 1)):
                    raise SessionError(
                        f"cannot fork through message {boundary}: the source "
                        "message sequence is not contiguous"
                    )

                copied: list[tuple[int, str, Message]] = [
                    (int(seq), created_at, _stored_message(session_id, int(seq), payload))
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
                fork_title = requested_title or (
                    default_base[: 200 - len(suffix)] + suffix
                )
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
                for event_seq, message_seq, kind, created_at, payload_json in event_rows:
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
                            "SELECT 1 FROM messages "
                            "WHERE session_id = ? AND seq = ?",
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
                                    snapshot = CompactionSnapshot.model_validate({
                                        **event.payload,
                                        "event_seq": event.event_seq,
                                        "message_seq": event.message_seq,
                                    })
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
                            else:
                                normalized = _normalized_skill_event_payload(
                                    event.payload
                                )
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

    def rename(self, session_id: str, title: str) -> SessionMeta:
        title = " ".join(title.split())[:200]
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )
            if cur.rowcount == 0:
                raise SessionError(f"no session with id {session_id!r}")
        meta = self.get(session_id)
        assert meta is not None
        return meta

    def delete(self, session_id: str) -> bool:
        """Delete a session and (via cascade) its messages."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            return cur.rowcount > 0

    # -- read -----------------------------------------------------------
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
            raise SessionError(
                f"session id prefix {prefix!r} is ambiguous:\n{listing}"
            )
        return _row_to_meta(rows[0])

    def message_records(
        self,
        session_id: str,
        *,
        after_seq: int = -1,
        through_seq: int | None = None,
    ) -> list[StoredMessage]:
        """Stored messages in a stable inclusive/exclusive sequence window."""
        if after_seq < -1:
            raise SessionError("message replay cursor must be -1 or greater")
        if through_seq is not None and through_seq < 0:
            raise SessionError("message replay boundary must be non-negative")
        sql = (
            "SELECT seq, payload FROM messages "
            "WHERE session_id = ? AND seq > ?"
        )
        params: list[Any] = [session_id, after_seq]
        if through_seq is not None:
            sql += " AND seq <= ?"
            params.append(through_seq)
        sql += " ORDER BY seq"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: list[StoredMessage] = []
        for seq, payload in rows:
            message = _stored_message(session_id, seq, payload)
            out.append(StoredMessage(seq=seq, message=message))
        return out

    def _canonical_message_suffix(
        self, session_id: str, *, through_seq: int, count: int
    ) -> list[Message]:
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
            with self._lock:
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

    def messages(self, session_id: str) -> list[Message]:
        """Stored messages in order; empty list when the session has no row."""
        return [record.message for record in self.message_records(session_id)]

    def events(
        self,
        session_id: str,
        *,
        after_seq: int = -1,
        kind: str | None = None,
    ) -> list[SessionEvent]:
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
                out.append(SessionEvent(
                    event_seq=event_seq,
                    message_seq=message_seq,
                    kind=event_kind,
                    created_at=created_at,
                    payload=payload,
                ))
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
                snapshot = CompactionSnapshot.model_validate({
                    **payload,
                    "event_seq": event_seq,
                    "message_seq": message_seq,
                })
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

    def discard_events_from_message_seq(self, session_id: str, seq: int) -> int:
        """Discard runtime events anchored at or after a removed branch point."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM session_events "
                "WHERE session_id = ? AND message_seq >= ?",
                (session_id, seq),
            )
            return cur.rowcount

    def next_sequence(self, session_id: str) -> int:
        """Sequence the next append will receive (0 for a fresh session)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row[0])

    def truncate_after(self, session_id: str, seq: int) -> int:
        """Delete messages after ``seq`` and return the number removed.

        This is the stop/cancel cleanup primitive: the current user message is
        kept while any partially committed assistant/tool tail is discarded.
        """
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND seq > ?",
                (session_id, seq),
            )
            event_cur = self._conn.execute(
                "DELETE FROM session_events "
                "WHERE session_id = ? AND message_seq >= ?",
                (session_id, seq),
            )
            if cur.rowcount or event_cur.rowcount:
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (_utcnow(), session_id),
                )
            return cur.rowcount

    def rewind_to_user_message(self, session_id: str, seq: int) -> Message:
        """Delete ``seq`` and the later tail, returning the edited user message.

        Only an ordinary user message is a valid branch point. Synthetic user
        messages (tool-media hoists and compaction summaries) cannot be edited.
        The returned attachments let a frontend preserve them on regeneration.
        """
        if seq < 0:
            raise SessionError("message sequence must be non-negative")
        with self._lock, self._conn:
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
                "DELETE FROM session_events "
                "WHERE session_id = ? AND message_seq >= ?",
                (session_id, seq),
            )
            # If this was the auto-titled first prompt, clear it so appending the
            # edited prompt derives a fresh title. Preserve explicit renames.
            title = current_title
            if earlier_user is None and current_title == _message_title(target):
                title = ""
            self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, _utcnow(), session_id),
            )
        return target

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _row_to_meta(row: tuple) -> SessionMeta:
    sid, title, profile_name, created_at, updated_at, raw_meta, count = row
    raw_fork = _session_meta(raw_meta).get("fork")
    try:
        fork = SessionForkMeta.model_validate(raw_fork) if raw_fork else None
    except Exception:
        fork = None
    return SessionMeta(
        id=sid,
        title=title,
        profile_name=profile_name,
        created_at=created_at,
        updated_at=updated_at,
        message_count=count,
        fork=fork,
    )


class SessionMemory:
    """A ``ShortTermMemory`` that mirrors every ``add`` into a SessionStore.

    Wraps a ``WindowMemory``: rendering (and its window trimming) is untouched,
    while the store keeps the *full* history regardless of the window.
    """

    def __init__(self, inner: "ShortTermMemory", store: SessionStore, session_id: str) -> None:
        self._inner = inner
        self._store = store
        self._session_id = session_id
        self._compaction_turn_index: int | None = None
        self._last_sequence: int | None = None

    def add(self, message: Message) -> None:
        self._inner.add(message)
        self._last_sequence = self._store.append(self._session_id, message)

    def replace(self, messages: list[Message]) -> None:
        # Working-set repair must not append/rewrite the durable transcript;
        # Agent.finalize_cancelled_turn truncates its persisted tail atomically.
        self._inner.replace(messages)

    def render(self, system_prompt: str) -> list[Message]:
        return self._inner.render(system_prompt)

    def set_compaction_turn_index(self, turn_index: int) -> None:
        """Supply the agent's validated iteration count for the next snapshot."""
        self._compaction_turn_index = turn_index

    async def maybe_compact(self, system_prompt: str = ""):
        # The transcript remains lossless, while the derived working-set
        # snapshot is anchored separately so resume can start from it and replay
        # only messages appended after the compaction watermark.
        turn_index = self._compaction_turn_index
        self._compaction_turn_index = None
        event = await self._inner.maybe_compact(system_prompt)
        if event is not None:
            self._store.save_compaction(
                self._session_id,
                messages=self._inner.messages,
                summarized_messages=event.summarized_messages,
                before_tokens=event.before_tokens,
                after_tokens=event.after_tokens,
                turn_index=turn_index,
            )
        return event

    @property
    def messages(self) -> list[Message]:
        return self._inner.messages

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_sequence(self) -> int | None:
        """Exact durable sequence assigned to the most recent ``add``."""
        return self._last_sequence


def _trim_dangling_records(records: list[StoredMessage]) -> list[StoredMessage]:
    """Sequence-preserving implementation shared by replay and validation."""
    out: list[StoredMessage] = []
    i, n = 0, len(records)
    while i < n:
        record = records[i]
        message = record.message
        if message.role == "assistant" and message.tool_calls:
            wanted = {call.id for call in message.tool_calls}
            j = i + 1
            got: set[str | None] = set()
            while j < n and records[j].message.role == "tool":
                got.add(records[j].message.tool_call_id)
                j += 1
            if wanted == got:
                out.extend(records[i:j])
            i = j
        elif message.role == "tool":
            i += 1
        else:
            out.append(record)
            i += 1
    return out


def trim_dangling(messages: list[Message]) -> list[Message]:
    """Drop incomplete tool blocks so the result is valid API history.

    An assistant message with ``tool_calls`` is kept only when *all* its tool
    results immediately follow; orphaned ``tool`` messages are dropped. This
    scans the whole list (a crashed turn can leave a dangling block mid-list
    once later turns append after it), and keeps a trailing lone user message.
    """
    records = [
        StoredMessage(seq=index, message=message)
        for index, message in enumerate(messages)
    ]
    return [record.message for record in _trim_dangling_records(records)]


def attach_session(
    inner: "ShortTermMemory", store: SessionStore, session_id: str | None
) -> tuple[SessionMemory, str, int]:
    """Hydrate ``inner`` from the store and wrap it for recording.

    Returns ``(memory, session_id, restored_turn_index)``. A fresh id (or an
    id with no stored row yet) yields an empty replay — the row appears with
    the first appended message. ``restored_turn_index`` is the number of
    stored assistant messages: the loop appends exactly one per iteration, so
    the count restores ``Agent._turn_index``.
    """
    sid = session_id or new_session_id()
    snapshot = store.latest_compaction(sid)
    if snapshot is None:
        replayed = trim_dangling(store.messages(sid))
        turn_index = sum(1 for m in replayed if m.role == "assistant")
    else:
        # The snapshot contains the exact compacted working set through its
        # anchor. Only hydrate and validate the later raw tail; old full-history
        # rows remain available for transcript display and future branch edits.
        replayed = list(snapshot.messages)
        tail = trim_dangling([
            record.message
            for record in store.message_records(
                sid, after_seq=snapshot.message_seq
            )
        ])
        replayed.extend(tail)
        turn_index = snapshot.turn_index + sum(
            1 for message in tail if message.role == "assistant"
        )
    inner.replace(replayed)  # never re-append replayed state to the durable store
    return SessionMemory(inner, store, sid), sid, turn_index


def open_store(profile: "AgentProfile") -> tuple[SessionStore | None, str | None]:
    """Open the profile's session store, applying the path policy.

    Returns ``(store, notice)``: a usable store with no notice, or ``None``
    with an optional one-line notice explaining why persistence is off
    (a profile directory inside the installed package, or one without a
    source directory).
    Misconfiguration — an escaping relative path, or an absolute path without
    ``allow_absolute_path`` — raises ``ConfigError`` instead.
    """
    cfg = profile.sessions
    if not cfg.enabled:
        return None, None

    raw = Path(cfg.path)
    if raw.is_absolute():
        if not cfg.allow_absolute_path:
            raise ConfigError(
                "sessions.path is absolute; set sessions.allow_absolute_path: true "
                "to permit this"
            )
        db_path = raw
    else:
        source_dir = getattr(profile, "_source_dir", None)
        if source_dir is None:
            return None, "session persistence disabled: profile has no source directory"
        resolved = (source_dir / raw).resolve()
        if not resolved.is_relative_to(source_dir.resolve()):
            raise ConfigError(f"sessions.path escapes profile directory: {cfg.path}")
        try:
            resolved.relative_to(_PACKAGE_DIR)
            return None, (
                "session persistence disabled: the profile directory is inside "
                "the installed lingcore package — copy it elsewhere, or set "
                "an absolute sessions.path with allow_absolute_path: true"
            )
        except ValueError:
            pass
        db_path = resolved

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SessionStore(db_path, profile_name=profile.name), None
    except (OSError, sqlite3.DatabaseError) as e:
        raise ConfigError(f"cannot open session store at {db_path}: {e}") from None
