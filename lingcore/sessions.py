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
- **v1 limitations (deliberate):** active skills are not restored on resume
  (the model can re-activate them); per-session "allow always" shell patterns
  are not persisted (re-approval after resume is the conservative default);
  attach-exclusivity is enforced per process, not across processes.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from lingcore.errors import ConfigError, SessionError
from lingcore.message import Message

if TYPE_CHECKING:
    from lingcore.config import AgentProfile
    from lingcore.memory import ShortTermMemory, WindowMemory

# The installed package root — a sessions.db inside it is refused (gracefully).
_PACKAGE_DIR = Path(__file__).parent.resolve()

_SCHEMA_VERSION = 1
_TITLE_LIMIT = 60
_ID_RE = re.compile(r"[0-9a-f]{32}")
_PREFIX_RE = re.compile(r"[0-9a-f]{1,32}")

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
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)",
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
    title = _derive_title(message.content)
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
    return message


class SessionMeta(BaseModel):
    """Summary row for one stored session."""

    id: str
    title: str = ""
    profile_name: str = ""
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


_META_QUERY = """
SELECT s.id, s.title, s.profile_name, s.created_at, s.updated_at,
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
    def append(self, session_id: str, message: Message) -> None:
        """Append one message; creates the session row lazily (same txn)."""
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

    def messages(self, session_id: str) -> list[Message]:
        """Stored messages in order; empty list when the session has no row."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, payload FROM messages WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        out: list[Message] = []
        for seq, payload in rows:
            try:
                out.append(Message.model_validate_json(payload))
            except Exception as e:
                # Attachment validation tightens over releases, so a row that
                # was valid when written may fail today's rules (e.g. an
                # over-limit hoist persisted before the caps existed). Dropping
                # the media but keeping the text degrades one message instead
                # of bricking the whole session on load.
                salvaged = _strip_invalid_attachments(payload)
                if salvaged is not None:
                    out.append(salvaged)
                    continue
                raise SessionError(
                    f"corrupt message row {session_id[:8]}/{seq}: {e}"
                ) from None
        return out

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _row_to_meta(row: tuple) -> SessionMeta:
    sid, title, profile_name, created_at, updated_at, count = row
    return SessionMeta(
        id=sid,
        title=title,
        profile_name=profile_name,
        created_at=created_at,
        updated_at=updated_at,
        message_count=count,
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

    def add(self, message: Message) -> None:
        self._inner.add(message)
        self._store.append(self._session_id, message)

    def render(self, system_prompt: str) -> list[Message]:
        return self._inner.render(system_prompt)

    async def maybe_compact(self, system_prompt: str = ""):
        # Compaction rewrites only the in-memory working set; the store keeps the
        # full history, so the summary is never persisted (resume replays raw).
        return await self._inner.maybe_compact(system_prompt)

    @property
    def messages(self) -> list[Message]:
        return self._inner.messages

    @property
    def session_id(self) -> str:
        return self._session_id


def trim_dangling(messages: list[Message]) -> list[Message]:
    """Drop incomplete tool blocks so the result is valid API history.

    An assistant message with ``tool_calls`` is kept only when *all* its tool
    results immediately follow; orphaned ``tool`` messages are dropped. This
    scans the whole list (a crashed turn can leave a dangling block mid-list
    once later turns append after it), and keeps a trailing lone user message.
    """
    out: list[Message] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            wanted = {tc.id for tc in m.tool_calls}
            j = i + 1
            got: set[str | None] = set()
            while j < n and messages[j].role == "tool":
                got.add(messages[j].tool_call_id)
                j += 1
            if wanted == got:
                out.extend(messages[i:j])
            i = j
        elif m.role == "tool":
            i += 1
        else:
            out.append(m)
            i += 1
    return out


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
    replayed = trim_dangling(store.messages(sid))
    for m in replayed:
        inner.add(m)  # directly into the inner memory — must not re-append to the store
    turn_index = sum(1 for m in replayed if m.role == "assistant")
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
