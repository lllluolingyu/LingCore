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
- **Exclusive ownership:** attached agents hold renewable SQLite leases. Each
  mutation checks its owner token and epoch in the same write transaction.
  Expired owners cannot write after takeover; Edit fences the old working set.
- **Still deliberately ephemeral:** per-session "allow always" shell patterns
  are not persisted (re-approval after resume is the conservative default).
"""

from __future__ import annotations

import json as json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from lingcore._sessions.events import SessionEvents
from lingcore._sessions.forks import fork_session
from lingcore._sessions.models import (
    CompactionEventPayload,
    CompactionSnapshot,
    PersistedSkillState,
    SessionEvent,
    SessionForkMeta,
    SessionMeta,
    StoredMessage,
    is_session_id,
    new_session_id,
    trim_dangling,
)
from lingcore.errors import ConfigError
from lingcore.events import Compacted
from lingcore.message import Message
from lingcore.session_leases import SessionLease

if TYPE_CHECKING:
    from lingcore.config import AgentProfile
    from lingcore.memory import ShortTermMemory


__all__ = [
    "CompactionEventPayload",
    "CompactionSnapshot",
    "PersistedSkillState",
    "SessionEvent",
    "SessionForkMeta",
    "SessionMeta",
    "StoredMessage",
    "is_session_id",
    "new_session_id",
    "trim_dangling",
    "SessionStore",
    "SessionMemory",
    "attach_session",
    "open_store",
]

# The installed package root — a sessions.db inside it is refused (gracefully).
_PACKAGE_DIR = Path(__file__).parent.resolve()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class SessionStore(SessionEvents):
    """SQLite-backed session history with lease-fenced writes and replay.

    Each store owns a synchronized connection. Message storage, runtime events,
    and atomic forks live in separate implementation modules.
    """

    def fork_session(
        self,
        session_id: str,
        *,
        through_seq: int | None = None,
        title: str | None = None,
    ) -> SessionMeta:
        return fork_session(self, session_id, through_seq=through_seq, title=title)


class SessionMemory:
    """A ``ShortTermMemory`` that mirrors every ``add`` into a SessionStore.

    Wraps a ``WindowMemory``: rendering (and its window trimming) is untouched,
    while the store keeps the *full* history regardless of the window.
    """

    def __init__(
        self,
        inner: "ShortTermMemory",
        store: SessionStore,
        session_id: str,
        *,
        lease: SessionLease | None = None,
    ) -> None:
        self._inner = inner
        self._store = store
        self._session_id = session_id
        self.lease = lease if lease is not None else store.acquire_session(session_id)
        self._compaction_turn_index: int | None = None
        self._last_sequence: int | None = None

    def add(self, message: Message) -> None:
        self._last_sequence = self._store.append(
            self._session_id, message, lease=self.lease
        )
        self._inner.add(message)

    def close(self) -> None:
        """Release ownership after the driver has stopped using this memory."""
        self._store.release_session(self.lease)

    def replace(self, messages: list[Message]) -> None:
        # Working-set repair must not append/rewrite the durable transcript;
        # Agent.finalize_cancelled_turn truncates its persisted tail atomically.
        self._inner.replace(messages)

    def render(self, system_prompt: str) -> list[Message]:
        return self._inner.render(system_prompt)

    def set_compaction_turn_index(self, turn_index: int) -> None:
        """Supply the agent's validated iteration count for the next snapshot."""
        self._compaction_turn_index = turn_index

    def set_pinned_note(self, note: str) -> None:
        """Forward live state (todo list) that must survive compaction/eviction."""
        setter = getattr(self._inner, "set_pinned_note", None)
        if callable(setter):
            setter(note)

    async def maybe_compact(self, system_prompt: str = "") -> Compacted | None:
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
                lease=self.lease,
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
    lease = store.acquire_session(sid)
    try:
        return _hydrate_session(inner, store, sid, lease)
    except BaseException:
        store.release_session(lease)
        raise


def _hydrate_session(
    inner: "ShortTermMemory", store: SessionStore, sid: str, lease: SessionLease
) -> tuple[SessionMemory, str, int]:
    snapshot = store.latest_compaction(sid)
    if snapshot is None:
        replayed = trim_dangling(store.messages(sid))
        turn_index = sum(1 for m in replayed if m.role == "assistant")
    else:
        # The snapshot contains the exact compacted working set through its
        # anchor. Only hydrate and validate the later raw tail; old full-history
        # rows remain available for transcript display and future branch edits.
        replayed = list(snapshot.messages)
        tail = trim_dangling(
            [
                record.message
                for record in store.message_records(sid, after_seq=snapshot.message_seq)
            ]
        )
        replayed.extend(tail)
        turn_index = snapshot.turn_index + sum(
            1 for message in tail if message.role == "assistant"
        )
    inner.replace(replayed)  # never re-append replayed state to the durable store
    # Long hydration must not hand a caller a working set whose lease expired.
    store.heartbeat_session(lease)
    return SessionMemory(inner, store, sid, lease=lease), sid, turn_index


def open_store(
    profile: "AgentProfile", *, state_dir: Path | None = None
) -> tuple[SessionStore | None, str | None]:
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
        db_path = raw.resolve()
    else:
        source_dir = state_dir or getattr(profile, "_source_dir", None)
        if source_dir is None:
            return None, "session persistence disabled: profile has no source directory"
        resolved = (source_dir / raw).resolve()
        if not resolved.is_relative_to(source_dir.resolve()):
            raise ConfigError(f"sessions.path escapes profile directory: {cfg.path}")
        db_path = resolved

    if state_dir is not None and not db_path.is_relative_to(state_dir.resolve()):
        raise ConfigError("sessions.path escapes the explicit state directory")

    # Both relative and absolute paths resolve before this shared package
    # guard, so allow_absolute_path can never create a sessions.db inside
    # installed package code.
    try:
        db_path.relative_to(_PACKAGE_DIR)
        return None, (
            "session persistence disabled: the profile directory is inside "
            "the installed lingcore package — copy it elsewhere, or set "
            "an absolute sessions.path with allow_absolute_path: true"
        )
    except ValueError:
        pass

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return SessionStore(db_path, profile_name=profile.name), None
    except (OSError, sqlite3.DatabaseError) as e:
        raise ConfigError(f"cannot open session store at {db_path}: {e}") from None
