"""Session models implementation."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from lingcore.errors import SessionError
from lingcore.message import Message

_TITLE_LIMIT = 60
_ID_RE = re.compile(r"[0-9a-f]{32}")
_PREFIX_RE = re.compile(r"[0-9a-f]{1,32}")
_SESSION_EVENT_KINDS = frozenset(
    {"compaction", "skill_state", "todo_state", "turn_stats"}
)


def new_session_id() -> str:
    return uuid.uuid4().hex


def is_session_id(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value))


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
    suffix = (
        f" [{dropped} stored attachment(s) no longer pass validation; dropped on load]"
    )
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
            name: sorted(tools) for name, tools in state.approved_high_risk.items()
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
