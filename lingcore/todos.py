"""Task-list state for the ``todo_write`` tool.

The model keeps a short checklist for multi-step work by rewriting it whole
with ``todo_write``. The live list is a ``TodoState`` shared through
``ToolContext.options`` (like ``SkillState``); the loop diffs it around each
tool batch, persists changes as ``todo_state`` session events, and emits
``TodoUpdated``. It is never injected into the system prompt — a changing list
there would invalidate the cached prompt prefix on every update. Instead the
tool result carries the list (append-only), and the memory pins it verbatim
into a compaction summary or at the head of what eviction retains, so it
survives old history being condensed or dropped.

Depends only on pydantic so events, sessions, and frontends can import it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TodoStatus = Literal["pending", "in_progress", "completed"]

DEFAULT_MAX_TODOS = 30
TODO_CONTENT_MAX_CHARS = 500

_MARKS: dict[str, str] = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


class TodoItem(BaseModel):
    """One checklist entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(
        min_length=1,
        max_length=TODO_CONTENT_MAX_CHARS,
        description="Imperative description of the task, e.g. 'Run the tests'.",
    )
    status: TodoStatus = Field(
        default="pending",
        description="pending, in_progress (at most one), or completed.",
    )

    @field_validator("content")
    @classmethod
    def _one_line(cls, value: str) -> str:
        text = " ".join(value.split())
        if not text:
            raise ValueError("todo content must not be blank")
        return text


def validate_todos(
    items: Iterable[TodoItem], *, max_items: int = DEFAULT_MAX_TODOS
) -> tuple[TodoItem, ...]:
    """Return the list if it is a valid complete checklist, else raise."""
    todos = tuple(items)
    if len(todos) > max_items:
        raise ValueError(f"at most {max_items} todos are allowed, got {len(todos)}")
    in_progress = sum(1 for item in todos if item.status == "in_progress")
    if in_progress > 1:
        raise ValueError(
            f"at most one todo may be in_progress, got {in_progress}; finish or "
            "reset the others first"
        )
    return todos


def render_todos(items: Sequence[TodoItem]) -> str:
    """Plain-text checklist, one ``[x]``/``[>]``/``[ ]`` line per item."""
    if not items:
        return "(no todos)"
    return "\n".join(f"{_MARKS[item.status]} {item.content}" for item in items)


def todos_payload(items: Sequence[TodoItem]) -> dict[str, Any]:
    """The JSON object persisted in a ``todo_state`` session event."""
    return {"todos": [item.model_dump() for item in items]}


def todos_from_payload(payload: object) -> tuple[TodoItem, ...] | None:
    """Parse a persisted payload; ``None`` when it is not a valid checklist."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("todos")
    if not isinstance(raw, list):
        return None
    try:
        items = [TodoItem.model_validate(entry) for entry in raw]
        # Persisted rows were bounded when written; accept any valid length so
        # a profile that lowers max_items can still restore older sessions.
        return validate_todos(items, max_items=max(len(items), 1))
    except ValueError:
        return None


@dataclass
class TodoState:
    """Live checklist shared between the agent loop and ``todo_write``."""

    items: tuple[TodoItem, ...] = field(default_factory=tuple)

    def pinned_note(self) -> str:
        """Verbatim text memory pins so the list survives compaction and eviction."""
        if not self.items:
            return ""
        return "[Current todo list]\n" + render_todos(self.items)
