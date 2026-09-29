"""todo_write — the model's task checklist for multi-step work.

Each call replaces the whole list (idempotent, and simpler for models than
incremental edits). The tool only validates and swaps the shared
``TodoState``; the agent loop diffs it around the tool batch, persists the
change, and emits ``TodoUpdated`` for frontends. See ``lingcore/todos.py``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lingcore.errors import ConfigError, ToolError
from lingcore.todos import TodoItem, TodoState, render_todos, validate_todos
from lingcore.tool_options import parse_todo_max_items
from lingcore.tools import ToolContext, tool

# Reserved options key the agent uses to share live TodoState with this tool.
TODO_STATE_KEY = "_todo_state"


class TodoWriteArgs(BaseModel):
    todos: list[TodoItem] = Field(
        description=(
            "The complete, updated todo list in order. Replaces the previous "
            "list; pass [] to clear it."
        )
    )


@tool(
    description=(
        "Create or update your task checklist for multi-step work. Pass the "
        "complete list every time (it replaces the previous one). Keep exactly "
        "one item in_progress while working, mark items completed as soon as "
        "they are done, and add items you discover along the way. Use it for "
        "tasks with three or more steps; skip it for trivial requests."
    )
)
async def todo_write(args: TodoWriteArgs, ctx: ToolContext) -> str:
    state = ctx.options.get(TODO_STATE_KEY)
    if not isinstance(state, TodoState):
        raise ToolError("todo_write is not available for this agent")
    try:
        max_items = parse_todo_max_items(ctx.options.get("todo_write", {}))
        todos = validate_todos(args.todos, max_items=max_items)
    except (ConfigError, ValueError) as exc:
        raise ToolError(str(exc)) from None
    state.items = todos
    done = sum(1 for item in todos if item.status == "completed")
    return f"Todo list updated ({done}/{len(todos)} completed):\n" + render_todos(todos)
