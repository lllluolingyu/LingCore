"""Prompt composition.

``PromptComposer`` is the per-turn seam that assembles the system prompt from
static layers (world/role/workflow), live memory, active skills, and optional
retrieved context.  The loop calls ``compose(ctx)`` at the top of every
iteration so skill activation and memory writes take effect on the *next*
model request without any additional plumbing.

``ComposeContext`` is a frozen value object — no mutable hidden state, safe for
concurrent sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ComposeContext:
    """Immutable per-iteration snapshot passed to every ``compose()`` call."""

    user_message: str
    turn_index: int
    session_id: str | None = None
    active_skills: tuple[str, ...] = field(default_factory=tuple)


class PromptComposer(Protocol):
    async def compose(self, ctx: ComposeContext) -> str: ...


@dataclass
class StaticComposer:
    """Zero-overhead drop-in: wraps a frozen string.  Used when no layers,
    memory, or skills are configured — behaviour is identical to the old
    ``system_prompt`` attribute on ``Agent``."""

    text: str

    async def compose(self, ctx: ComposeContext) -> str:
        return self.text


@dataclass
class LayeredComposer:
    """Compose world/role/workflow layers, live memory, active skill
    instructions, and optionally retrieved context on every call."""

    # Static layers resolved at build time (already expanded strings).
    layers: list[str]
    # Path to memory.md; re-read on every compose() if it exists.
    memory_path: Path | None
    # Skill name -> instruction body; populated by Agent.from_profile.
    skill_instructions: dict[str, str] = field(default_factory=dict)

    async def compose(self, ctx: ComposeContext) -> str:
        parts: list[str] = list(self.layers)

        if self.memory_path and self.memory_path.is_file():
            parts.append(self.memory_path.read_text("utf-8"))

        for skill_name in ctx.active_skills:
            body = self.skill_instructions.get(skill_name)
            if body:
                parts.append(body)

        return "\n\n".join(p.strip() for p in parts if p.strip())
