"""The frontend boundary.

Every frontend — the CLI today, a web or Discord adapter later — implements
the ``Frontend`` protocol, and ``run_session`` drives the agent through it.
The agent and its event stream never change when a new frontend is added;
that is the whole point of routing everything through ``AgentEvent``.
"""

from __future__ import annotations

from typing import Protocol

from lingcore.agent import Agent
from lingcore.events import AgentEvent
from lingcore.message import UserInput


class Frontend(Protocol):
    async def read_input(self) -> str | UserInput | None:
        """Return the next user message, or ``None`` to end the session."""
        ...

    def render(self, event: AgentEvent) -> None:
        """Render one agent event (text delta, tool call, final, error)."""
        ...

    async def confirm(self, command: str) -> bool:
        """Ask the user to approve a risky action (e.g. a shell command)."""
        ...


async def run_session(agent: Agent, frontend: Frontend) -> None:
    """Read user turns and stream agent events back until input ends.

    This loop is frontend-agnostic: it speaks only ``Frontend`` and
    ``AgentEvent``. A web server would call ``agent.run`` per request instead,
    but reuse the exact same agent and events.
    """
    while True:
        user_input = await frontend.read_input()
        if user_input is None:
            return
        incoming = user_input if isinstance(user_input, UserInput) else UserInput(text=user_input)
        if not incoming.text.strip() and not incoming.attachments:
            continue
        async for event in agent.run(incoming):
            frontend.render(event)
