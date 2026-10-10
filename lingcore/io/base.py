"""The frontend boundary.

Every frontend — the CLI today, a web or Discord adapter later — implements
the ``Frontend`` protocol, and ``run_session`` drives the agent through it.
The agent and its event stream never change when a new frontend is added;
that is the whole point of routing everything through ``AgentEvent``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager, aclosing, nullcontext
from typing import Protocol

from lingcore.agent import Agent
from lingcore.events import AgentEvent, PluginNotice
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


class InterruptibleFrontend(Frontend, Protocol):
    """Optional extension: a frontend that can stop a running turn.

    ``run_session`` enters ``interrupt_scope(stop)`` around each turn; the
    frontend calls ``stop()`` (``Agent.cancel_turn``) when the user asks to
    stop, and ``run_session`` performs the await -> finalize handshake.
    """

    def interrupt_scope(
        self, stop: Callable[[], bool]
    ) -> AbstractContextManager[None]: ...


async def _drive(agent: Agent, incoming: UserInput, frontend: Frontend) -> None:
    turn = agent.run(incoming)
    # ``async for`` does not guarantee immediate async-generator closure
    # when code in its body raises. Own the stream explicitly so a broken
    # renderer cannot leave Agent's turn checkpoint leased until a later
    # garbage-collection pass.
    async with aclosing(turn):
        async for event in turn:
            frontend.render(event)


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
        incoming = (
            user_input
            if isinstance(user_input, UserInput)
            else UserInput(text=user_input)
        )
        if not incoming.text.strip() and not incoming.attachments:
            continue
        # The turn runs in its own task so a frontend-requested stop cancels
        # only the turn, never the session loop.
        task = asyncio.create_task(_drive(agent, incoming, frontend))
        scope_factory = getattr(frontend, "interrupt_scope", None)
        scope: AbstractContextManager[None] = (
            scope_factory(agent.cancel_turn) if scope_factory else nullcontext()
        )
        try:
            with scope:
                await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if (current is not None and current.cancelling()) or not task.cancelled():
                raise  # the session itself is being cancelled
            # The user stopped the turn: repair it, keep the submitted message,
            # and report requests billed before the cancellation landed.
            if agent.turn_pending_finalization:
                notices: list[PluginNotice] = getattr(
                    agent, "drain_plugin_notices", lambda: []
                )()
                terminal = agent.finalize_cancelled_turn()
                # Usage precedes the terminal event, as it does on a live turn.
                for usage_event in agent.drain_usage():
                    frontend.render(usage_event)
                for notice in notices:
                    frontend.render(notice)
                frontend.render(terminal)
