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
from lingcore.background import BackgroundInput
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
    background = agent.background
    background.enabled = bool(getattr(frontend, "supports_background", False))
    while True:
        user_input = await _read_next(agent, frontend)
        if user_input is None:
            return
        incoming = (
            user_input
            if isinstance(user_input, UserInput)
            else UserInput(text=user_input)
        )
        if not incoming.text.strip() and not incoming.attachments:
            continue
        if not isinstance(incoming, BackgroundInput):
            background.paused = False  # the user is back in control
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
            # A Stop hands control back to the user: queued or later monitor
            # events must not start the next turn before they can type.
            if background.enabled and not background.closed:
                background.paused = True
                paused = getattr(frontend, "background_paused", None)
                if paused is not None and background.busy:
                    paused()


async def _read_next(agent: Agent, frontend: Frontend) -> str | UserInput | None:
    """Let an event wake a cancellable editor; never race a blocking stdin read."""
    inbox = agent.background
    if not inbox.enabled or inbox.paused:
        return await frontend.read_input()
    while True:
        reader = asyncio.create_task(frontend.read_input())
        wake = asyncio.create_task(inbox.wait())
        try:
            await asyncio.wait({reader, wake}, return_when=asyncio.FIRST_COMPLETED)
            if reader.done():  # user input wins a simultaneous arrival
                return reader.result()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            if not reader.cancelled():  # it finished before the cancel landed
                return reader.result()
            if inbox.closed:
                return None
            incoming = inbox.take()
            if incoming is not None:
                received = getattr(frontend, "background_received", None)
                if received is not None:
                    received(incoming)
                return incoming
            # The source may have been stopped after waking this reader. Keep
            # listening to later events instead of falling back to stdin alone.
        finally:
            for task in (reader, wake):
                if not task.done():
                    task.cancel()
            await asyncio.gather(reader, wake, return_exceptions=True)
