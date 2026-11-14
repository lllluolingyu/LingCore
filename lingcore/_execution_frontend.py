"""Driver-owned stopping around an injectable headless frontend."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

from lingcore.events import AgentEvent
from lingcore.message import UserInput

if TYPE_CHECKING:
    from lingcore.agent import Agent
    from lingcore.io.base import Frontend


class HeadlessFrontend:
    def __init__(self, agent: Agent, target: Frontend) -> None:
        self.agent = agent
        self.target = target
        self.supports_background = bool(getattr(target, "supports_background", False))
        self.stopped = False
        self.timed_out = False

    def stop(self) -> bool:
        self.stopped = True
        return self.agent.cancel_turn()

    def interrupt_scope(self, stop: Callable[[], bool]) -> AbstractContextManager[None]:
        factory = getattr(self.target, "interrupt_scope", None)
        return factory(stop) if factory else nullcontext()

    async def read_input(self) -> str | UserInput | None:
        outcome = self.agent.execution_outcome
        if self.stopped or (outcome is not None and outcome.status != "completed"):
            return None
        return await self.target.read_input()

    def render(self, event: AgentEvent) -> None:
        self.target.render(event)

    async def confirm(self, command: str) -> bool:
        return await self.target.confirm(command)
