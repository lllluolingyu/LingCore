"""Common source contract: transport-authenticated events and durable positions."""

from __future__ import annotations

from typing import Any, Protocol

from lingcore.superagent.core.models import Trigger


class Source(Protocol):
    name: str

    async def start(self) -> None: ...
    async def close(self) -> None: ...


def bounded_trigger(**values: Any) -> Trigger:
    values["text"] = str(values.get("text", ""))[:20000]
    return Trigger.model_validate(values)
