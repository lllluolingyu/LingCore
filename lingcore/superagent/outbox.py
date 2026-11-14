"""Channels send committed outbox entries; authenticated actions carry opaque nonces."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Protocol

from lingcore.superagent.core.models import Trigger


class Channel(Protocol):
    async def send(
        self,
        trigger: Trigger,
        text: str,
        *,
        key: str,
        approval: dict[str, Any] | None = None,
    ) -> str | None: ...


@dataclass(frozen=True)
class VerifiedAction:
    """Created only by a transport after verifying the sender and callback context."""

    actor: str
    token: str
    allow: bool
    source: str
    conversation: str = ""


def approval_card(
    ticket_id: str, tools: list[str], envelope_digest: str
) -> dict[str, Any]:
    return {
        "ticket_id": ticket_id,
        "tools": tools,
        "envelope_digest": envelope_digest,
        "token": secrets.token_urlsafe(24),
    }
