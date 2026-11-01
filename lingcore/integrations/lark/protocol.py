"""SDK-independent message and transport contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LarkResource:
    key: str
    kind: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class LarkMessage:
    message_id: str
    user_id: str
    chat_id: str
    chat_type: str
    text: str = ""
    message_type: str = "text"
    mentioned_bot: bool = False
    sender_type: str | None = "user"
    resources: tuple[LarkResource, ...] = ()

    @property
    def scope(self) -> tuple[str, str]:
        return self.chat_id, self.user_id


class LarkSender(Protocol):
    async def send_text(self, message: LarkMessage, text: str) -> None: ...

    async def send_file(
        self, message: LarkMessage, data: bytes, *, name: str, image: bool
    ) -> None: ...

    async def download(self, message: LarkMessage, resource: LarkResource) -> bytes: ...
