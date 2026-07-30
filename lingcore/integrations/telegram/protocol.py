"""PTB-independent message and transport contracts for Telegram."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TelegramFile:
    file_id: str
    name: str
    file_size: int | None = None
    media_type: str | None = None
    width: int = 0
    height: int = 0


@dataclass(frozen=True, slots=True)
class TelegramMessage:
    update_id: int
    user_id: int | None
    chat_id: int
    chat_type: str
    text: str = ""
    photos: tuple[TelegramFile, ...] = field(default_factory=tuple)
    document: TelegramFile | None = None
    media_group_id: str | None = None
    unsupported_media: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramCallback:
    update_id: int
    callback_query_id: str
    user_id: int | None
    chat_id: int | None
    data: str | None


@dataclass(frozen=True, slots=True)
class InlineButton:
    text: str
    callback_data: str


InlineButtons = tuple[tuple[InlineButton, ...], ...]


class TelegramSender(Protocol):
    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: InlineButtons | None = None,
    ) -> Any: ...

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        reply_markup: InlineButtons | None = None,
    ) -> Any: ...

    async def edit_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        *,
        reply_markup: InlineButtons | None = None,
    ) -> Any: ...

    async def delete_message(self, chat_id: int, message_id: int) -> Any: ...

    async def send_photo(
        self, chat_id: int, data: bytes, *, filename: str
    ) -> Any: ...

    async def send_document(
        self, chat_id: int, data: bytes, *, filename: str
    ) -> Any: ...

    async def download_file(self, file_id: str, *, max_bytes: int) -> bytes: ...

    async def answer_callback(
        self,
        callback_query_id: str,
        *,
        text: str | None = None,
        show_alert: bool = False,
    ) -> Any: ...


def message_id(value: Any) -> int:
    """Extract a Telegram message id from a duck-typed sender result."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    result = getattr(value, "message_id", None)
    if isinstance(result, int) and not isinstance(result, bool):
        return result
    raise TypeError("Telegram sender did not return a message id")
