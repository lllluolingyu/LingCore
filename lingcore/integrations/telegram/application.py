"""Thin python-telegram-bot adapter and synchronous Telegram runners."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import httpx

# All PTB imports intentionally live below lingcore.integrations.telegram.
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.error import InvalidToken
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)
from typing_extensions import Buffer

from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, ToolError
from lingcore.integrations.telegram.bridge import LLMFactory, TelegramBridge
from lingcore.integrations.telegram.config import (
    TelegramConfig,
    load_telegram_config,
)
from lingcore.integrations.telegram.protocol import (
    InlineButtons,
    TelegramCallback,
    TelegramFile,
    TelegramMessage,
    TelegramSender,
)

if TYPE_CHECKING:
    from telegram import Bot

_ALLOWED_UPDATES = ["message", "callback_query"]
_BRIDGE_KEY = "lingcore.telegram.bridge"
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


class _BoundedBuffer(io.BytesIO):
    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self.max_bytes = max_bytes

    def write(self, data: Buffer) -> int:
        if self.tell() + memoryview(data).nbytes > self.max_bytes:
            raise ToolError(
                f"file too large (download exceeded limit {self.max_bytes})"
            )
        return super().write(data)


def _read_local_file_bounded(path: Path, max_bytes: int) -> bytes:
    buffer = _BoundedBuffer(max_bytes)
    with path.open("rb") as source:
        while True:
            remaining = max_bytes - buffer.tell()
            chunk = source.read(min(_DOWNLOAD_CHUNK_BYTES, remaining + 1))
            if not chunk:
                return buffer.getvalue()
            buffer.write(chunk)


async def _download_http_file_bounded(url: str, max_bytes: int) -> bytes:
    buffer = _BoundedBuffer(max_bytes)
    async with httpx.AsyncClient(follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            raw_length = response.headers.get("content-length")
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                except ValueError:
                    pass
                else:
                    if content_length > max_bytes:
                        raise ToolError(
                            f"file too large ({content_length} bytes; "
                            f"limit {max_bytes})"
                        )
            chunk_size = min(_DOWNLOAD_CHUNK_BYTES, max_bytes + 1)
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                buffer.write(chunk)
    return buffer.getvalue()


def _markup(buttons: InlineButtons | None) -> InlineKeyboardMarkup | None:
    if buttons is None:
        return None
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(button.text, callback_data=button.callback_data)
                for button in row
            ]
            for row in buttons
        ]
    )


class PTBSender(TelegramSender):
    """Duck-typed sender implementation backed by one PTB Bot."""

    def __init__(self, bot: "Bot") -> None:
        self.bot = bot

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: InlineButtons | None = None,
    ) -> Any:
        return await self.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=None,
            reply_markup=_markup(reply_markup),
        )

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        reply_markup: InlineButtons | None = None,
    ) -> Any:
        return await self.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=None,
            reply_markup=_markup(reply_markup),
        )

    async def edit_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        *,
        reply_markup: InlineButtons | None = None,
    ) -> Any:
        return await self.bot.edit_message_reply_markup(
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=_markup(reply_markup),
        )

    async def delete_message(self, chat_id: int, message_id: int) -> Any:
        return await self.bot.delete_message(chat_id=chat_id, message_id=message_id)

    async def send_photo(self, chat_id: int, data: bytes, *, filename: str) -> Any:
        return await self.bot.send_photo(
            chat_id=chat_id,
            photo=InputFile(io.BytesIO(data), filename=filename),
        )

    async def send_document(self, chat_id: int, data: bytes, *, filename: str) -> Any:
        return await self.bot.send_document(
            chat_id=chat_id,
            document=InputFile(io.BytesIO(data), filename=filename),
        )

    async def download_file(self, file_id: str, *, max_bytes: int) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        remote = await self.bot.get_file(file_id)
        size = getattr(remote, "file_size", None)
        if size is not None and size > max_bytes:
            raise ToolError(f"file too large ({size} bytes; limit {max_bytes})")
        if getattr(remote, "_credentials", None) is not None:
            raise ToolError("encrypted Telegram files are not supported")
        raw_path = getattr(remote, "file_path", None)
        if not raw_path:
            raise ToolError("Telegram did not provide a downloadable file path")
        local_path = Path(raw_path)
        if local_path.is_absolute():
            return await asyncio.to_thread(
                _read_local_file_bounded, local_path, max_bytes
            )
        parsed = urlsplit(str(raw_path))
        if parsed.scheme in {"http", "https"}:
            return await _download_http_file_bounded(str(raw_path), max_bytes)
        if parsed.scheme:
            raise ToolError("Telegram returned an unsupported file path")
        raise ToolError("Telegram returned an invalid local file path")

    async def answer_callback(
        self,
        callback_query_id: str,
        *,
        text: str | None = None,
        show_alert: bool = False,
    ) -> Any:
        return await self.bot.answer_callback_query(
            callback_query_id=callback_query_id,
            text=text,
            show_alert=show_alert,
        )


def _file(value: Any, *, default_name: str) -> TelegramFile:
    return TelegramFile(
        file_id=value.file_id,
        name=getattr(value, "file_name", None) or default_name,
        file_size=getattr(value, "file_size", None),
        media_type=getattr(value, "mime_type", None),
        width=int(getattr(value, "width", 0) or 0),
        height=int(getattr(value, "height", 0) or 0),
    )


def _normalize_message(update: Any) -> TelegramMessage | None:
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return None
    user = update.effective_user
    photos = tuple(
        _file(photo, default_name="photo.jpg")
        for photo in (getattr(message, "photo", None) or ())
    )
    raw_document = getattr(message, "document", None)
    document = (
        _file(raw_document, default_name="document")
        if raw_document is not None
        else None
    )
    unsupported_names = (
        "animation",
        "audio",
        "contact",
        "dice",
        "game",
        "gift",
        "giveaway",
        "giveaway_completed",
        "giveaway_created",
        "giveaway_winners",
        "invoice",
        "live_photo",
        "location",
        "paid_media",
        "passport_data",
        "poll",
        "sticker",
        "story",
        "unique_gift",
        "venue",
        "video",
        "video_note",
        "voice",
    )
    unsupported = next(
        (
            name
            for name in unsupported_names
            if getattr(message, name, None) is not None
        ),
        None,
    )
    if (
        unsupported is None
        and not photos
        and document is None
        and getattr(message, "effective_attachment", None) is not None
    ):
        unsupported = "media"
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    return TelegramMessage(
        update_id=update.update_id,
        user_id=getattr(user, "id", None),
        chat_id=chat.id,
        chat_type=str(chat.type),
        text=text,
        photos=photos,
        document=document,
        media_group_id=getattr(message, "media_group_id", None),
        unsupported_media=unsupported,
    )


def _normalize_callback(update: Any) -> TelegramCallback | None:
    query = update.callback_query
    if query is None:
        return None
    query_message = getattr(query, "message", None)
    chat_id = getattr(query_message, "chat_id", None)
    if chat_id is None:
        chat = getattr(query_message, "chat", None)
        chat_id = getattr(chat, "id", None)
    return TelegramCallback(
        update_id=update.update_id,
        callback_query_id=query.id,
        user_id=getattr(getattr(query, "from_user", None), "id", None),
        chat_id=chat_id,
        data=getattr(query, "data", None),
    )


def create_telegram_application(
    profile: AgentProfile,
    config: TelegramConfig | None = None,
    *,
    bot: "Bot | None" = None,
    llm_factory: LLMFactory | None = None,
) -> Application:
    """Construct a PTB Application with the official LingCore bridge handlers."""
    telegram_config = config or load_telegram_config(
        profile, require_secrets=bot is None
    )
    holder: dict[str, TelegramBridge] = {}

    async def close_bridge(_: Application) -> None:
        bridge = holder.get(_BRIDGE_KEY)
        if bridge is not None:
            await bridge.shutdown()

    async def register_commands(app: Application) -> None:
        bridge = holder[_BRIDGE_KEY]
        reserved = {"start", "help", "new", "sessions", "resume", "stop"}
        entries = [
            BotCommand(name, description)
            for name, description in (
                ("help", "Show help"),
                ("new", "Start a session"),
                ("sessions", "List sessions"),
                ("resume", "Resume a session"),
                ("stop", "Stop the active turn"),
            )
        ]
        entries.extend(
            BotCommand(c.telegram_name, (c.description or c.name)[:256])
            for c in bridge.commands.telegram_commands(reserved=reserved)
        )
        await app.bot.set_my_commands(entries[:100])

    # Keep update handlers serialized: bridge Stop/session transitions rely on
    # PTB's one-update-at-a-time dispatch contract.
    builder = Application.builder().concurrent_updates(False)
    if bot is None:
        builder = builder.token(telegram_config.resolve_token()).rate_limiter(
            AIORateLimiter(max_retries=1)  # type: ignore[arg-type]
        )
    else:
        builder = builder.bot(bot)  # type: ignore[arg-type]
    # post_stop runs before the Bot is shut down (so confirmation buttons can
    # still be removed); post_shutdown covers manual lifecycle usage too.
    builder = (
        builder.post_init(register_commands)
        .post_stop(close_bridge)
        .post_shutdown(close_bridge)
    )
    try:
        application = builder.build()
    except InvalidToken:
        raise ConfigError(
            f"telegram.token_env names {telegram_config.token_env!r}, but its "
            "value is not a valid Telegram bot token"
        ) from None
    bridge = TelegramBridge(profile, telegram_config, llm_factory=llm_factory)
    holder[_BRIDGE_KEY] = bridge
    application.bot_data[_BRIDGE_KEY] = bridge

    async def on_message(update: Any, context: Any) -> None:
        incoming = _normalize_message(update)
        if incoming is not None:
            await bridge.handle_message(incoming, PTBSender(context.bot))

    async def on_callback(update: Any, context: Any) -> None:
        incoming = _normalize_callback(update)
        if incoming is not None:
            await bridge.handle_callback(incoming, PTBSender(context.bot))

    application.add_handler(MessageHandler(filters.ALL, on_message))
    application.add_handler(CallbackQueryHandler(on_callback))
    return application


def run_telegram(
    profile_path: str | Path,
    *,
    telegram_config_path: str | Path | None = None,
    mode: Literal["polling", "webhook"] | None = None,
) -> int:
    """Synchronously run polling/webhook so PTB owns loop and signal handling."""
    profile = AgentProfile.load(profile_path)
    config = load_telegram_config(
        profile,
        telegram_config_path,
        mode=mode,  # argparse constrains this to the literal values
        require_secrets=True,
    )
    application = create_telegram_application(profile, config)
    try:
        if config.mode == "polling":
            application.run_polling(
                allowed_updates=_ALLOWED_UPDATES,
                drop_pending_updates=False,
            )
        else:
            application.run_webhook(
                listen=config.webhook.listen,
                port=config.webhook.port,
                url_path=config.webhook_path,
                webhook_url=config.webhook.public_url,
                secret_token=config.resolve_webhook_secret(),
                allowed_updates=_ALLOWED_UPDATES,
                drop_pending_updates=False,
            )
    except InvalidToken:
        raise ConfigError(
            f"telegram.token_env names {config.token_env!r}, but its value is "
            "not a valid Telegram bot token"
        ) from None
    return 0
