"""PTB-light Telegram bridge: isolation, commands, turns, and lifecycle."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lingcore.agent import Agent
from lingcore.errors import ConfigError, LingCoreError, SessionError, ToolError
from lingcore.events import AgentEvent, Error, Final, TurnCancelled
from lingcore.integrations.telegram.config import TelegramConfig
from lingcore.integrations.telegram.confirmations import ConfirmationManager
from lingcore.integrations.telegram.protocol import (
    TelegramCallback,
    TelegramFile,
    TelegramMessage,
    TelegramSender,
)
from lingcore.integrations.telegram.rendering import (
    EMPTY_RESPONSE,
    TelegramTurnRenderer,
    chunk_text,
)
from lingcore.integrations.telegram.state import TelegramStateStore
from lingcore.media import FILE_MAX_BYTES, IMAGE_MAX_BYTES, attachment_from_bytes
from lingcore.message import UserInput
from lingcore.sessions import SessionStore, new_session_id

if TYPE_CHECKING:
    from lingcore.config import AgentProfile

_LOG = logging.getLogger(__name__)
_UPDATE_CACHE_SIZE = 4_096
_ALBUM_CACHE_SIZE = 1_024
_ALBUM_TTL_SECONDS = 300.0

LLMFactory = Callable[[int], Any]

_HELP = """LingCore Telegram

Send text, one photo, or one document.

/new — start a new session
/sessions — show your 10 most recent sessions
/resume <id-prefix> — switch to a stored session
/stop — cancel your active turn
/help — show this help"""

_SUPPORTED_INPUT = (
    "Supported input is text (or a caption) with at most one photo or document."
)


@dataclass(slots=True)
class TelegramUserRuntime:
    user_id: int
    profile: "AgentProfile"
    store: SessionStore
    agent: Agent
    user_dir: Path
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    renderer: TelegramTurnRenderer | None = None
    final_delivery: bool = False


def select_photo(
    photos: tuple[TelegramFile, ...],
    *,
    max_bytes: int = IMAGE_MAX_BYTES,
) -> TelegramFile | None:
    """Choose the largest representation that is not known to exceed the cap."""
    eligible = [
        photo
        for photo in photos
        if photo.file_size is None or photo.file_size <= max_bytes
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda photo: (
            photo.width * photo.height,
            photo.file_size if photo.file_size is not None else -1,
        ),
    )


class TelegramBridge:
    """One-process bridge with one cached Agent runtime per allowlisted user."""

    def __init__(
        self,
        profile: "AgentProfile",
        config: TelegramConfig,
        *,
        llm_factory: LLMFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        confirmation_sleep=asyncio.sleep,
    ) -> None:
        self.profile = profile
        self.config = config
        self.llm_factory = llm_factory
        self._clock = clock
        self._allowed = frozenset(config.allowed_user_ids)
        self._state = TelegramStateStore(config.state_path / "bridge.sqlite3")
        self.confirmations = ConfirmationManager(
            config.confirmation_timeout, sleep=confirmation_sleep
        )
        self._runtimes: dict[int, TelegramUserRuntime] = {}
        self._runtime_lock = asyncio.Lock()
        self._senders: dict[int, TelegramSender] = {}
        self._chat_ids: dict[int, int] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._user_tasks: dict[int, asyncio.Task[None]] = {}
        self._updates: OrderedDict[int, None] = OrderedDict()
        self._albums: OrderedDict[tuple[int, str], float] = OrderedDict()
        self._closing = False
        self._closed = False

    @property
    def background_tasks(self) -> frozenset[asyncio.Task[None]]:
        return frozenset(self._tasks)

    async def handle_message(
        self, message: TelegramMessage, sender: TelegramSender
    ) -> None:
        if self._closing or not self._accept_update(message.update_id):
            return
        if message.chat_type != "private" or message.user_id is None:
            return
        user_id = message.user_id
        if user_id not in self._allowed:
            await sender.send_message(
                message.chat_id,
                f"This bot is private. Your Telegram user ID is {user_id}.",
            )
            return

        self._senders[user_id] = sender
        self._chat_ids[user_id] = message.chat_id

        if message.media_group_id is not None:
            if self._accept_album(message.chat_id, message.media_group_id):
                await sender.send_message(
                    message.chat_id,
                    f"Albums are not supported. {_SUPPORTED_INPUT}",
                )
            return

        command, argument = self._command(message.text)
        if command is not None:
            await self._handle_command(
                command, argument, user_id, message.chat_id, sender
            )
            return

        if message.unsupported_media is not None:
            await sender.send_message(message.chat_id, _SUPPORTED_INPUT)
            return
        if message.photos and message.document is not None:
            await sender.send_message(message.chat_id, _SUPPORTED_INPUT)
            return
        if not message.text.strip() and not message.photos and message.document is None:
            return

        attachment: TelegramFile | None = None
        attachment_limit = FILE_MAX_BYTES
        if message.photos:
            attachment = select_photo(message.photos)
            attachment_limit = IMAGE_MAX_BYTES
            if attachment is None:
                await sender.send_message(
                    message.chat_id,
                    f"Photo is too large (limit {IMAGE_MAX_BYTES} bytes).",
                )
                return
        elif message.document is not None:
            attachment = message.document
            if (
                attachment.file_size is not None
                and attachment.file_size > FILE_MAX_BYTES
            ):
                await sender.send_message(
                    message.chat_id,
                    f"Document is too large (limit {FILE_MAX_BYTES} bytes).",
                )
                return

        try:
            runtime = await self.runtime_for(user_id, message.chat_id, sender)
        except Exception as exc:
            await sender.send_message(
                message.chat_id,
                f"Could not start your LingCore runtime: "
                f"{self._safe_runtime_error(exc)}",
            )
            return
        await self._spawn_turn(
            runtime,
            text=message.text,
            attachment=attachment,
            attachment_limit=attachment_limit,
            sender=sender,
            chat_id=message.chat_id,
        )

    async def handle_callback(
        self, callback: TelegramCallback, sender: TelegramSender
    ) -> None:
        if self._closing or not self._accept_update(callback.update_id):
            # Telegram still expects duplicate callback queries to be answered.
            try:
                await sender.answer_callback(
                    callback.callback_query_id,
                    text="This confirmation was already handled.",
                )
            except Exception:
                _LOG.debug("failed to answer duplicate callback", exc_info=True)
            return
        try:
            answer, alert = await self.confirmations.resolve_callback(
                callback_data=callback.data,
                user_id=callback.user_id,
                chat_id=callback.chat_id,
            )
        except Exception:
            _LOG.exception("Telegram confirmation callback failed")
            answer, alert = "Could not resolve this confirmation.", True
        try:
            await sender.answer_callback(
                callback.callback_query_id, text=answer, show_alert=alert
            )
        except Exception:
            _LOG.debug("failed to answer Telegram callback query", exc_info=True)

    async def runtime_for(
        self,
        user_id: int,
        chat_id: int,
        sender: TelegramSender,
    ) -> TelegramUserRuntime:
        """Return the cached, isolated user runtime, constructing it lazily."""
        existing = self._runtimes.get(user_id)
        if existing is not None:
            self._senders[user_id] = sender
            self._chat_ids[user_id] = chat_id
            return existing
        async with self._runtime_lock:
            existing = self._runtimes.get(user_id)
            if existing is not None:
                self._senders[user_id] = sender
                self._chat_ids[user_id] = chat_id
                return existing
            self._senders[user_id] = sender
            self._chat_ids[user_id] = chat_id
            user_dir = self._user_directory(user_id)
            store = SessionStore(
                user_dir / "sessions.db", profile_name=self.profile.name
            )
            try:
                session_id = self._state.active_session(user_id) or new_session_id()
                scoped, agent = self._build_agent(user_id, user_dir, store, session_id)
                # Building/hydrating the Agent must succeed before the selection
                # becomes durable.
                self._state.set_active_session(user_id, session_id)
            except BaseException:
                store.close()
                raise
            runtime = TelegramUserRuntime(
                user_id=user_id,
                profile=scoped,
                store=store,
                agent=agent,
                user_dir=user_dir,
            )
            self._runtimes[user_id] = runtime
            return runtime

    async def shutdown(self) -> None:
        """Cancel turns, repair checkpoints, deny prompts, and close all stores."""
        if self._closed:
            return
        self._closing = True
        tasks = list(self._tasks)
        for runtime in self._runtimes.values():
            runtime.agent.cancel_turn()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for runtime in self._runtimes.values():
            if runtime.agent.turn_pending_finalization:
                try:
                    runtime.agent.finalize_cancelled_turn(
                        reason="Telegram bridge shutting down"
                    )
                except Exception:
                    _LOG.exception("failed to finalize a Telegram turn at shutdown")
            runtime.renderer = None
        await self.confirmations.deny_all()
        for runtime in self._runtimes.values():
            runtime.store.close()
        self._state.close()
        self._closed = True

    async def _spawn_turn(
        self,
        runtime: TelegramUserRuntime,
        *,
        text: str,
        attachment: TelegramFile | None,
        attachment_limit: int,
        sender: TelegramSender,
        chat_id: int,
    ) -> None:
        async with runtime.lock:
            active = self._user_tasks.get(runtime.user_id)
            if active is not None and not active.done():
                await sender.send_message(
                    chat_id,
                    "A turn is already running. Use /stop before sending another.",
                )
                return
            task = asyncio.create_task(
                self._run_turn(
                    runtime,
                    text=text,
                    attachment=attachment,
                    attachment_limit=attachment_limit,
                    sender=sender,
                    chat_id=chat_id,
                ),
                name=f"lingcore-telegram-{runtime.user_id}",
            )
            self._user_tasks[runtime.user_id] = task
            self._tasks.add(task)
            task.add_done_callback(
                lambda done, uid=runtime.user_id: self._task_done(uid, done)
            )

    async def _run_turn(
        self,
        runtime: TelegramUserRuntime,
        *,
        text: str,
        attachment: TelegramFile | None,
        attachment_limit: int,
        sender: TelegramSender,
        chat_id: int,
    ) -> None:
        attachments = []
        if attachment is not None:
            try:
                payload = await sender.download_file(
                    attachment.file_id, max_bytes=attachment_limit
                )
                if len(payload) > attachment_limit:
                    raise ToolError(
                        f"file too large ({len(payload)} bytes; "
                        f"limit {attachment_limit})"
                    )
                attachments.append(
                    attachment_from_bytes(payload, name=attachment.name)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Do not widen this tuple: Telegram/httpx exceptions can embed
                # the token-bearing download URL in their string value.
                detail = (
                    str(exc)
                    if isinstance(exc, (ToolError, ValueError))
                    else "download failed"
                )
                await sender.send_message(
                    chat_id, f"Could not accept that attachment: {detail}"
                )
                return

        incoming = UserInput(text=text, attachments=attachments)
        renderer = TelegramTurnRenderer(
            sender,
            chat_id,
            edit_interval=self.config.stream_edit_interval,
        )
        runtime.renderer = renderer
        delivery_warning_sent = False
        try:
            turn = runtime.agent.run(incoming)
            async with aclosing(turn):
                async for event in turn:
                    if isinstance(event, Final):
                        # Agent commits and releases its checkpoint before
                        # yielding Final. Keep Stop from misclassifying the
                        # remaining Telegram I/O as a pre-turn cancellation.
                        runtime.final_delivery = True
                    try:
                        await renderer.handle(event)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # Telegram/httpx exceptions can contain token-bearing
                        # URLs. Log only types, never exception text/tracebacks.
                        _LOG.warning(
                            "Telegram %s delivery failed with %s",
                            type(event).__name__,
                            type(exc).__name__,
                        )
                        delivery_warning_sent = await self._fallback_delivery(
                            event,
                            renderer=renderer,
                            sender=sender,
                            chat_id=chat_id,
                            warning_sent=delivery_warning_sent,
                        )
        finally:
            runtime.final_delivery = False
            if not runtime.agent.turn_pending_finalization:
                runtime.renderer = None

    async def _fallback_delivery(
        self,
        event: AgentEvent,
        *,
        renderer: TelegramTurnRenderer,
        sender: TelegramSender,
        chat_id: int,
        warning_sent: bool,
    ) -> bool:
        """Best-effort delivery after one renderer operation failed."""
        if isinstance(event, Final):
            # Remove stale placeholders/partial chunks where possible before
            # sending the authoritative response as fresh plain messages.
            for response_id in renderer.response_message_ids:
                try:
                    await sender.delete_message(chat_id, response_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
            await self._best_effort_plain(
                sender,
                chat_id,
                event.content or EMPTY_RESPONSE,
            )
            return True
        if isinstance(event, Error):
            await self._best_effort_plain(sender, chat_id, f"❌ {event.message}")
            return True
        if isinstance(event, TurnCancelled):
            await self._best_effort_plain(sender, chat_id, f"⏹️ {event.reason}")
            return True
        if not warning_sent:
            await self._best_effort_plain(
                sender,
                chat_id,
                "⚠️ Live Telegram updates were interrupted. "
                "The final response will be sent as a new message if needed.",
            )
        return True

    @staticmethod
    async def _best_effort_plain(
        sender: TelegramSender,
        chat_id: int,
        text: str,
    ) -> None:
        for chunk in chunk_text(text):
            try:
                await sender.send_message(chat_id, chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Exception strings may contain the bot token.
                _LOG.warning(
                    "Telegram fallback delivery failed with %s",
                    type(exc).__name__,
                )
                return

    async def _handle_command(
        self,
        command: str,
        argument: str,
        user_id: int,
        chat_id: int,
        sender: TelegramSender,
    ) -> None:
        if command in {"start", "help"}:
            await sender.send_message(chat_id, _HELP)
            return
        if command == "stop":
            await self._stop(user_id, chat_id, sender)
            return
        if command not in {"new", "sessions", "resume"}:
            await sender.send_message(chat_id, "Unknown command. Use /help.")
            return
        try:
            runtime = await self.runtime_for(user_id, chat_id, sender)
        except Exception as exc:
            await sender.send_message(
                chat_id,
                f"Could not start your LingCore runtime: "
                f"{self._safe_runtime_error(exc)}",
            )
            return
        if command == "sessions":
            await self._sessions(runtime, chat_id, sender)
            return
        active = self._user_tasks.get(user_id)
        if active is not None and not active.done():
            await sender.send_message(
                chat_id, "Cannot switch sessions while a turn is running."
            )
            return
        if command == "new":
            await self._replace_session(
                runtime, new_session_id(), chat_id, sender, resumed=False
            )
        else:
            if not argument:
                await sender.send_message(chat_id, "Usage: /resume <session-id-prefix>")
                return
            try:
                meta = runtime.store.resolve_prefix(argument.lower())
            except SessionError as exc:
                await sender.send_message(chat_id, str(exc))
                return
            await self._replace_session(
                runtime, meta.id, chat_id, sender, resumed=True
            )

    async def _sessions(
        self,
        runtime: TelegramUserRuntime,
        chat_id: int,
        sender: TelegramSender,
    ) -> None:
        sessions = runtime.store.list(limit=10)
        if not sessions:
            await sender.send_message(chat_id, "No stored sessions.")
            return
        lines = ["Your recent sessions:"]
        for meta in sessions:
            title = meta.title or "(untitled)"
            title = title.replace("\n", " ")[:80]
            lines.append(f"{meta.id[:8]} · {title} · {meta.message_count} messages")
        await sender.send_message(chat_id, "\n".join(lines))

    async def _replace_session(
        self,
        runtime: TelegramUserRuntime,
        session_id: str,
        chat_id: int,
        sender: TelegramSender,
        *,
        resumed: bool,
    ) -> None:
        async with runtime.lock:
            active = self._user_tasks.get(runtime.user_id)
            if active is not None and not active.done():
                await sender.send_message(
                    chat_id, "Cannot switch sessions while a turn is running."
                )
                return
            try:
                scoped, replacement = self._build_agent(
                    runtime.user_id,
                    runtime.user_dir,
                    runtime.store,
                    session_id,
                )
                self._state.set_active_session(runtime.user_id, session_id)
            except Exception as exc:
                await sender.send_message(
                    chat_id,
                    f"Could not switch sessions: {self._safe_runtime_error(exc)}",
                )
                return
            runtime.profile = scoped
            runtime.agent = replacement
        verb = "Resumed" if resumed else "Started"
        await sender.send_message(chat_id, f"{verb} session {session_id[:8]}.")

    async def _stop(
        self, user_id: int, chat_id: int, sender: TelegramSender
    ) -> None:
        runtime = self._runtimes.get(user_id)
        task = self._user_tasks.get(user_id)
        if runtime is None or task is None or task.done():
            await self.confirmations.deny_user(user_id)
            await sender.send_message(chat_id, "No active turn to stop.")
            return

        if runtime.final_delivery:
            # Final is already committed to memory/SQLite. Cancelling its
            # renderer would leave that durable response partially or wholly
            # unseen, so let authoritative delivery finish.
            await self.confirmations.deny_user(user_id)
            outcome = (await asyncio.gather(task, return_exceptions=True))[0]
            if isinstance(outcome, BaseException):
                await sender.send_message(
                    chat_id,
                    "The turn completed, but Telegram could not finish delivering "
                    "its final response.",
                )
            else:
                await sender.send_message(
                    chat_id,
                    "The turn had already completed; its final response was delivered.",
                )
            return

        agent_cancelled = runtime.agent.cancel_turn()
        task_cancelled = task.cancel()
        await self.confirmations.deny_user(user_id)
        await asyncio.gather(task, return_exceptions=True)
        if not agent_cancelled and not task_cancelled:
            await sender.send_message(chat_id, "No active turn to stop.")
            return
        if runtime.agent.turn_pending_finalization:
            try:
                event = runtime.agent.finalize_cancelled_turn()
            except Exception as exc:
                event = Error(f"failed to finalize stopped turn: {exc}")
        else:
            # Cancellation landed while the attachment was downloading or
            # before the Agent acquired its first checkpoint.
            event = TurnCancelled("stopped before the agent turn began")
        renderer = runtime.renderer
        runtime.renderer = None
        if renderer is not None:
            await renderer.handle(event)
        elif isinstance(event, Error):
            await sender.send_message(chat_id, f"❌ {event.message}")
        else:
            await sender.send_message(chat_id, f"⏹️ {event.reason}")

    def _build_agent(
        self,
        user_id: int,
        user_dir: Path,
        store: SessionStore,
        session_id: str,
    ) -> tuple["AgentProfile", Agent]:
        scoped = self.profile.model_copy(deep=True)
        # Be explicit even though pydantic's deep copy currently handles private
        # mappings: future model changes must not alias mutable tenant options.
        scoped.tool_options = copy.deepcopy(scoped.tool_options)
        workspace = user_dir / "workspace"
        memory_path = user_dir / "memory.md"
        sessions_path = user_dir / "sessions.db"
        absolute_state = Path(self.config.state_dir).expanduser().is_absolute()
        if absolute_state:
            configured_memory_path = str(memory_path)
            configured_sessions_path = str(sessions_path)
        else:
            source_dir = getattr(scoped, "_source_dir", None)
            if source_dir is None:  # guarded by Telegram config loading
                raise ConfigError("Telegram scoped profile has no source directory")
            configured_memory_path = str(
                memory_path.relative_to(source_dir.resolve())
            )
            configured_sessions_path = str(
                sessions_path.relative_to(source_dir.resolve())
            )
        scoped.workspace = str(workspace)
        session_updates: dict[str, Any] = {
            "enabled": True,
            "path": configured_sessions_path,
        }
        if absolute_state:
            # The state root already passed its single operator-consent gate;
            # all derived absolute files inherit that consent.
            session_updates["allow_absolute_path"] = True
        scoped.sessions = scoped.sessions.model_copy(update=session_updates)
        if "memory" in scoped.tools:
            raw_memory = scoped.tool_options.get("memory", {})
            if not isinstance(raw_memory, Mapping):
                raise ConfigError("tool_options.memory must be a mapping")
            memory_options = copy.deepcopy(dict(raw_memory))
            memory_options["path"] = configured_memory_path
            if absolute_state:
                memory_options["allow_absolute_path"] = True
            scoped.tool_options["memory"] = memory_options

        async def confirm(command: str) -> bool:
            sender = self._senders.get(user_id)
            chat_id = self._chat_ids.get(user_id)
            if sender is None or chat_id is None:
                return False
            return await self.confirmations.request(
                user_id, chat_id, command, sender
            )

        llm = self.llm_factory(user_id) if self.llm_factory is not None else None
        agent = Agent.from_profile(
            scoped,
            confirm=confirm,
            llm=llm,
            base_dir=Path.cwd(),
            tool_options=scoped.tool_options,
            session_store=store,
            session_id=session_id,
        )
        return scoped, agent

    def _user_directory(self, user_id: int) -> Path:
        users = self.config.state_path / "users"
        users.mkdir(parents=True, exist_ok=True)
        candidate = users / str(user_id)
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve()
        if not resolved.is_relative_to(users.resolve()):
            raise ConfigError(
                f"Telegram user state for {user_id} escapes the configured state root"
            )
        return resolved

    def _accept_update(self, update_id: int) -> bool:
        if update_id in self._updates:
            return False
        self._updates[update_id] = None
        if len(self._updates) > _UPDATE_CACHE_SIZE:
            self._updates.popitem(last=False)
        return True

    def _accept_album(self, chat_id: int, media_group_id: str) -> bool:
        now = self._clock()
        cutoff = now - _ALBUM_TTL_SECONDS
        while self._albums:
            _, oldest = next(iter(self._albums.items()))
            if oldest >= cutoff:
                break
            self._albums.popitem(last=False)
        key = (chat_id, media_group_id)
        if key in self._albums:
            return False
        self._albums[key] = now
        if len(self._albums) > _ALBUM_CACHE_SIZE:
            self._albums.popitem(last=False)
        return True

    @staticmethod
    def _command(text: str) -> tuple[str | None, str]:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None, ""
        head, _, tail = stripped.partition(" ")
        command = head[1:].split("@", 1)[0].lower()
        return command, tail.strip()

    def _task_done(self, user_id: int, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if self._user_tasks.get(user_id) is task:
            self._user_tasks.pop(user_id, None)
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            _LOG.error(
                "Telegram background turn failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    @staticmethod
    def _safe_runtime_error(exc: Exception) -> str:
        if isinstance(exc, (ConfigError, LingCoreError, OSError)):
            return str(exc)
        return type(exc).__name__
