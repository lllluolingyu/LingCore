"""Lark conversation isolation, commands, agent turns and cancellation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lingcore.agent import Agent
from lingcore.config import AgentProfile
from lingcore.errors import ConfigError, SessionError, ToolError
from lingcore.events import AgentEvent, Error, Final, TurnCancelled
from lingcore.integrations._profiles import scoped_profile
from lingcore.integrations.lark.config import LarkConfig
from lingcore.integrations.lark.confirmations import Confirmations
from lingcore.integrations.lark.protocol import LarkMessage, LarkSender
from lingcore.integrations.lark.rendering import deliver
from lingcore.integrations.lark.state import LarkStateStore
from lingcore.media import FILE_MAX_BYTES, IMAGE_MAX_BYTES, attachment_from_bytes
from lingcore.message import UserInput
from lingcore.plugins.commands import discover_commands
from lingcore.sessions import SessionStore, new_session_id

_LOG = logging.getLogger(__name__)
_RESERVED = {"start", "help", "new", "sessions", "resume", "stop", "approve", "deny"}
_HELP = """LingCore Lark / Feishu
Send text, a rich-text post, one image or one file.
/new — start a new session
/sessions — list recent sessions
/resume <id-prefix> — resume a session
/stop — cancel the active turn
/approve <token> or /deny <token> — answer a tool approval
/help — show this help
In groups, mention the bot. Replies are visible to everyone in the group."""

LLMFactory = Callable[[tuple[str, str]], Any]


@dataclass
class LarkRuntime:
    scope_id: str
    user_dir: Path
    store: SessionStore
    agent: Agent
    task: asyncio.Task[None] | None = None
    final_delivery: bool = False
    message: LarkMessage | None = None


class LarkBridge:
    def __init__(
        self,
        profile: AgentProfile,
        config: LarkConfig,
        sender: LarkSender,
        *,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        self.profile = profile
        self.config = config
        self.sender = sender
        self.llm_factory = llm_factory
        self.commands = discover_commands(profile)
        self.confirmations = Confirmations(config.confirmation_timeout)
        if (
            not (config.state_path / config.domain)
            .resolve()
            .is_relative_to(config.state_path)
        ):
            raise ConfigError("Lark domain state escapes the configured state root")
        self._state = LarkStateStore(
            config.state_path / config.domain / "bridge.sqlite3"
        )
        self._runtimes: dict[tuple[str, str], LarkRuntime] = {}
        self._updates: OrderedDict[str, None] = OrderedDict()
        self._tasks: set[asyncio.Task[None]] = set()
        self._dispatch_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    @property
    def background_tasks(self) -> frozenset[asyncio.Task[None]]:
        return frozenset(self._tasks)

    async def handle_message(self, message: LarkMessage) -> None:
        # Only admission and commands serialize here. Agent turns run in tasks,
        # leaving /stop and /approve reachable while a model/tool is waiting.
        async with self._dispatch_lock:
            if self._closing or not message.message_id or not message.chat_id:
                return
            if message.sender_type != "user" or not message.user_id:
                return
            if message.chat_type != "p2p" and not (
                message.chat_type == "group"
                and message.chat_id in self.config.allowed_chat_ids
                and message.mentioned_bot
            ):
                return
            if message.message_id in self._updates:
                return
            self._updates[message.message_id] = None
            if len(self._updates) > 4096:
                self._updates.popitem(last=False)
            if message.user_id not in self.config.allowed_user_ids:
                if message.chat_type == "p2p":
                    await self.sender.send_text(
                        message,
                        f"This bot is private. Your Lark open_id is {message.user_id}.",
                    )
                return
            if message.message_type not in {"text", "post", "image", "file"} or (
                len(message.resources) > 1
                or any(
                    resource.kind not in {"image", "file"}
                    for resource in message.resources
                )
            ):
                await self.sender.send_text(
                    message, "Send text, a rich-text post, one image or one file."
                )
                return
            text = message.text.strip()
            parts = text.split(maxsplit=1)
            command = parts[0][1:].lower() if text.startswith("/") else ""
            argument = parts[1] if len(parts) > 1 else ""
            if command in {"approve", "deny"}:
                accepted = self.confirmations.resolve(
                    message, argument, approve=command == "approve"
                )
                await self.sender.send_text(
                    message,
                    ("Approved." if command == "approve" else "Denied.")
                    if accepted
                    else "No matching pending approval for you in this chat.",
                )
                return
            if command in {"start", "help"}:
                extra = "\n".join(
                    f"{item.qualified_name} {item.argument_hint} — {item.description}"
                    for item in self.commands.commands
                    if item.name not in _RESERVED
                )
                await self.sender.send_text(
                    message, _HELP + ("\n\n" + extra if extra else "")
                )
                return
            if command == "stop":
                await self._stop(message)
                return
            if (
                command
                and command not in _RESERVED
                and self.commands.resolve(text, reserved=_RESERVED) is None
            ):
                await self.sender.send_text(message, "Unknown command. Use /help.")
                return
            if not text and not message.resources:
                return
            try:
                runtime = await self.runtime_for(message)
            except Exception as exc:
                _LOG.warning("Lark runtime creation failed: %s", type(exc).__name__)
                await self.sender.send_text(
                    message,
                    "Could not start your LingCore runtime. Check the server configuration.",
                )
                return
            if command in {"new", "sessions", "resume"}:
                await self._session_command(runtime, message, command, argument)
                return
            if runtime.task is not None and not runtime.task.done():
                await self.sender.send_text(
                    message,
                    "A turn is already running. Use /stop before sending another.",
                )
                return
            runtime.message = message
            task = asyncio.create_task(
                self._run_turn(runtime, message),
                name=f"lingcore-lark-{runtime.scope_id[:12]}",
            )
            runtime.task = task
            self._tasks.add(task)
            task.add_done_callback(self._task_done)

    async def runtime_for(self, message: LarkMessage) -> LarkRuntime:
        existing = self._runtimes.get(message.scope)
        if existing is not None:
            return existing
        scope_id = hashlib.sha256(json.dumps(message.scope).encode()).hexdigest()
        root = self.config.state_path / self.config.domain
        user_dir = root / "conversations" / scope_id
        if not user_dir.resolve().is_relative_to(root.resolve()):
            raise ConfigError(
                "Lark conversation state escapes the configured state root"
            )
        user_dir.mkdir(parents=True, exist_ok=True)
        store = SessionStore(user_dir / "sessions.db", profile_name=self.profile.name)
        agent = None
        try:
            session_id = self._state.get(scope_id) or new_session_id()
            agent = self._build_agent(message, user_dir, store, session_id)
            self._state.set(scope_id, session_id)
        except BaseException:
            try:
                if agent is not None:
                    await agent.aclose()
            finally:
                store.close()
            raise
        runtime = LarkRuntime(scope_id, user_dir, store, agent)
        self._runtimes[message.scope] = runtime
        return runtime

    def _build_agent(
        self, message: LarkMessage, user_dir: Path, store: SessionStore, session_id: str
    ) -> Agent:
        scoped = scoped_profile(
            self.profile, state_dir=self.config.state_dir, user_dir=user_dir
        )

        async def confirm(command: str) -> bool:
            runtime = self._runtimes.get(message.scope)
            if self._closing or runtime is None or runtime.message is None:
                return False
            return await self.confirmations.request(
                runtime.message, command, self.sender
            )

        return Agent.from_profile(
            scoped,
            confirm=confirm,
            llm=self.llm_factory(message.scope) if self.llm_factory else None,
            base_dir=Path.cwd(),
            tool_options=scoped.tool_options,
            session_store=store,
            session_id=session_id,
        )

    async def _session_command(
        self, runtime: LarkRuntime, message: LarkMessage, command: str, argument: str
    ) -> None:
        if command == "sessions":
            sessions = runtime.store.list(limit=10)
            active = runtime.agent.tool_ctx.session_id
            await self.sender.send_text(
                message,
                "\n".join(
                    f"{'*' if item.id == active else ' '} {item.id[:8]} — {item.title or '(untitled)'}"
                    for item in sessions
                )
                or "No stored sessions.",
            )
            return
        if runtime.task is not None and not runtime.task.done():
            await self.sender.send_text(
                message, "Cannot switch sessions while a turn is running."
            )
            return
        if command == "resume" and not argument:
            await self.sender.send_text(message, "Usage: /resume <session-id-prefix>")
            return
        replacement = None
        try:
            session_id = (
                new_session_id()
                if command == "new"
                else runtime.store.resolve_prefix(argument.lower()).id
            )
            if session_id == runtime.agent.tool_ctx.session_id:
                await self.sender.send_text(
                    message, f"Resumed session {session_id[:8]}."
                )
                return
            replacement = self._build_agent(
                message, runtime.user_dir, runtime.store, session_id
            )
            self._state.set(runtime.scope_id, session_id)
        except Exception as exc:
            if replacement is not None:
                await replacement.aclose()
            text = (
                str(exc)
                if isinstance(exc, SessionError)
                else "Could not switch sessions. Check the server configuration."
            )
            await self.sender.send_text(message, text)
            return
        previous = runtime.agent
        runtime.agent = replacement
        await previous.aclose()
        await self.sender.send_text(
            message,
            f"{'Started' if command == 'new' else 'Resumed'} session {session_id[:8]}.",
        )

    async def _run_turn(self, runtime: LarkRuntime, message: LarkMessage) -> None:
        try:
            incoming = runtime.agent.commands.resolve(
                message.text, reserved=_RESERVED
            ) or UserInput(text=message.text)
            for resource in message.resources:
                data = await self.sender.download(message, resource)
                limit = IMAGE_MAX_BYTES if resource.kind == "image" else FILE_MAX_BYTES
                if len(data) > limit:
                    raise ToolError(f"attachment exceeds the {limit}-byte limit")
                name = resource.name or "attachment"
                if resource.kind == "image":
                    # Lark image descriptors have no filename. Infer the extension
                    # through LingCore's magic-byte allowlist before classification.
                    from lingcore.media_types import media_bytes_match

                    suffix = next(
                        (
                            extension
                            for mime, extension in (
                                ("image/png", ".png"),
                                ("image/jpeg", ".jpg"),
                                ("image/gif", ".gif"),
                                ("image/webp", ".webp"),
                            )
                            if media_bytes_match(data, mime)
                        ),
                        None,
                    )
                    if suffix is None:
                        raise ToolError("unsupported image format")
                    name = "image" + suffix
                incoming.attachments.append(attachment_from_bytes(data, name=name))
            async with aclosing(runtime.agent.run(incoming)) as turn:
                async for event in turn:
                    if isinstance(event, Final | Error):
                        runtime.final_delivery = True
                    await self._deliver(event, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOG.warning("Lark turn failed: %s", type(exc).__name__)
            detail = (
                str(exc)
                if isinstance(exc, ToolError)
                else "Could not complete the turn. Check the server logs."
            )
            await self._deliver(Error(detail), message)
        finally:
            runtime.final_delivery = False
            # Also repair a turn if an unexpected frontend exception stopped
            # iteration before Agent committed its terminal event.
            if runtime.agent.turn_pending_finalization:
                notices = runtime.agent.drain_plugin_notices()
                event = runtime.agent.finalize_cancelled_turn()
                for notice in notices:
                    await self._deliver(notice, message)
                runtime.agent.drain_usage()
                await self._deliver(event, message)

    async def _deliver(self, event: AgentEvent, message: LarkMessage) -> None:
        try:
            await deliver(event, message, self.sender)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # SDK errors may contain credentials. Never log their text/traceback.
            _LOG.warning(
                "Lark %s delivery failed: %s", type(event).__name__, type(exc).__name__
            )

    async def _stop(self, message: LarkMessage) -> None:
        runtime = self._runtimes.get(message.scope)
        self.confirmations.deny(message.scope)
        if runtime is None or runtime.task is None or runtime.task.done():
            await self.sender.send_text(message, "No active turn to stop.")
            return
        if runtime.final_delivery:
            await asyncio.gather(runtime.task, return_exceptions=True)
            await self.sender.send_text(message, "The turn had already completed.")
            return
        agent_cancelled = runtime.agent.cancel_turn()
        runtime.task.cancel()
        await asyncio.gather(runtime.task, return_exceptions=True)
        if not agent_cancelled:
            await self._deliver(
                TurnCancelled("stopped before the agent turn began"), message
            )

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            _LOG.warning(
                "Lark background turn failed: %s", type(task.exception()).__name__
            )

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closing = True
        async with self._dispatch_lock:
            if self._closed:
                return
            self.confirmations.deny()
            for runtime in self._runtimes.values():
                runtime.agent.cancel_turn()
                if runtime.task is not None and not runtime.task.done():
                    runtime.task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            try:
                for runtime in self._runtimes.values():
                    try:
                        await runtime.agent.aclose()
                    except Exception as exc:
                        _LOG.warning(
                            "Lark agent cleanup failed: %s", type(exc).__name__
                        )
                    finally:
                        runtime.store.close()
            finally:
                self._state.close()
                self._closed = True
