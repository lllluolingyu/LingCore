"""Immutable hook inputs and ordered, per-Agent plugin execution.

Plugins may receive parallel tool calls on the same instance. They own any
locking their state needs. Hooks only narrow tool policy; an ``allow`` never
grants a tool or suppresses the tool's own confirmation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from lingcore.events import PluginNotice
from lingcore.message import Attachment, ToolResult
from lingcore.tools import ToolContext

MAX_CONTEXT_CHARS = 16_000
MAX_RESULT_PATCH_CHARS = 16_000


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class AttachmentView:
    kind: str
    media_type: str
    data: str = field(repr=False)
    name: str | None = None
    fallback_text: str | None = None

    @classmethod
    def from_attachment(cls, attachment: Attachment) -> AttachmentView:
        return cls(
            attachment.kind,
            attachment.media_type,
            attachment.data,
            attachment.name,
            attachment.fallback_text,
        )


def _attachments(
    attachments: Sequence[Attachment | AttachmentView],
) -> tuple[AttachmentView, ...]:
    return tuple(
        item
        if isinstance(item, AttachmentView)
        else AttachmentView.from_attachment(item)
        for item in attachments
    )


class _GetenvFn(Protocol):
    def __call__(self, name: str, default: str | None = None) -> str | None: ...


@dataclass(frozen=True, slots=True)
class PluginContext:
    name: str
    root: Path
    workspace: Path
    profile_dir: Path | None
    session_id: str | None
    options: Mapping[str, Any] = field(repr=False)
    getenv: _GetenvFn = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _freeze(self.options))


@dataclass(frozen=True, slots=True)
class UserMessageEvent:
    text: str
    input_text: str
    attachments: tuple[AttachmentView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attachments", _attachments(self.attachments))


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolResultView:
    call_id: str
    name: str
    content: str
    ok: bool
    attachments: tuple[AttachmentView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attachments", _attachments(self.attachments))

    @classmethod
    def from_result(cls, result: ToolResult) -> ToolResultView:
        return cls(
            result.call_id,
            result.name,
            result.content,
            result.ok,
            _attachments(result.attachments),
        )


@dataclass(frozen=True, slots=True)
class TurnEndEvent:
    content: str
    error: str | None = None
    turn_index: int | None = None


@dataclass(frozen=True, slots=True)
class UserMessageDecision:
    action: Literal["block", "add_context"]
    reason: str = ""
    context: str = ""
    plugin: str | None = None

    @classmethod
    def block(cls, reason: str) -> UserMessageDecision:
        return cls("block", reason=reason)

    @classmethod
    def add_context(cls, text: str) -> UserMessageDecision:
        return cls("add_context", context=text)


@dataclass(frozen=True, slots=True)
class ToolDecision:
    action: Literal["allow", "deny", "ask"]
    reason: str = ""
    prompt: str = ""
    plugin: str | None = None

    @classmethod
    def allow(cls) -> ToolDecision:
        return cls("allow")

    @classmethod
    def deny(cls, reason: str) -> ToolDecision:
        return cls("deny", reason=reason)

    @classmethod
    def ask(cls, prompt: str) -> ToolDecision:
        return cls("ask", prompt=prompt)


@dataclass(frozen=True, slots=True)
class ToolResultPatch:
    content: str | None = None
    note: str | None = None

    @classmethod
    def replace(cls, content: str) -> ToolResultPatch:
        return cls(content=content)

    @classmethod
    def append_note(cls, note: str) -> ToolResultPatch:
        return cls(note=note)


class PluginHooks:
    """Subclass and override only the async hooks the plugin needs."""

    def __init__(self, ctx: PluginContext) -> None:
        self.ctx = ctx

    async def start(self) -> None:
        pass

    async def user_message(self, event: UserMessageEvent) -> UserMessageDecision | None:
        return None

    async def before_tool(self, event: ToolCallEvent) -> ToolDecision | None:
        return None

    async def after_tool(
        self, event: ToolCallEvent, result: ToolResultView
    ) -> ToolResultPatch | None:
        return None

    async def turn_end(self, event: TurnEndEvent) -> None:
        pass

    async def aclose(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class HookFactory:
    name: str
    root: Path
    hooks: type[PluginHooks]
    options_key: str
    on_hook_error: Literal["block", "ignore"] = "block"
    hook_timeout: float = 10

    def __post_init__(self) -> None:
        if self.on_hook_error not in ("block", "ignore"):
            raise ValueError("invalid hook error mode")
        if not 0 < self.hook_timeout <= 60:
            raise ValueError("hook timeout must be greater than zero and at most 60")


class HookRunner:
    """Execute overridden hooks in declaration order with bounded calls.

    Failed starts are retried on the next ``start``; instances that already
    started successfully are left alone. Close runs once per instance in reverse
    order, including instances whose startup failed. Cancellation propagates;
    a subsequent close resumes with the instances that have not finished closing.
    """

    def __init__(self, factories: Sequence[HookFactory], tool_ctx: ToolContext) -> None:
        self._ctx = tool_ctx
        self._entries: list[tuple[HookFactory, PluginHooks]] = []
        self._notices: list[PluginNotice] = []
        self._started: set[str] = set()
        self._closed: set[str] = set()
        self._closing = False
        self._lifecycle_lock = asyncio.Lock()
        plugins: dict[str, PluginHooks] = {}
        for factory in factories:
            if factory.name in plugins:
                raise ValueError("duplicate hook factory name")
            options = tool_ctx.options.get(factory.options_key, {})
            if not isinstance(options, Mapping):
                raise ValueError("plugin options must be a mapping")
            context = PluginContext(
                factory.name,
                factory.root,
                tool_ctx.workspace,
                tool_ctx.profile_dir,
                tool_ctx.session_id,
                options,
                tool_ctx.getenv,
            )
            instance = factory.hooks(context)
            plugins[factory.name] = instance
            self._entries.append((factory, instance))
        tool_ctx.plugins = MappingProxyType(plugins)

    def drain_notices(self) -> list[PluginNotice]:
        notices, self._notices = self._notices, []
        return notices

    def _notice(
        self,
        factory: HookFactory,
        hook: str,
        action: Literal["denied", "blocked", "asked", "modified", "failed"],
        message: str,
    ) -> None:
        self._notices.append(PluginNotice(factory.name, hook, action, message))

    def _failure(self, factory: HookFactory, hook: str, error: Exception) -> str:
        message = f"Plugin {factory.name} {hook} failed ({type(error).__name__})"
        self._notice(factory, hook, "failed", message)
        return message

    async def _invoke(
        self, factory: HookFactory, instance: PluginHooks, hook: str, *args: Any
    ) -> Any:
        if getattr(type(instance), hook) is getattr(PluginHooks, hook):
            return None
        async with asyncio.timeout(factory.hook_timeout):
            return await getattr(instance, hook)(*args)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closing:
                raise RuntimeError("plugin runner is closed")
            for factory, instance in self._entries:
                if factory.name in self._started:
                    continue
                try:
                    await self._invoke(factory, instance, "start")
                except Exception as error:
                    message = self._failure(factory, "start", error)
                    raise RuntimeError(message) from None
                self._started.add(factory.name)

    async def user_message(self, event: UserMessageEvent) -> UserMessageDecision | None:
        context = ""
        for factory, instance in self._entries:
            try:
                decision = await self._invoke(factory, instance, "user_message", event)
                if decision is None:
                    continue
                if (
                    not isinstance(decision, UserMessageDecision)
                    or not isinstance(decision.action, str)
                    or decision.action not in ("block", "add_context")
                ):
                    raise TypeError("invalid user message decision")
                if not isinstance(decision.reason, str) or not isinstance(
                    decision.context, str
                ):
                    raise TypeError("invalid user message decision fields")
                if decision.action == "block":
                    self._notice(factory, "user_message", "blocked", decision.reason)
                    return replace(decision, plugin=factory.name)
                context = (context + ("\n\n" if context else "") + decision.context)[
                    :MAX_CONTEXT_CHARS
                ]
                self._notice(factory, "user_message", "modified", "Added user context")
            except Exception as error:
                message = self._failure(factory, "user_message", error)
                if factory.on_hook_error == "block":
                    return UserMessageDecision(
                        "block", reason=message, plugin=factory.name
                    )
        return UserMessageDecision.add_context(context) if context else None

    async def before_tool(self, event: ToolCallEvent) -> ToolDecision | None:
        for factory, instance in self._entries:
            asking = False
            try:
                decision = await self._invoke(factory, instance, "before_tool", event)
                if decision is None:
                    continue
                if (
                    not isinstance(decision, ToolDecision)
                    or not isinstance(decision.action, str)
                    or decision.action not in ("allow", "deny", "ask")
                ):
                    raise TypeError("invalid tool decision")
                if not isinstance(decision.reason, str) or not isinstance(
                    decision.prompt, str
                ):
                    raise TypeError("invalid tool decision fields")
                if decision.action == "allow":
                    continue
                if decision.action == "ask":
                    asking = True
                    self._notice(factory, "before_tool", "asked", decision.prompt)
                    accepted = False
                    if self._ctx.confirm is not None:
                        # The hook timeout bounds plugin code, not the human:
                        # the frontend's confirmation handler owns that wait.
                        accepted = await self._ctx.confirm(decision.prompt) is True
                    if accepted:
                        continue
                    decision = ToolDecision.deny("Plugin confirmation declined")
                self._notice(factory, "before_tool", "denied", decision.reason)
                return replace(decision, plugin=factory.name)
            except Exception as error:
                message = self._failure(factory, "before_tool", error)
                # A failed confirmation never opts out of a requested approval,
                # even when hook execution errors are configured to be ignored.
                if factory.on_hook_error == "block" or asking:
                    return ToolDecision("deny", reason=message, plugin=factory.name)
        return None

    async def after_tool(self, event: ToolCallEvent, result: ToolResult) -> ToolResult:
        for factory, instance in self._entries:
            try:
                patch = await self._invoke(
                    factory,
                    instance,
                    "after_tool",
                    event,
                    ToolResultView.from_result(result),
                )
                if patch is None:
                    continue
                if not isinstance(patch, ToolResultPatch):
                    raise TypeError("invalid result patch")
                if (
                    patch.content is not None and not isinstance(patch.content, str)
                ) or (patch.note is not None and not isinstance(patch.note, str)):
                    raise TypeError("invalid result patch fields")
                # Cap only plugin-authored text: a note must never truncate the
                # tool's own output, nor be cut off by a long original result.
                content = (
                    result.content
                    if patch.content is None
                    else patch.content[:MAX_RESULT_PATCH_CHARS]
                )
                if patch.note:
                    note = patch.note[:MAX_RESULT_PATCH_CHARS]
                    content += ("\n" if content else "") + note
                if patch.content is not None or patch.note:
                    result = result.model_copy(update={"content": content})
                    self._notice(
                        factory, "after_tool", "modified", "Modified tool result"
                    )
            except Exception as error:
                message = self._failure(factory, "after_tool", error)
                if factory.on_hook_error == "block":
                    result = result.model_copy(update={"content": message})
                    return result
        return result

    async def turn_end(self, event: TurnEndEvent) -> None:
        for factory, instance in self._entries:
            try:
                await self._invoke(factory, instance, "turn_end", event)
            except Exception as error:
                self._failure(factory, "turn_end", error)

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            self._closing = True
            for factory, instance in reversed(self._entries):
                if factory.name in self._closed:
                    continue
                try:
                    await self._invoke(factory, instance, "aclose")
                except Exception as error:
                    self._failure(factory, "aclose", error)
                self._closed.add(factory.name)
