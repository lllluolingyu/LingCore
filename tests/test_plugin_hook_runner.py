from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lingcore.message import Attachment, ToolResult
from lingcore.plugins import (
    HookFactory,
    HookRunner,
    PluginHooks,
    ToolCallEvent,
    ToolDecision,
    ToolResultPatch,
    TurnEndEvent,
    UserMessageDecision,
    UserMessageEvent,
)
from lingcore.tools import ToolContext


def runner(tmp_path: Path, *hooks: type[PluginHooks], **kwargs):
    ctx = ToolContext(workspace=tmp_path, **kwargs)
    factories = [
        HookFactory(str(i), tmp_path, hook, str(i)) for i, hook in enumerate(hooks)
    ]
    return HookRunner(factories, ctx), ctx


def call() -> ToolCallEvent:
    return ToolCallEvent("c1", "read_file", {"path": "foo"})


async def test_plugin_context_is_isolated_frozen_and_scopes_environment(tmp_path):
    options = {"0": {"nested": {"items": [1, {"value": "original"}]}}}
    hooks, ctx = runner(
        tmp_path, PluginHooks, options=options, environment={"TOKEN": "scoped"}
    )
    instance = ctx.plugins["0"]
    options["0"]["nested"]["items"][1]["value"] = "changed"
    assert instance.ctx.options["nested"]["items"][1]["value"] == "original"
    assert instance.ctx.getenv("TOKEN") == "scoped"
    assert instance.ctx.workspace == tmp_path
    with pytest.raises(TypeError):
        instance.ctx.options["nested"]["items"][1]["value"] = "bad"
    with pytest.raises(FrozenInstanceError):
        instance.ctx.workspace = Path("/other")
    with pytest.raises(TypeError):
        ctx.plugins["new"] = instance
    await hooks.start()
    assert hooks.drain_notices() == []


async def test_decisions_run_in_order_and_first_block_stops_later_plugins(tmp_path):
    seen = []

    class Context(PluginHooks):
        async def user_message(self, event):
            seen.append(self.ctx.name)
            return UserMessageDecision.add_context(self.ctx.name)

    class Block(PluginHooks):
        async def user_message(self, event):
            seen.append(self.ctx.name)
            return UserMessageDecision.block("blocked input")

    hooks, _ = runner(tmp_path, Context, Context)
    decision = await hooks.user_message(UserMessageEvent("hello", "hello"))
    assert decision.action == "add_context"
    assert decision.context == "0\n\n1"
    hooks, _ = runner(tmp_path, Context, Block, Context)
    decision = await hooks.user_message(UserMessageEvent("hello", "hello"))
    assert decision.action == "block"
    assert decision.reason == "blocked input"
    assert seen == ["0", "1", "0", "1"]
    assert hooks.drain_notices()[-1].action == "blocked"


async def test_allow_cannot_override_a_later_denial(tmp_path):
    class Allow(PluginHooks):
        async def before_tool(self, event):
            return ToolDecision.allow()

    class Deny(PluginHooks):
        async def before_tool(self, event):
            return ToolDecision.deny("policy")

    hooks, _ = runner(tmp_path, Allow, Deny, Allow)
    decision = await hooks.before_tool(call())
    assert decision.action == "deny"
    assert decision.reason == "policy"
    assert hooks.drain_notices()[0].action == "denied"


@pytest.mark.parametrize("accepted", [True, False, None, "error"])
async def test_ask_uses_scoped_confirmation_and_fails_closed(tmp_path, accepted):
    prompts = []

    class Ask(PluginHooks):
        async def before_tool(self, event):
            return ToolDecision.ask("Approve plugin request?")

    async def confirm(prompt):
        prompts.append(prompt)
        if accepted == "error":
            raise ValueError("secret-token")
        return accepted

    hooks, _ = runner(tmp_path, Ask, confirm=confirm if accepted is not None else None)
    decision = await hooks.before_tool(call())
    assert decision is None if accepted is True else decision.action == "deny"
    assert prompts == ([] if accepted is None else ["Approve plugin request?"])
    notices = hooks.drain_notices()
    assert notices[0].action == "asked"
    assert "secret-token" not in repr(notices)


async def test_call_and_attachment_views_are_immutable_snapshots(tmp_path):
    args = {"values": [1, {"value": "original"}]}
    event = ToolCallEvent("c1", "read_file", args)
    args["values"][1]["value"] = "changed"
    assert event.arguments["values"][1]["value"] == "original"
    with pytest.raises(TypeError):
        event.arguments["values"][1]["value"] = "bad"
    attachment = Attachment(
        kind="text", media_type="text/plain", data="aGk=", name="a.txt"
    )
    user = UserMessageEvent("hi", "hi", [attachment])
    attachment.name = "changed.txt"
    assert user.attachments[0].name == "a.txt"
    with pytest.raises(FrozenInstanceError):
        user.attachments[0].name = "bad"


async def test_ordered_patches_see_previous_result_and_preserve_metadata(tmp_path):
    seen = []

    class Replace(PluginHooks):
        async def after_tool(self, event, result):
            return ToolResultPatch.replace("redacted")

    class Note(PluginHooks):
        async def after_tool(self, event, result):
            seen.append(result.content)
            with pytest.raises(FrozenInstanceError):
                result.content = "bad"
            with pytest.raises(FrozenInstanceError):
                result.attachments[0].name = "bad"
            return ToolResultPatch.append_note("checked")

    hooks, _ = runner(tmp_path, Replace, Note)
    attachment = Attachment(
        kind="text", media_type="text/plain", data="aGk=", name="a.txt"
    )
    original = ToolResult(
        call_id="c1",
        name="read_file",
        content="secret",
        ok=False,
        attachments=[attachment],
    )
    result = await hooks.after_tool(call(), original)
    assert result.content == "redacted\nchecked"
    assert seen == ["redacted"]
    assert (result.call_id, result.name, result.ok, result.attachments) == (
        "c1",
        "read_file",
        False,
        [attachment],
    )
    assert original.content == "secret"
    assert [n.action for n in hooks.drain_notices()] == ["modified", "modified"]


@pytest.mark.parametrize("mode", ["block", "ignore"])
@pytest.mark.parametrize(
    "hook", ["user_message", "before_tool", "after_tool", "turn_end"]
)
async def test_hook_errors_are_value_free_and_respect_mode(tmp_path, mode, hook):
    class Broken(PluginHooks):
        async def user_message(self, event):
            raise ValueError("secret-token")

        async def before_tool(self, event):
            raise ValueError("secret-token")

        async def after_tool(self, event, result):
            raise ValueError("secret-token")

        async def turn_end(self, event):
            raise ValueError("secret-token")

    ctx = ToolContext(tmp_path)
    hooks = HookRunner(
        [HookFactory("bad", tmp_path, Broken, "bad", on_hook_error=mode)], ctx
    )
    if hook == "user_message":
        result = await hooks.user_message(UserMessageEvent("hi", "hi"))
        assert result is None if mode == "ignore" else result.action == "block"
    elif hook == "before_tool":
        result = await hooks.before_tool(call())
        assert result is None if mode == "ignore" else result.action == "deny"
    elif hook == "after_tool":
        original = ToolResult(call_id="c1", name="read_file", content="secret output")
        result = await hooks.after_tool(call(), original)
        assert (result.content == "secret output") is (mode == "ignore")
        assert (result.call_id, result.name, result.ok) == ("c1", "read_file", True)
    else:
        await hooks.turn_end(TurnEndEvent("done"))
    notices = hooks.drain_notices()
    assert len(notices) == 1
    assert (notices[0].plugin, notices[0].hook, notices[0].action) == (
        "bad",
        hook,
        "failed",
    )
    assert "ValueError" in notices[0].message
    assert "secret" not in repr(notices)
    assert hooks.drain_notices() == []


async def test_timeout_blocks_and_cancellation_propagates(tmp_path):
    class Slow(PluginHooks):
        async def before_tool(self, event):
            await asyncio.Event().wait()

    ctx = ToolContext(tmp_path)
    hooks = HookRunner(
        [HookFactory("slow", tmp_path, Slow, "slow", hook_timeout=0.01)], ctx
    )
    assert (await hooks.before_tool(call())).action == "deny"
    assert "TimeoutError" in hooks.drain_notices()[0].message
    task = asyncio.create_task(hooks.before_tool(call()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert hooks.drain_notices() == []


async def test_start_retries_only_failed_instances_and_close_is_reverse_idempotent(
    tmp_path,
):
    seen = []

    class Lifecycle(PluginHooks):
        async def start(self):
            seen.append("start:" + self.ctx.name)
            if self.ctx.name == "1" and seen.count("start:1") == 1:
                raise ValueError("secret-token")

        async def aclose(self):
            seen.append("close:" + self.ctx.name)
            if self.ctx.name == "1":
                raise ValueError("secret-token")

    hooks, _ = runner(tmp_path, Lifecycle, Lifecycle, Lifecycle)
    with pytest.raises(RuntimeError, match="ValueError") as exc:
        await hooks.start()
    assert "secret-token" not in str(exc.value)
    await hooks.start()
    await hooks.start()
    await hooks.aclose()
    await hooks.aclose()
    assert seen == [
        "start:0",
        "start:1",
        "start:1",
        "start:2",
        "close:2",
        "close:1",
        "close:0",
    ]
    notices = hooks.drain_notices()
    assert [n.hook for n in notices] == ["start", "aclose"]
    assert "secret-token" not in repr(notices)


async def test_context_and_result_patch_sizes_are_bounded(tmp_path):
    class Oversized(PluginHooks):
        async def user_message(self, event):
            return UserMessageDecision.add_context("x" * 20000)

        async def after_tool(self, event, result):
            return ToolResultPatch(content="x" * 20000, note="y" * 20000)

    hooks, _ = runner(tmp_path, Oversized, Oversized)
    decision = await hooks.user_message(UserMessageEvent("hi", "hi"))
    assert len(decision.context) <= 16000
    result = await hooks.after_tool(
        call(), ToolResult(call_id="c1", name="read_file", content="original")
    )
    # Replacement content and each note are capped independently.
    assert result.content == "x" * 16000 + "\n" + "y" * 16000


async def test_note_never_truncates_long_tool_output(tmp_path):
    class Note(PluginHooks):
        async def after_tool(self, event, result):
            return ToolResultPatch.append_note("AUDITED")

    hooks, _ = runner(tmp_path, Note)
    original = "x" * 20000
    result = await hooks.after_tool(
        call(), ToolResult(call_id="c1", name="read_file", content=original)
    )
    assert result.content == original + "\nAUDITED"


async def test_ask_waits_for_human_beyond_hook_timeout(tmp_path):
    class Ask(PluginHooks):
        async def before_tool(self, event):
            return ToolDecision.ask("Approve?")

    async def slow_confirm(prompt):
        await asyncio.sleep(0.2)
        return True

    ctx = ToolContext(tmp_path, confirm=slow_confirm)
    hooks = HookRunner(
        [HookFactory("ask", tmp_path, Ask, "ask", hook_timeout=0.05)], ctx
    )
    assert await hooks.before_tool(call()) is None


async def test_confirmation_failure_denies_even_with_ignore_mode(tmp_path):
    class Ask(PluginHooks):
        async def before_tool(self, event):
            return ToolDecision.ask("Approve?")

    async def confirm(prompt):
        raise RuntimeError("secret-token")

    ctx = ToolContext(tmp_path, confirm=confirm)
    hooks = HookRunner(
        [HookFactory("ask", tmp_path, Ask, "ask", on_hook_error="ignore")], ctx
    )
    assert (await hooks.before_tool(call())).action == "deny"
    assert "secret-token" not in repr(hooks.drain_notices())


async def test_previous_ask_does_not_make_later_ignored_failure_deny(tmp_path):
    class Ask(PluginHooks):
        async def before_tool(self, event):
            return ToolDecision.ask("Approve?")

    class Broken(PluginHooks):
        async def before_tool(self, event):
            raise RuntimeError("private")

    async def confirm(prompt):
        return True

    ctx = ToolContext(tmp_path, confirm=confirm)
    hooks = HookRunner(
        [
            HookFactory("ask", tmp_path, Ask, "ask"),
            HookFactory("bad", tmp_path, Broken, "bad", on_hook_error="ignore"),
        ],
        ctx,
    )
    assert await hooks.before_tool(call()) is None


@pytest.mark.parametrize(
    "hook", ["start", "user_message", "before_tool", "after_tool", "turn_end", "aclose"]
)
async def test_every_hook_preserves_explicit_cancellation(tmp_path, hook):
    class Cancel(PluginHooks):
        async def start(self):
            raise asyncio.CancelledError()

        async def user_message(self, event):
            raise asyncio.CancelledError()

        async def before_tool(self, event):
            raise asyncio.CancelledError()

        async def after_tool(self, event, result):
            raise asyncio.CancelledError()

        async def turn_end(self, event):
            raise asyncio.CancelledError()

        async def aclose(self):
            raise asyncio.CancelledError()

    hooks, _ = runner(tmp_path, Cancel)
    args = {
        "start": (),
        "user_message": (UserMessageEvent("hi", "hi"),),
        "before_tool": (call(),),
        "after_tool": (
            call(),
            ToolResult(call_id="c1", name="read_file", content="out"),
        ),
        "turn_end": (TurnEndEvent("done"),),
        "aclose": (),
    }
    with pytest.raises(asyncio.CancelledError):
        await getattr(hooks, hook)(*args[hook])
    assert hooks.drain_notices() == []


async def test_close_resumes_after_cancellation_without_reclosing_completed_instances(
    tmp_path,
):
    seen = []

    class Close(PluginHooks):
        async def aclose(self):
            seen.append(self.ctx.name)
            if self.ctx.name == "1" and seen.count("1") == 1:
                raise asyncio.CancelledError()

    hooks, _ = runner(tmp_path, Close, Close, Close)
    with pytest.raises(asyncio.CancelledError):
        await hooks.aclose()
    await hooks.aclose()
    await hooks.aclose()
    assert seen == ["2", "1", "1", "0"]


async def test_parallel_tool_calls_share_one_instance_without_serializing_hooks(
    tmp_path,
):
    entered = asyncio.Event()

    class Parallel(PluginHooks):
        async def before_tool(self, event):
            if event.call_id == "first":
                await entered.wait()
            else:
                entered.set()
            return ToolDecision.deny(self.ctx.name + ":" + event.call_id)

    hooks, ctx = runner(tmp_path, Parallel)
    async with asyncio.timeout(1):
        first, second = await asyncio.gather(
            hooks.before_tool(ToolCallEvent("first", "read_file", {})),
            hooks.before_tool(ToolCallEvent("second", "read_file", {})),
        )
    assert (first.reason, second.reason) == ("0:first", "0:second")
    assert len(ctx.plugins) == 1


@pytest.mark.parametrize("hook", ["user_message", "before_tool", "after_tool"])
async def test_malformed_hook_return_is_a_safe_failure(tmp_path, hook):
    class Malformed(PluginHooks):
        async def user_message(self, event):
            return "bad"

        async def before_tool(self, event):
            return "bad"

        async def after_tool(self, event, result):
            return "bad"

    hooks, _ = runner(tmp_path, Malformed)
    if hook == "user_message":
        assert (
            await hooks.user_message(UserMessageEvent("hi", "hi"))
        ).action == "block"
    elif hook == "before_tool":
        assert (await hooks.before_tool(call())).action == "deny"
    else:
        assert (
            "private output"
            not in (
                await hooks.after_tool(
                    call(),
                    ToolResult(
                        call_id="c1", name="read_file", content="private output"
                    ),
                )
            ).content
        )
    assert "TypeError" in hooks.drain_notices()[0].message


@pytest.mark.parametrize("mode", ["block", "ignore"])
@pytest.mark.parametrize(
    "payload",
    [
        ToolResultPatch(content=["private"]),
        ToolResultPatch(content={"private": "value"}),
        ToolResultPatch(content=0),
        ToolResultPatch(note=["private"]),
        ToolResultPatch(note={"private": "value"}),
        ToolResultPatch(note=0),
        ToolResultPatch(content="safe", note=False),
    ],
)
async def test_malformed_patch_fields_follow_error_policy(tmp_path, mode, payload):
    class Malformed(PluginHooks):
        async def after_tool(self, event, result):
            return payload

    ctx = ToolContext(tmp_path)
    hooks = HookRunner(
        [HookFactory("bad", tmp_path, Malformed, "bad", on_hook_error=mode)], ctx
    )
    original = ToolResult(call_id="c1", name="read_file", content="original", ok=False)
    result = await hooks.after_tool(call(), original)
    assert isinstance(result.content, str)
    assert (result.content == "original") is (mode == "ignore")
    assert (result.call_id, result.name, result.ok) == ("c1", "read_file", False)
    notices = hooks.drain_notices()
    assert len(notices) == 1
    assert notices[0].action == "failed"
    assert "TypeError" in notices[0].message
    assert "private" not in repr(notices)
    assert original.content == "original"


@pytest.mark.parametrize("mode", ["block", "ignore"])
@pytest.mark.parametrize(
    "payload",
    [
        UserMessageDecision("block", reason=["private"]),
        UserMessageDecision("block", reason=None),
        UserMessageDecision("add_context", context=["private"]),
        UserMessageDecision("add_context", context=0),
        UserMessageDecision("add_context", reason=False, context="safe"),
        UserMessageDecision("block", reason="safe", context=None),
    ],
)
async def test_malformed_user_decision_fields_follow_error_policy(
    tmp_path, mode, payload
):
    class Malformed(PluginHooks):
        async def user_message(self, event):
            return payload

    ctx = ToolContext(tmp_path)
    hooks = HookRunner(
        [HookFactory("bad", tmp_path, Malformed, "bad", on_hook_error=mode)], ctx
    )
    decision = await hooks.user_message(UserMessageEvent("hi", "hi"))
    assert decision is None if mode == "ignore" else decision.action == "block"
    if decision is not None:
        assert isinstance(decision.reason, str)
    notices = hooks.drain_notices()
    assert len(notices) == 1
    assert notices[0].action == "failed"
    assert "TypeError" in notices[0].message
    assert "private" not in repr(notices)


@pytest.mark.parametrize("mode", ["block", "ignore"])
@pytest.mark.parametrize(
    "payload",
    [
        ToolDecision("deny", reason=["private"]),
        ToolDecision("deny", reason=None),
        ToolDecision("ask", prompt=["private"]),
        ToolDecision("ask", prompt=0),
        ToolDecision("allow", reason=False),
        ToolDecision("allow", prompt=None),
    ],
)
async def test_malformed_tool_decision_fields_follow_error_policy(
    tmp_path, mode, payload
):
    prompts = []

    class Malformed(PluginHooks):
        async def before_tool(self, event):
            return payload

    async def confirm(prompt):
        prompts.append(prompt)
        return True

    ctx = ToolContext(tmp_path, confirm=confirm)
    hooks = HookRunner(
        [HookFactory("bad", tmp_path, Malformed, "bad", on_hook_error=mode)], ctx
    )
    decision = await hooks.before_tool(call())
    assert decision is None if mode == "ignore" else decision.action == "deny"
    if decision is not None:
        assert isinstance(decision.reason, str)
    assert prompts == []
    notices = hooks.drain_notices()
    assert len(notices) == 1
    assert notices[0].action == "failed"
    assert "TypeError" in notices[0].message
    assert "private" not in repr(notices)
