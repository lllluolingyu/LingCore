"""Cancellation and per-user approval lifecycles."""

from __future__ import annotations

import asyncio

import pytest

from lingcore.integrations.lark.confirmations import Confirmations
from lingcore.integrations.lark.protocol import LarkResource
from lingcore.llm import LLMChunk
from tests.support.lark_bridge import FakeSender, finish, message, setup_bridge


async def test_stop_during_model_and_busy_commands(tmp_path, monkeypatch):
    started = asyncio.Event()

    class BlockingLLM:
        async def stream(self, messages, tools=None):
            started.set()
            await asyncio.Event().wait()
            yield LLMChunk(text_delta="unreachable")

    _, _, bridge, sender = setup_bridge(
        tmp_path, monkeypatch, llm_factory=lambda _: BlockingLLM()
    )
    try:
        await bridge.handle_message(message())
        await asyncio.wait_for(started.wait(), 2)
        runtime = await bridge.runtime_for(message())
        for idx, text in enumerate(("/new", "/resume abc", "another prompt")):
            await bridge.handle_message(message(f"om_busy{idx}", text))
            assert "running" in sender.sent[-1][1]
        await bridge.handle_message(message("om_stop", "/stop"))
        assert not runtime.agent.turn_pending_finalization
        assert [(m.role, m.content) for m in runtime.agent.memory.messages] == [
            ("user", "hello")
        ]
        assert any("Stopped" in text for _, text in sender.sent)
        assert not bridge.background_tasks
    finally:
        await bridge.shutdown()


async def test_stop_download_and_shutdown_are_idempotent(tmp_path, monkeypatch):
    _, _, bridge, sender = setup_bridge(tmp_path, monkeypatch)
    sender.download_gate = asyncio.Event()
    await bridge.handle_message(
        message(message_type="file", resources=(LarkResource("f", "file", "f.txt"),))
    )
    await asyncio.wait_for(sender.download_started.wait(), 2)
    await bridge.handle_message(message("om_stop", "/stop"))
    assert "before the agent turn began" in sender.sent[-1][1]
    assert not bridge.background_tasks
    await bridge.shutdown()
    await bridge.shutdown()
    count = len(sender.sent)
    await bridge.handle_message(message("om_late"))
    assert len(sender.sent) == count


async def test_stop_during_final_delivery_preserves_committed_reply(
    tmp_path, monkeypatch
):
    entered, gate = asyncio.Event(), asyncio.Event()

    class SlowSender(FakeSender):
        async def send_text(self, msg, text):
            if text == "reply":
                entered.set()
                await gate.wait()
            await super().send_text(msg, text)

    _, _, bridge, sender = setup_bridge(tmp_path, monkeypatch, sender=SlowSender())
    try:
        await bridge.handle_message(message())
        await asyncio.wait_for(entered.wait(), 2)
        stopping = asyncio.create_task(
            bridge.handle_message(message("om_stop", "/stop"))
        )
        await asyncio.sleep(0)
        assert not stopping.done()
        gate.set()
        await asyncio.wait_for(stopping, 2)
        assert [text for _, text in sender.sent] == [
            "reply",
            "The turn had already completed.",
        ]
        assert len((await bridge.runtime_for(message())).agent.memory.messages) == 2
    finally:
        gate.set()
        await bridge.shutdown()


async def test_failed_delivery_keeps_committed_turn_and_hides_exception(
    tmp_path, monkeypatch, caplog
):
    class BrokenSender(FakeSender):
        async def send_text(self, message, text):
            raise RuntimeError("secret-SENTINEL")

    _, _, bridge, _ = setup_bridge(tmp_path, monkeypatch, sender=BrokenSender())
    try:
        await bridge.handle_message(message())
        await finish(bridge)
        runtime = await bridge.runtime_for(message())
        assert len(runtime.store.messages(runtime.agent.tool_ctx.session_id)) == 2
        assert "SENTINEL" not in caplog.text
    finally:
        await bridge.shutdown()


@pytest.mark.parametrize("incomplete", [False, True])
async def test_stop_in_end_hook_delivers_committed_terminal(
    tmp_path, monkeypatch, incomplete
):
    from lingcore.plugins.hooks import HookRunner
    from tests.fakes import FakeLLMClient, ScriptedTurn

    entered = asyncio.Event()

    class SlowHooks(HookRunner):
        async def turn_end(self, event):
            entered.set()
            await asyncio.Event().wait()

    _, _, bridge, sender = setup_bridge(
        tmp_path,
        monkeypatch,
        llm_factory=lambda _: FakeLLMClient(
            [
                ScriptedTurn(
                    text="committed reply",
                    finish_reason="length" if incomplete else "stop",
                )
            ]
        ),
    )
    runtime = await bridge.runtime_for(message())
    runtime.agent._hooks = SlowHooks([], runtime.agent.tool_ctx)
    try:
        await bridge.handle_message(message())
        await asyncio.wait_for(entered.wait(), 2)
        await bridge.handle_message(message("om_stop", "/stop"))
        assert "committed reply" in sender.sent[-1][1]
        assert ("incomplete" in sender.sent[-1][1].lower()) is incomplete
        assert not runtime.agent.turn_pending_finalization
        assert (
            runtime.store.messages(runtime.agent.tool_ctx.session_id)[-1].content
            == "committed reply"
        )
    finally:
        await bridge.shutdown()


@pytest.mark.parametrize("approve", [True, False])
async def test_approvals_bind_user_chat_and_are_one_use(approve):
    sender = FakeSender()
    confirmations = Confirmations(30)
    task = asyncio.create_task(confirmations.request(message(), "echo hello", sender))
    await sender.sent_event.wait()
    token = next(iter(confirmations.pending))
    for wrong in (message(user_id="ou_bob"), message(chat_id="oc_group")):
        assert not confirmations.resolve(wrong, token, approve=True)
    assert not task.done()
    assert confirmations.resolve(message(), token, approve=approve)
    assert not confirmations.resolve(message(), token, approve=approve)
    assert await task is approve
    assert not confirmations.pending


@pytest.mark.parametrize("cancel", [False, True])
async def test_expiration_and_cancellation_deny_pending_approvals(cancel):
    sender = FakeSender()
    confirmations = Confirmations(30 if cancel else 0.01)
    task = asyncio.create_task(confirmations.request(message(), "operation", sender))
    await sender.sent_event.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await task is False
        assert "expired" in sender.sent[-1][1]
    assert not confirmations.pending


async def test_bridge_dispatches_approval_while_turn_is_waiting(tmp_path, monkeypatch):
    _, _, bridge, sender = setup_bridge(tmp_path, monkeypatch)
    runtime = await bridge.runtime_for(message())
    runtime.message = message()
    task = asyncio.create_task(runtime.agent.tool_ctx.confirm("echo hello"))
    try:
        await sender.sent_event.wait()
        token = next(iter(bridge.confirmations.pending))
        await bridge.handle_message(
            message("om_wrong", f"/approve {token}", user_id="ou_bob")
        )
        assert not task.done()
        await bridge.handle_message(message("om_ok", f"/approve {token}"))
        assert await asyncio.wait_for(task, 1) is True
        await bridge.handle_message(message("om_repeat", f"/approve {token}"))
        assert "No matching" in sender.sent[-1][1]
    finally:
        await bridge.shutdown()


async def test_shutdown_denies_pending_confirmation(tmp_path, monkeypatch):
    _, _, bridge, sender = setup_bridge(tmp_path, monkeypatch)
    runtime = await bridge.runtime_for(message())
    runtime.message = message()
    task = asyncio.create_task(runtime.agent.tool_ctx.confirm("operation"))
    await sender.sent_event.wait()
    await bridge.shutdown()
    assert await task is False
