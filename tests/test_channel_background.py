from __future__ import annotations

import asyncio

import pytest

from lingcore.background import EventSource
from lingcore.integrations.telegram.protocol import TelegramMessage
from tests.support import lark_bridge as lark
from tests.support import telegram_bridge as telegram


@pytest.mark.parametrize("channel", ["lark", "telegram"])
async def test_channel_drains_background_runs_and_can_confirm_while_idle(
    tmp_path, monkeypatch, channel
):
    if channel == "lark":
        _, _, bridge, sender = lark.setup_bridge(
            tmp_path,
            monkeypatch,
            profile_text=lark.PROFILE.replace("[memory]", "[delegate]"),
        )
        message = lark.message()
        await bridge.handle_message(message)
        await lark.finish(bridge)
        runtime = await bridge.runtime_for(message)
    else:
        _, _, bridge = telegram._setup(
            tmp_path,
            monkeypatch,
            profile_text=telegram.PROFILE.replace("[memory]", "[delegate]"),
        )
        sender = telegram.FakeSender()
        await bridge.handle_message(
            TelegramMessage(1, 11, 11, "private", "hello"), sender
        )
        await telegram._finish(bridge)
        runtime = bridge._runtimes[11]
    try:
        agent = runtime.agent
        assert agent.tool_ctx.supports_async_confirm
        broker = agent.tool_ctx.delegation.broker
        assert broker.async_confirm
        before = len(agent.llm.calls)
        source = EventSource(agent.background, "r-completed", kind="run")
        source.publish("Worker finished its requested task.")
        source.finish()
        async with asyncio.timeout(2):
            while len(agent.llm.calls) == before:
                await asyncio.sleep(0.01)
        assert "Worker finished" in agent.llm.calls[-1][-1].content
        await asyncio.gather(*bridge.background_tasks)
        approval = asyncio.create_task(
            broker.for_worker("r-test", "worker", asyncio.Event(), background=True)(
                "Review this request"
            )
        )
        async with asyncio.timeout(2):
            while True:
                pending = (
                    bridge.confirmations.pending
                    if channel == "lark"
                    else bridge.confirmations._pending
                )
                if pending:
                    break
                assert not approval.done(), (
                    "background confirmation was silently refused"
                )
                await asyncio.sleep(0)
        if channel == "lark":
            bridge.confirmations.deny()
        else:
            await bridge.confirmations.deny_all()
        assert await approval is False
    finally:
        await bridge.shutdown()
