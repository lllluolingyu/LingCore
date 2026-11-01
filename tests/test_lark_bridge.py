"""Offline agent turns, admission, durable session isolation and media."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lingcore.guardrails import NoopGuardrail
from lingcore.integrations.lark.bridge import LarkBridge
from lingcore.integrations.lark.protocol import LarkResource
from tests.fakes import FakeLLMClient, ScriptedTurn, StreamFailure
from tests.support.lark_bridge import PROFILE, finish, message, setup_bridge


async def test_admission_dedup_and_untrusted_users_never_start_agent(
    tmp_path, monkeypatch
):
    _, _, bridge, sender = setup_bridge(tmp_path, monkeypatch)
    try:
        for item in (
            message(user_id="ou_unknown"),
            message(user_id="ou_unknown"),
            message("om_bot", sender_type="app"),
            message("om_missing", sender_type=None),
            message(
                "om_group", chat_type="group", chat_id="oc_unknown", mentioned_bot=True
            ),
            message("om_no_mention", chat_type="group", chat_id="oc_group"),
            message(
                "om_unknown_group_user",
                user_id="ou_unknown",
                chat_type="group",
                chat_id="oc_group",
                mentioned_bot=True,
            ),
        ):
            await bridge.handle_message(item)
        assert not bridge._runtimes
        assert len(sender.sent) == 1 and "ou_unknown" in sender.sent[0][1]
        accepted = message("om_accepted")
        await bridge.handle_message(accepted)
        await finish(bridge)
        await bridge.handle_message(accepted)
        await finish(bridge)
        assert [text for _, text in sender.sent].count("reply") == 1
    finally:
        await bridge.shutdown()


async def test_conversations_users_and_mutable_files_are_isolated(
    tmp_path, monkeypatch
):
    _, _, bridge, _ = setup_bridge(tmp_path, monkeypatch)
    try:
        messages = [
            message(),
            message("om_b", user_id="ou_bob"),
            message("om_g", chat_type="group", chat_id="oc_group", mentioned_bot=True),
        ]
        for item in messages:
            await bridge.handle_message(item)
        await finish(bridge)
        runtimes = [(await bridge.runtime_for(item)) for item in messages]
        assert len({runtime.user_dir for runtime in runtimes}) == 3
        assert len({runtime.agent.tool_ctx.session_id for runtime in runtimes}) == 3
        paths = [
            runtime.agent.tool_ctx.options["memory"]["path"] for runtime in runtimes
        ]
        assert len(set(paths)) == 3
        for runtime in runtimes:
            assert runtime.user_dir.is_relative_to(bridge.config.state_path)
            assert len(runtime.agent.memory.messages) == 2
    finally:
        await bridge.shutdown()


async def test_session_commands_and_explicit_selection_survive_restart(
    tmp_path, monkeypatch
):
    profile, config, bridge, sender = setup_bridge(tmp_path, monkeypatch)
    await bridge.handle_message(message())
    await finish(bridge)
    first = (await bridge.runtime_for(message())).agent.tool_ctx.session_id
    original_agent = (await bridge.runtime_for(message())).agent
    await bridge.handle_message(message("om_same", f"/resume {first[:8]}"))
    assert (await bridge.runtime_for(message())).agent is original_agent
    await bridge.handle_message(message("om_new", "/new"))
    second = (await bridge.runtime_for(message())).agent.tool_ctx.session_id
    assert first != second
    await bridge.handle_message(message("om_next", "next"))
    await finish(bridge)
    await bridge.handle_message(message("om_sessions", "/sessions"))
    assert first[:8] in sender.sent[-1][1] and second[:8] in sender.sent[-1][1]
    await bridge.handle_message(message("om_resume", f"/resume {first[:8]}"))
    await bridge.shutdown()
    restarted = LarkBridge(
        profile, config, sender, llm_factory=lambda _: FakeLLMClient([])
    )
    try:
        assert (
            await restarted.runtime_for(message())
        ).agent.tool_ctx.session_id == first
        await restarted.handle_message(
            message("om_other", f"/resume {first[:8]}", user_id="ou_bob")
        )
        assert (
            await restarted.runtime_for(message(user_id="ou_bob"))
        ).agent.tool_ctx.session_id != first
    finally:
        await restarted.shutdown()


async def test_clock_context_is_applied_without_chat_diagnostics(tmp_path, monkeypatch):
    llm = FakeLLMClient([ScriptedTurn(text="OK")])
    _, _, bridge, sender = setup_bridge(
        tmp_path,
        monkeypatch,
        profile_text=PROFILE + "plugins: [clock]\n",
        llm_factory=lambda _: llm,
    )
    incoming = message(text="reply OK")
    try:
        await bridge.handle_message(incoming)
        await finish(bridge)
        user = next(item for item in llm.calls[0] if item.role == "user")
        assert user.content.startswith("reply OK\n[clock] Now: ")
        assert user.input_text == "reply OK"
        assert sender.sent == [(incoming, "OK")]
    finally:
        await bridge.shutdown()


@pytest.mark.parametrize("incomplete", [False, True])
async def test_only_guarded_terminal_content_is_visible(
    tmp_path, monkeypatch, incomplete
):
    class Redactor(NoopGuardrail):
        async def post_output(self, text):
            return text.replace("SECRET", "redacted")

    _, _, bridge, sender = setup_bridge(
        tmp_path,
        monkeypatch,
        llm_factory=lambda _: FakeLLMClient(
            [
                ScriptedTurn(
                    text="SECRET", finish_reason="length" if incomplete else "stop"
                )
            ]
        ),
    )
    try:
        (await bridge.runtime_for(message())).agent.guardrail = Redactor()
        await bridge.handle_message(message())
        await finish(bridge)
        assert "redacted" in sender.sent[-1][1]
        assert all("SECRET" not in text for _, text in sender.sent)
        if incomplete:
            assert "incomplete" in sender.sent[-1][1].lower()
    finally:
        await bridge.shutdown()


async def test_retry_does_not_publish_discarded_text(tmp_path, monkeypatch):
    _, _, bridge, sender = setup_bridge(
        tmp_path,
        monkeypatch,
        llm_factory=lambda _: FakeLLMClient(
            [StreamFailure(text="discard this"), ScriptedTurn(text="final")]
        ),
    )
    try:
        await bridge.handle_message(message())
        await finish(bridge)
        assert [text for _, text in sender.sent] == ["final"]
    finally:
        await bridge.shutdown()


async def test_attachment_ingest_unsupported_and_oversize_rejection(
    tmp_path, monkeypatch
):
    _, _, bridge, sender = setup_bridge(tmp_path, monkeypatch)
    sender.downloads["file"] = b"some notes"
    resource = LarkResource("file", "file", "notes.txt")
    item = message(message_type="file", text="", resources=(resource,))
    try:
        await bridge.handle_message(item)
        await finish(bridge)
        runtime = await bridge.runtime_for(item)
        assert (
            runtime.user_dir / "workspace" / "attachments" / "notes.txt"
        ).read_bytes() == b"some notes"
        for kind in ("audio", "video", "sticker", "interactive"):
            await bridge.handle_message(message("om_" + kind, message_type=kind))
            assert "Send text" in sender.sent[-1][1]
        await bridge.handle_message(
            replace(item, message_id="om_many", resources=(resource, resource))
        )
        assert "Send text" in sender.sent[-1][1]
        monkeypatch.setattr("lingcore.integrations.lark.bridge.FILE_MAX_BYTES", 3)
        await bridge.handle_message(replace(item, message_id="om_big"))
        await finish(bridge)
        assert "3-byte limit" in sender.sent[-1][1]
        assert len(runtime.agent.memory.messages) == 2
    finally:
        await bridge.shutdown()


async def test_profile_commands_and_help(tmp_path, monkeypatch):
    _, _, bridge, sender = setup_bridge(tmp_path, monkeypatch)
    from lingcore.plugins.commands import Command

    bridge.commands.add(
        Command("greet", "Greet $ARGUMENTS", description="Greet someone")
    )
    runtime = await bridge.runtime_for(message())
    runtime.agent.commands = bridge.commands
    try:
        await bridge.handle_message(message("om_help", "/help"))
        assert "/greet" in sender.sent[-1][1]
        await bridge.handle_message(message("om_cmd", "/greet Jane"))
        await finish(bridge)
        row = runtime.agent.memory.messages[0]
        assert row.input_text == "/greet Jane" and row.content == "Greet Jane"
        await bridge.handle_message(message("om_unknown", "/unknown"))
        assert "Unknown command" in sender.sent[-1][1]
    finally:
        await bridge.shutdown()
