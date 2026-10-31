"""Telegram bridge tests: core behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lingcore.config import AgentProfile
from lingcore.integrations.telegram.bridge import TelegramBridge
from lingcore.integrations.telegram.config import load_telegram_config
from lingcore.integrations.telegram.protocol import TelegramFile, TelegramMessage
from tests.fakes import FakeLLMClient, ScriptedTurn
from tests.support.telegram_bridge import PROFILE, _finish, _setup
from tests.support.telegram_bridge import FakeSender as FakeSender


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("guarded", ["Safe partial", ""])
async def test_incomplete_fallback_delivers_guarded_text(
    tmp_path, monkeypatch, stop, guarded
):
    from lingcore.guardrails import NoopGuardrail
    from lingcore.plugins.hooks import HookRunner

    class Redactor(NoopGuardrail):
        async def post_output(self, text):
            assert text == "SECRET"
            return guarded

    class BrokenEditor(FakeSender):
        async def edit_message(self, *args, **kwargs):
            raise RuntimeError("delivery failed")

    entered = asyncio.Event()

    class SlowHooks(HookRunner):
        async def turn_end(self, event):
            entered.set()
            await asyncio.Event().wait()

    _, _, bridge = _setup(
        tmp_path,
        monkeypatch,
        llm_factory=lambda _: FakeLLMClient(
            [ScriptedTurn(text="SECRET", finish_reason="length")]
        ),
    )
    sender = BrokenEditor()
    runtime = await bridge.runtime_for(11, 11, sender)
    runtime.agent.guardrail = Redactor()
    if stop:
        runtime.agent._hooks = SlowHooks([], runtime.agent.tool_ctx)
    try:
        await bridge.handle_message(
            TelegramMessage(1, 11, 11, "private", text="go"), sender
        )
        if stop:
            await asyncio.wait_for(entered.wait(), 2)
            await bridge.handle_message(
                TelegramMessage(2, 11, 11, "private", text="/stop"), sender
            )
        await _finish(bridge)
        visible = "\n".join(sender.messages.values())
        assert guarded in visible and "Model response incomplete" in visible
        assert "SECRET" not in visible
        saved = runtime.store.messages(runtime.agent.tool_ctx.session_id)[-1]
        assert saved.content == guarded and saved.finish_reason == "length"
    finally:
        await bridge.shutdown()


async def test_incomplete_stop_delivery_without_renderer(tmp_path, monkeypatch):
    from lingcore.events import Error

    _, _, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    runtime = await bridge.runtime_for(11, 11, sender)
    try:
        await bridge._deliver_cancelled_events(
            runtime,
            Error(
                "Incomplete reply",
                finish_reason="length",
                partial_content="Saved partial",
            ),
            [],
        )
        assert any(
            "Saved partial\n\n⚠️ Incomplete reply" == text
            for text in sender.messages.values()
        )
    finally:
        await bridge.shutdown()


async def test_two_users_are_isolated_in_state_profile_and_attachments(
    tmp_path, monkeypatch
):
    profile, config, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    sender.downloads["doc"] = b"private user eleven"

    await bridge.handle_message(
        TelegramMessage(
            update_id=1,
            user_id=11,
            chat_id=11,
            chat_type="private",
            text="one",
            document=TelegramFile("doc", "notes.txt", len(sender.downloads["doc"])),
        ),
        sender,
    )
    await bridge.handle_message(
        TelegramMessage(2, 22, 22, "private", text="two"), sender
    )
    await _finish(bridge)

    one = await bridge.runtime_for(11, 11, sender)
    two = await bridge.runtime_for(22, 22, sender)
    assert one.profile is not two.profile
    assert one.profile.tool_options is not two.profile.tool_options
    assert one.profile.tool_options["memory"] is not two.profile.tool_options["memory"]
    assert one.profile.tool_options["memory"]["path"].endswith(
        ".lingcore/telegram/users/11/memory.md"
    )
    assert "allow_absolute_path" not in one.profile.tool_options["memory"]
    assert one.profile.sessions.path.endswith(".lingcore/telegram/users/11/sessions.db")
    assert one.profile.sessions.allow_absolute_path is False
    assert one.agent.tool_ctx.workspace == config.state_path / "users/11/workspace"
    assert two.agent.tool_ctx.workspace == config.state_path / "users/22/workspace"
    assert (one.agent.tool_ctx.workspace / "attachments/notes.txt").is_file()
    assert not (two.agent.tool_ctx.workspace / "attachments/notes.txt").exists()
    assert one.store.db_path != two.store.db_path
    assert profile.tool_options["memory"] == {"max_bytes": 1234}
    await bridge.shutdown()


async def test_each_user_keeps_their_own_calendar(tmp_path, monkeypatch):
    profile_text = (
        PROFILE.replace(
            "tools: [memory]",
            "tools: [memory, calendar_add, calendar_update, calendar_list]",
        )
        + "  calendar:\n    timezone: Europe/Berlin\n"
    )
    profile, config, bridge = _setup(tmp_path, monkeypatch, profile_text=profile_text)
    sender = FakeSender()
    one = await bridge.runtime_for(11, 11, sender)
    two = await bridge.runtime_for(22, 22, sender)
    for user, runtime in ((11, one), (22, two)):
        options = runtime.profile.tool_options["calendar"]
        assert options["path"].endswith(f".lingcore/telegram/users/{user}/calendar.ics")
        assert options["timezone"] == "Europe/Berlin"
        assert "allow_absolute_path" not in options
    assert profile.tool_options["calendar"] == {"timezone": "Europe/Berlin"}
    add = one.agent.tools.get("calendar_add")
    args = add.validate_args({"events": [{"title": "Private", "start": "2026-10-21"}]})
    await add.run(args, one.agent.tool_ctx)
    saved = (config.state_path / "users/11/calendar.ics").read_text()
    uid = next(
        line.removeprefix("UID:")
        for line in saved.splitlines()
        if line.startswith("UID:")
    )
    update = one.agent.tools.get("calendar_update")
    await update.run(
        update.validate_args({"id": uid, "changes": {"title": "Updated private"}}),
        one.agent.tool_ctx,
    )
    assert (
        "SUMMARY:Updated private"
        in (config.state_path / "users/11/calendar.ics").read_text()
    )
    assert (config.state_path / "users/11/calendar.ics").is_file()
    assert not (config.state_path / "users/22/calendar.ics").exists()
    assert not (profile._source_dir / "calendar.ics").exists()
    await bridge.shutdown()


async def test_update_only_profiles_still_use_each_users_calendar(
    tmp_path, monkeypatch
):
    profile_text = PROFILE.replace("tools: [memory]", "tools: [calendar_update]")
    profile, config, bridge = _setup(tmp_path, monkeypatch, profile_text=profile_text)
    sender = FakeSender()
    one = await bridge.runtime_for(11, 11, sender)
    await bridge.runtime_for(22, 22, sender)
    original = (
        "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:shared-id\n"
        "SUMMARY:Private\nDTSTART;VALUE=DATE:20261021\nEND:VEVENT\nEND:VCALENDAR\n"
    )
    first = config.state_path / "users/11/calendar.ics"
    second = config.state_path / "users/22/calendar.ics"
    first.write_text(original)
    second.write_text(original)
    second_before = second.read_bytes()
    tool = one.agent.tools.get("calendar_update")
    await tool.run(
        tool.validate_args(
            {"id": "shared-id", "changes": {"title": "Updated for one"}}
        ),
        one.agent.tool_ctx,
    )
    assert "SUMMARY:Updated for one" in first.read_text()
    assert second.read_bytes() == second_before
    assert not (profile._source_dir / "calendar.ics").exists()
    await bridge.shutdown()


async def test_absolute_state_consent_cascades_only_to_derived_paths(
    tmp_path, monkeypatch
):
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(PROFILE, encoding="utf-8")
    state = tmp_path / "outside-state"
    (root / "telegram.yaml").write_text(
        f"""
token_env: TELEGRAM_TEST_TOKEN
allowed_user_ids: [11]
state_dir: {state}
allow_absolute_state_dir: true
""",
        encoding="utf-8",
    )
    profile = AgentProfile.load(root)
    config = load_telegram_config(profile, require_secrets=False)

    class Encoding:
        def encode(self, text, *, disallowed_special=()):
            return list(text)

    monkeypatch.setattr("lingcore.memory._encoding", lambda _: Encoding())
    bridge = TelegramBridge(
        profile,
        config,
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="ok")]),
    )
    sender = FakeSender()
    runtime = await bridge.runtime_for(11, 11, sender)
    assert Path(runtime.profile.tool_options["memory"]["path"]).is_absolute()
    assert runtime.profile.tool_options["memory"]["allow_absolute_path"] is True
    assert Path(runtime.profile.sessions.path).is_absolute()
    assert runtime.profile.sessions.allow_absolute_path is True
    # Unrelated profile permissions remain unchanged.
    assert runtime.profile.memory == profile.memory
    assert runtime.profile.loop == profile.loop
    await bridge.shutdown()


async def test_resume_selection_survives_silent_restart(tmp_path, monkeypatch):
    profile, config, bridge = _setup(tmp_path, monkeypatch)
    sender = FakeSender()
    await bridge.handle_message(
        TelegramMessage(1, 11, 11, "private", text="first"), sender
    )
    await _finish(bridge)
    runtime = await bridge.runtime_for(11, 11, sender)
    first_id = runtime.agent.memory.session_id

    await bridge.handle_message(
        TelegramMessage(2, 11, 11, "private", text="/new"), sender
    )
    second_id = runtime.agent.memory.session_id
    assert second_id != first_id
    await bridge.handle_message(
        TelegramMessage(3, 11, 11, "private", text=f"/resume {first_id[:8]}"),
        sender,
    )
    assert runtime.agent.memory.session_id == first_id
    await bridge.shutdown()

    restarted = TelegramBridge(
        profile,
        config,
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="after")]),
    )
    restored = await restarted.runtime_for(11, 11, sender)
    assert restored.agent.memory.session_id == first_id
    await restarted.shutdown()
