"""Thin PTB adapter construction and synchronous runner arguments."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from telegram import Bot
from telegram.ext import AIORateLimiter

from lingcore.config import AgentProfile
from lingcore.errors import ToolError
from lingcore.integrations.telegram.application import (
    _ALLOWED_UPDATES,
    _BRIDGE_KEY,
    PTBSender,
    create_telegram_application,
    run_telegram,
)
from lingcore.integrations.telegram.config import load_telegram_config
from tests.fakes import FakeLLMClient, ScriptedTurn

PROFILE = """
name: app-test
llm:
  model: test-model
  base_url: http://localhost:11434/v1
tools: []
"""


def _loaded(tmp_path: Path):
    root = tmp_path / "profile"
    root.mkdir()
    (root / "config.yaml").write_text(PROFILE, encoding="utf-8")
    (root / "telegram.yaml").write_text(
        "token_env: BOT_TOKEN\nallowed_user_ids: [1]\n", encoding="utf-8"
    )
    profile = AgentProfile.load(root)
    config = load_telegram_config(profile, require_secrets=False)
    return profile, config


async def test_factory_registers_handlers_and_bridge_with_bot_seam(
    tmp_path, monkeypatch
):
    profile, config = _loaded(tmp_path)
    application = create_telegram_application(
        profile,
        config,
        bot=Bot("123456:TESTTOKEN"),
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="ok")]),
    )
    assert _BRIDGE_KEY in application.bot_data
    assert len(application.handlers[0]) == 2
    await application.bot_data[_BRIDGE_KEY].shutdown()


async def test_factory_enables_serial_updates_and_retrying_rate_limiter(tmp_path):
    profile, config = _loaded(tmp_path)
    config._profile_env["BOT_TOKEN"] = "123456:TESTTOKEN"

    application = create_telegram_application(
        profile,
        config,
        llm_factory=lambda _: FakeLLMClient([ScriptedTurn(text="ok")]),
    )

    assert application.update_processor.max_concurrent_updates == 1
    assert isinstance(application.bot.rate_limiter, AIORateLimiter)
    assert application.bot.rate_limiter._max_retries == 1
    await application.bot_data[_BRIDGE_KEY].shutdown()


def test_polling_runner_retains_updates_and_limits_update_types(
    tmp_path, monkeypatch
):
    profile, config = _loaded(tmp_path)
    calls = {}

    class App:
        def run_polling(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.AgentProfile.load",
        lambda _: profile,
    )
    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.load_telegram_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.create_telegram_application",
        lambda *args, **kwargs: App(),
    )
    assert run_telegram(tmp_path) == 0
    assert calls == {
        "allowed_updates": _ALLOWED_UPDATES,
        "drop_pending_updates": False,
    }


def test_webhook_runner_uses_derived_path_secret_and_reverse_proxy_tls(
    tmp_path, monkeypatch
):
    profile, config = _loaded(tmp_path)
    config.mode = "webhook"
    config.webhook.public_url = "https://example.test/hooks/lingcore"
    config.webhook.listen = "127.0.0.2"
    config.webhook.port = 9000
    config.webhook.secret_token_env = "HOOK_SECRET"
    config._profile_env["HOOK_SECRET"] = "secret"
    calls = {}

    class App:
        def run_webhook(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.AgentProfile.load",
        lambda _: profile,
    )
    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.load_telegram_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.create_telegram_application",
        lambda *args, **kwargs: App(),
    )
    assert run_telegram(tmp_path, mode="webhook") == 0
    assert calls == {
        "listen": "127.0.0.2",
        "port": 9000,
        "url_path": "hooks/lingcore",
        "webhook_url": "https://example.test/hooks/lingcore",
        "secret_token": "secret",
        "allowed_updates": _ALLOWED_UPDATES,
        "drop_pending_updates": False,
    }


async def test_download_stream_enforces_cap_when_metadata_is_understated(
    monkeypatch,
):
    class Stream(httpx.AsyncByteStream):
        def __init__(self):
            self.yielded = 0

        async def __aiter__(self):
            for chunk in (b"1234", b"5678", b"must-not-be-read"):
                self.yielded += 1
                yield chunk

    stream = Stream()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream)
        )
    )
    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.httpx.AsyncClient",
        lambda **_: client,
    )

    class Remote:
        file_size = 1
        file_path = "https://example.test/file"

        async def download_to_memory(self, _):
            pytest.fail("PTB's buffering download helper must not be used")

    class FakeBot:
        async def get_file(self, _):
            return Remote()

    with pytest.raises(ToolError, match="download exceeded limit 6"):
        await PTBSender(FakeBot()).download_file("file", max_bytes=6)
    assert stream.yielded == 2


async def test_download_stream_accepts_unknown_size_within_cap(monkeypatch):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b"bounded")
        )
    )
    monkeypatch.setattr(
        "lingcore.integrations.telegram.application.httpx.AsyncClient",
        lambda **_: client,
    )

    class Remote:
        file_size = None
        file_path = "https://example.test/file"

    class FakeBot:
        async def get_file(self, _):
            return Remote()

    assert (
        await PTBSender(FakeBot()).download_file("file", max_bytes=7)
        == b"bounded"
    )


async def test_local_bot_api_download_is_bounded_when_metadata_is_understated(
    tmp_path,
):
    local_file = tmp_path / "telegram-file"
    local_file.write_bytes(b"oversized local payload")

    class Remote:
        file_size = 1
        file_path = local_file

        async def download_to_memory(self, _):
            pytest.fail("PTB's buffering download helper must not be used")

    class FakeBot:
        async def get_file(self, _):
            return Remote()

    with pytest.raises(ToolError, match="download exceeded limit 4"):
        await PTBSender(FakeBot()).download_file("file", max_bytes=4)
