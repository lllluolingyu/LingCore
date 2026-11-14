from __future__ import annotations

from types import SimpleNamespace as NS

import httpx
import pytest

from lingcore.superagent.config import SourceConfig
from lingcore.superagent.sources.github import GitHubSource
from lingcore.superagent.sources.lark import LarkSource
from lingcore.superagent.sources.schedule import ScheduleSource
from lingcore.superagent.sources.telegram import TelegramSource
from tests.test_superagent_core import engine


class Runtime:
    def __init__(self, name):
        self.engine = engine()
        self.engine.config.sources[name] = SourceConfig(allowlist=["person"])
        self.engine.config.sources["lark"].allowlist = ["person"]
        self.store = NS(read=lambda: self.engine.state)
        self.profile = NS(plugins=["github"])
        self.router = NS(tool_ctx=NS(getenv=lambda name: "fake-test-token"))
        self.actions = []

    async def ingest(self, trigger):
        return self.engine.ingest(trigger, 1)

    async def action(self, action):
        self.actions.append(action)

    def command(self, fn):
        return fn(self.engine)


def telegram_update(text="hello", *, bot=False, actor="person", id=1):
    return NS(
        update_id=id,
        effective_user=NS(id=actor, is_bot=bot),
        effective_message=NS(
            text=text,
            caption=None,
            message_thread_id=None,
            chat_id=2,
            message_id=id,
            chat=NS(type="private"),
        ),
    )


def lark_message(text="hello", *, bot=False, actor="person", id=1):
    return NS(
        sender_type="app" if bot else "user",
        sender_id=actor,
        message_id=str(id),
        thread_id=None,
        chat_id="chat",
        chat_type="p2p",
        mentioned_bot=False,
        body_text=text,
    )


@pytest.mark.parametrize(
    "name,source,raw",
    [("lark", LarkSource, lark_message), ("telegram", TelegramSource, telegram_update)],
)
async def test_message_source_contract_dedup_allowlist_bots_and_bounded_text(
    name, source, raw
):
    runtime = Runtime(name)
    adapter = source(runtime, NS())
    await adapter.handle_message(raw("x" * 21000))
    await adapter.handle_message(raw("x" * 21000))
    await adapter.handle_message(raw(bot=True, id=2))
    await adapter.handle_message(raw(actor="stranger", id=3))
    assert len(runtime.engine.state.evaluations) == 1
    assert len(runtime.engine.state.triggers[f"{name}:1"].text) == 20000
    assert runtime.engine.state.source_positions[name] == "3"


async def test_lark_verified_action_uses_sdk_operator_not_card_actor():
    runtime = Runtime("lark")
    adapter = LarkSource(runtime, NS())
    await adapter.handle_action(
        NS(
            action=NS(
                value={
                    "kind": "lingcore_approval",
                    "allow": True,
                    "token": "nonce",
                    "actor": "spoof",
                }
            ),
            operator=NS(open_id="real"),
            chat_id="chat",
        )
    )
    assert runtime.actions[0].actor == "real"
    assert runtime.actions[0].conversation == "chat"


async def test_schedule_contract_checkpoint_survives_recreation():
    runtime = Runtime("schedule")
    config = SourceConfig(
        interval_seconds=60, options={"requester": "person", "prompt": "scheduled"}
    )
    await ScheduleSource(runtime, config).poll(120)
    await ScheduleSource(runtime, config).poll(120)
    await ScheduleSource(runtime, config).poll(180)
    assert list(runtime.engine.state.triggers) == ["schedule:2", "schedule:3"]
    assert runtime.engine.state.source_positions["schedule"] == "3"


async def test_github_contract_checkpoint_and_bot_filter():
    runtime = Runtime("github")
    config = SourceConfig(options={"repository": "owner/repo", "bot_login": "helper"})
    comments = [
        {
            "id": i,
            "user": {"login": actor, "type": kind},
            "body": "@helper review",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/1",
        }
        for i, actor, kind in [(1, "person", "User"), (2, "helper", "Bot")]
    ]
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=list(reversed(comments)))
        ),
        base_url="https://api.github.com",
    )
    source = GitHubSource(runtime, config, client=client)
    await source.poll()
    assert not runtime.engine.state.evaluations
    comments.append({**comments[0], "id": 3})
    await source.poll()
    assert len(runtime.engine.state.evaluations) == 1
    assert runtime.engine.state.source_positions["github"] == "3"
    await source.close()


async def test_github_source_requires_explicit_plugin_consent():
    from lingcore.errors import ConfigError

    runtime = Runtime("github")
    runtime.profile.plugins = []
    with pytest.raises(ConfigError, match="plugins"):
        GitHubSource(runtime, SourceConfig())


async def test_github_empty_initial_cursor_accepts_next_new_comment():
    runtime = Runtime("github")
    comments = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=comments)),
        base_url="https://api.github.com",
    )
    source = GitHubSource(
        runtime,
        SourceConfig(options={"repository": "owner/repo", "bot_login": "helper"}),
        client=client,
    )
    try:
        await source.poll()
        assert runtime.engine.state.source_positions["github"] == "0"
        comments.append(
            {
                "id": 1,
                "body": "@helper new request",
                "user": {"login": "person", "type": "User"},
                "issue_url": "https://api.github.com/repos/owner/repo/issues/1",
            }
        )
        await source.poll()
        assert len(runtime.engine.state.evaluations) == 1
    finally:
        await source.close()


async def test_github_reply_retry_checks_later_pages_before_posting():
    runtime = Runtime("github")
    config = SourceConfig(options={"repository": "owner/repo"})
    pages = []

    def respond(request):
        assert request.method == "GET", "retry must not post another comment"
        page = int(request.url.params["page"])
        pages.append(page)
        comments = (
            [{"body": "earlier comment"}] * 100
            if page == 1
            else [{"body": "delivered\n<!-- lingcore:effect-1 -->"}]
        )
        return httpx.Response(200, json=comments)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://api.github.com"
    )
    source = GitHubSource(runtime, config, client=client)
    try:
        await source.send(
            NS(address={"repository": "owner/repo", "issue": "1"}),
            "delivered",
            key="effect-1",
        )
        assert pages == [1, 2]
    finally:
        await source.close()
