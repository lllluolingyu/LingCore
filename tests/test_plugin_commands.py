from pathlib import Path

import pytest

from lingcore.errors import ConfigError
from lingcore.plugins.commands import CommandCatalog, load_commands


def command(root: Path, name: str, body: str = "Do $ARGUMENTS") -> None:
    root.mkdir(exist_ok=True)
    (root / f"{name}.md").write_text(body)


def test_catalog_resolves_and_preserves_exact_display_text(tmp_path):
    command(
        tmp_path,
        "review",
        "---\ndescription: Review code\nargument_hint: <path>\n---\nReview $ARGUMENTS",
    )
    catalog = load_commands(tmp_path, namespace="code-review")
    raw = "/code-review:review  a $ARGUMENTS\nnext "
    incoming = catalog.resolve(raw, reserved={"help"})
    assert incoming.text == "Review a $ARGUMENTS\nnext "
    assert incoming.display_text == raw
    assert catalog.resolve("/review src", reserved=set()).text == "Review src"
    assert (
        catalog.resolve("/code_review_review@MyBot src", reserved=set()).text
        == "Review src"
    )
    assert catalog.commands[0].description == "Review code"
    assert catalog.commands[0].argument_hint == "<path>"


def test_reserved_bare_and_ambiguous_aliases_fail_safe(tmp_path):
    command(tmp_path, "help")
    catalog = load_commands(tmp_path, namespace="one")
    assert catalog.resolve("/help", reserved={"/help"}) is None
    assert catalog.resolve("/one:help", reserved={"help"}) is not None
    other = tmp_path / "other"
    command(other, "help")
    catalog.extend(load_commands(other, namespace="two"))
    assert catalog.resolve("/help", reserved=set()) is None
    assert catalog.resolve("/one:help", reserved=set()) is not None
    # Telegram normalizes hyphens and underscores; collisions must not pick one.
    alias = CommandCatalog()
    alias.extend(load_commands(tmp_path, namespace="a-b"))
    command(other, "b_help")
    alias.extend(load_commands(other, namespace="a"))
    assert alias.resolve("/a_b_help", reserved=set()) is None


def test_profile_commands_and_duplicate_merge(tmp_path):
    command(tmp_path, "plan")
    catalog = load_commands(tmp_path)
    assert catalog.resolve("/plan hello", reserved=set()).text == "Do hello"
    with pytest.raises(ConfigError, match="duplicate"):
        catalog.extend(load_commands(tmp_path))
    assert len(catalog.commands) == 1
    assert load_commands(tmp_path / "missing").commands == ()


@pytest.mark.parametrize(
    "body", ["---\nunknown: value\n---\nhi", "---\ndescription: [bad]\n---\nhi"]
)
def test_bad_frontmatter_rejected(tmp_path, body):
    command(tmp_path, "plan", body)
    with pytest.raises(ConfigError):
        load_commands(tmp_path)


async def test_cli_expands_commands_after_attachment_parsing(tmp_path, monkeypatch):
    from lingcore.io.cli import CLIFrontend

    command(tmp_path, "plan", "Plan $ARGUMENTS")
    attachment = tmp_path / "note.txt"
    attachment.write_text("notes")
    frontend = CLIFrontend()
    frontend.set_commands(load_commands(tmp_path))
    raw = f"/plan work @{attachment}"

    async def read():
        return raw

    monkeypatch.setattr(frontend, "_read_message", read)
    incoming = await frontend.read_input()
    assert incoming.text == f"Plan work {attachment}"
    assert incoming.display_text == raw
    assert incoming.attachments[0].name == "note.txt"
    assert "/plan" in frontend._commands.metadata()[0]["name"]


def test_digit_prefixed_plugin_namespace_and_telegram_alias(tmp_path):
    command(tmp_path, "review")
    catalog = load_commands(tmp_path, namespace="1-review")
    assert catalog.resolve("/1-review:review code", reserved=()).text == "Do code"
    assert catalog.resolve("/1_review_review code", reserved=()).text == "Do code"
    assert catalog.telegram_commands()[0].telegram_name == "1_review_review"


def test_cli_plugin_notice_is_visible():
    import io

    from rich.console import Console

    from lingcore.events import PluginNotice
    from lingcore.io.cli import CLIFrontend

    frontend = CLIFrontend()
    output = io.StringIO()
    frontend.console = Console(file=output, force_terminal=False, width=100)
    frontend.render(
        PluginNotice("policy", "user_message", "denied", "Ask an administrator")
    )
    assert "policy" in output.getvalue()
    assert "denied: Ask an administrator" in output.getvalue()


async def test_telegram_plugin_notice_is_visible():
    from lingcore.events import PluginNotice
    from lingcore.integrations.telegram.rendering import TelegramTurnRenderer
    from tests.test_telegram_bridge import FakeSender

    sender = FakeSender()
    renderer = TelegramTurnRenderer(sender, 11, edit_interval=0)
    await renderer.handle(
        PluginNotice("policy", "before_tool", "blocked", "Approval required")
    )
    assert (
        sender.sent[-1][1] == "Plugin policy · before_tool · blocked: Approval required"
    )


def test_telegram_menu_omits_alias_shadowed_by_another_bare_command():
    from lingcore.plugins.commands import Command

    catalog = CommandCatalog(
        [Command("help", "one", "a"), Command("a_help", "two", "b")]
    )
    assert catalog.resolve("/a_help", reserved=()) is None
    assert all(c.telegram_name != "a_help" for c in catalog.telegram_commands())


def test_illegal_long_telegram_alias_is_not_resolved():
    from lingcore.plugins.commands import Command

    catalog = CommandCatalog([Command("command", "one", "a" * 35)])
    assert catalog.resolve("/" + "a" * 35 + "_command", reserved=()) is None
    assert catalog.resolve("/" + "a" * 35 + ":command", reserved=()) is not None


async def test_cli_stop_delivers_queued_plugin_notice_before_terminal_once(tmp_path):
    import asyncio
    from contextlib import contextmanager

    from lingcore.events import Final, PluginNotice, TurnCancelled
    from lingcore.io.base import run_session
    from lingcore.plugins import PluginHooks, ToolDecision
    from tests.fakes import ScriptedTurn
    from tests.test_cli import ScriptedFrontend
    from tests.test_plugin_hooks import build, tool_turn

    waiting = asyncio.Event()

    class Ask(PluginHooks):
        async def before_tool(self, event):
            return ToolDecision.ask("Approve this call?")

    class Wait(PluginHooks):
        async def before_tool(self, event):
            waiting.set()
            await asyncio.Event().wait()

    class Frontend(ScriptedFrontend):
        stop = None

        @contextmanager
        def interrupt_scope(self, stop):
            self.stop = stop
            yield

    frontend = Frontend(["first", "second"])
    agent, _ = build(
        tmp_path,
        [Ask, Wait],
        confirm=frontend.confirm,
        turns=[tool_turn(), ScriptedTurn(text="next turn")],
    )
    task = asyncio.create_task(run_session(agent, frontend))
    await asyncio.wait_for(waiting.wait(), 1)
    assert frontend.confirmed == ["Approve this call?"]
    assert frontend.stop()
    await asyncio.wait_for(task, 1)
    notices = [event for event in frontend.events if isinstance(event, PluginNotice)]
    assert len(notices) == 1
    assert notices[0].action == "asked"
    terminal = next(
        event for event in frontend.events if isinstance(event, TurnCancelled)
    )
    assert frontend.events.index(notices[0]) < frontend.events.index(terminal)
    assert isinstance(frontend.events[-1], Final)
    assert frontend.events[-1].content == "next turn"
    assert agent.drain_plugin_notices() == []
    await agent.aclose()


def test_command_names_match_case_insensitively(tmp_path):
    command(tmp_path, "review", "Review $ARGUMENTS")
    catalog = load_commands(tmp_path)
    assert catalog.resolve("/Review src", reserved=set()).text == "Review src"
    assert catalog.resolve("/REVIEW@MyBot src", reserved=set()).text == "Review src"
    assert catalog.resolve("/Review", reserved={"/review"}) is None


def test_crlf_frontmatter_is_parsed(tmp_path):
    (tmp_path / "review.md").write_bytes(
        b"---\r\ndescription: Review code\r\n---\r\nReview $ARGUMENTS\r\n"
    )
    catalog = load_commands(tmp_path)
    assert catalog.commands[0].description == "Review code"
    assert catalog.resolve("/review a", reserved=set()).text == "Review a\n"
