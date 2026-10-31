"""Fixtures and builders for cli tests."""

from __future__ import annotations

from pathlib import Path

from lingcore.config import AgentProfile
from lingcore.events import (
    ToolCallStarted,
)
from lingcore.io.cli import (
    CLIFrontend,
)
from lingcore.message import ToolCall

PROFILE = """
name: smoke
workspace: ${SMOKE_WS:-.}
llm:
  model: test-model
  base_url: http://localhost:11434/v1
persona:
  system_prompt: "You are a smoke-test agent."
tools:
  - read_file
  - run_shell
loop:
  max_iters: 5
"""


class ScriptedFrontend:
    """A Frontend that replays canned inputs and records rendered events."""

    def __init__(self, inputs: list[str], confirm_answer: bool = True):
        self._inputs = list(inputs)
        self.confirm_answer = confirm_answer
        self.events: list = []
        self.confirmed: list[str] = []

    async def read_input(self) -> str | None:
        if self._inputs:
            return self._inputs.pop(0)
        return None

    def render(self, event) -> None:
        self.events.append(event)

    async def confirm(self, command: str) -> bool:
        self.confirmed.append(command)
        return self.confirm_answer


def _profile(tmp_path: Path) -> AgentProfile:
    p = tmp_path / "p.yaml"
    p.write_text(PROFILE, encoding="utf-8")
    return AgentProfile.load(p)


def _announce_shell(cli: CLIFrontend, *commands: str) -> None:
    """Render the run_shell calls whose confirmations a test then answers.

    The agent always emits ``ToolCallStarted`` before dispatching a batch, and
    the CLI shows the shell prompt only for this turn's run_shell commands.
    """
    for i, command in enumerate(commands):
        call = ToolCall(id=f"s{i}", name="run_shell", arguments={"command": command})
        cli.render(ToolCallStarted(call))


def _recording_cli(width: int = 120) -> CLIFrontend:
    from rich.console import Console

    cli = CLIFrontend(agent_name="t")
    cli.console = Console(record=True, width=width)
    return cli


async def _prompt_with_keys(keys: str, pre_run=None) -> str:
    """Drive the CLI's prompt_toolkit key bindings with scripted terminal input."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from lingcore.io.cli import _input_key_bindings, _InputCompleter

    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            input=pipe,
            output=DummyOutput(),
            multiline=True,
            key_bindings=_input_key_bindings(),
            completer=_InputCompleter(),
        )
        pipe.send_text(keys)
        return await session.prompt_async(handle_sigint=False, pre_run=pre_run)
