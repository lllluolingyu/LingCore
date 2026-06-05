"""The agent runtime loop — the core of LingCore.

``Agent.run`` is the whole point of "swap config, not code": every agent type
(coding, role-play, teaching, psych) runs this same loop, differing only in
the profile that built it. The loop:

    assemble context -> stream a turn -> if tool calls: run them (in parallel)
    and feed results back -> repeat -> on a tool-free turn, post-guard + Final.

The loop talks only to an ``LLMClient``-shaped object (duck-typed ``stream``),
never to the OpenAI SDK directly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol

from lingcore.events import (
    AgentEvent,
    Error,
    Final,
    TextDelta,
    ToolCallStarted,
    ToolResultEvent,
)
from lingcore.guardrails import Guardrail, NoopGuardrail
from lingcore.llm import LLMChunk
from lingcore.memory import ShortTermMemory, WindowMemory
from lingcore.message import Message, ToolCall, ToolResult
from lingcore.tools import ToolContext, ToolRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from lingcore.config import AgentProfile
    from lingcore.tools import ConfirmFn


def _build_guardrail(policy: str) -> Guardrail:
    """Map a profile's guardrail policy name to an implementation.

    Only ``noop`` exists at MVP; real policies (e.g. a psych-consultant
    crisis-detection guardrail) register here later without touching the loop.
    """
    if policy == "noop":
        return NoopGuardrail()
    from lingcore.errors import ConfigError

    raise ConfigError(f"unknown guardrail policy: {policy!r}")



class _LLMLike(Protocol):
    def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]: ...


class Agent:
    def __init__(
        self,
        llm: _LLMLike,
        tools: ToolRegistry,
        tool_ctx: ToolContext,
        system_prompt: str,
        memory: ShortTermMemory | None = None,
        guardrail: Guardrail | None = None,
        max_iters: int = 25,
        parallel_tools: bool = True,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_ctx = tool_ctx
        self.system_prompt = system_prompt
        self.memory: ShortTermMemory = memory or WindowMemory()
        self.guardrail = guardrail or NoopGuardrail()
        self.max_iters = max_iters
        self.parallel_tools = parallel_tools

    @classmethod
    def from_profile(
        cls,
        profile: "AgentProfile",
        *,
        confirm: "ConfirmFn | None" = None,
        llm: _LLMLike | None = None,
        base_dir: "Path | None" = None,
        tool_options: "dict | None" = None,
    ) -> "Agent":
        """Assemble a runnable Agent from a validated profile.

        ``confirm`` is the frontend-supplied callback that gates risky tools
        (passed through to ToolContext). ``llm`` allows injecting a fake client
        in tests. ``base_dir`` resolves a relative workspace (the profile's
        own directory). ``tool_options`` lets a caller share a live options
        dict with a frontend (so an "allow always" action mutates the same
        dict the tools read); defaults to a copy of the profile's options.
        """
        # Import for registration side effect: populates the global REGISTRY.
        import lingcore.tools.builtin  # noqa: F401
        from lingcore.config import AgentProfile  # noqa: F401  (typing only)
        from lingcore.llm import LLMClient
        from lingcore.tools import REGISTRY, ToolContext

        client: _LLMLike = llm or LLMClient(
            model=profile.llm.model,
            api_key=profile.llm.resolve_api_key(),
            base_url=profile.llm.base_url,
            sampling=profile.llm.sampling.as_kwargs(),
        )

        tools = REGISTRY.subset(profile.tools)
        workspace = profile.workspace_path(base_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        tool_ctx = ToolContext(
            workspace=workspace,
            confirm=confirm,
            options=tool_options if tool_options is not None else dict(profile.tool_options),
        )
        memory = WindowMemory(
            max_messages=profile.memory.max_messages,
            max_tokens=profile.memory.max_tokens,
            model=profile.llm.model,
        )
        guardrail = _build_guardrail(profile.guardrail.policy)

        return cls(
            llm=client,
            tools=tools,
            tool_ctx=tool_ctx,
            system_prompt=profile.persona.system_prompt,
            memory=memory,
            guardrail=guardrail,
            max_iters=profile.loop.max_iters,
            parallel_tools=profile.loop.parallel_tools,
        )

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Drive one user turn to completion, yielding events as they happen.

        Yields ``Error`` (not raises) on max-iterations so a frontend session
        loop keeps running; callers wanting hard failure can inspect the event.
        """
        text = await self.guardrail.pre_input(user_input)
        self.memory.add(Message.user(text))

        for _ in range(self.max_iters):
            # Stream one assistant turn, surfacing text deltas live.
            messages = self.memory.render(self.system_prompt)
            schemas = self.tools.schemas()
            content_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            async for chunk in self.llm.stream(messages, tools=schemas):
                if chunk.text_delta:
                    content_parts.append(chunk.text_delta)
                    yield TextDelta(chunk.text_delta)
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

            assistant = Message.assistant(
                content="".join(content_parts), tool_calls=tool_calls
            )
            self.memory.add(assistant)

            if not assistant.tool_calls:
                final = await self.guardrail.post_output(assistant.content)
                yield Final(final)
                return

            # Announce every call before running them (deterministic order).
            for call in assistant.tool_calls:
                yield ToolCallStarted(call)

            results = await self._dispatch(assistant.tool_calls)
            for result in results:
                self.memory.add(Message.from_tool_result(result))
                yield ToolResultEvent(result)
            # loop continues; model now sees the tool results

        yield Error(f"reached max iterations ({self.max_iters}) without a final reply")

    async def _dispatch(self, calls: list[ToolCall]) -> list[ToolResult]:
        async def run_one(call: ToolCall) -> ToolResult:
            try:
                tool = self.tools.get(call.name)
                args = tool.validate_args(call.arguments)
                out = await tool.run(args, self.tool_ctx)
                return ToolResult(call_id=call.id, name=call.name, content=out, ok=True)
            except Exception as e:
                # Contain every failure: feed it back so the model can recover
                # instead of crashing the run. ToolError messages are clean;
                # unexpected errors are repr'd.
                from lingcore.errors import ToolError

                msg = str(e) if isinstance(e, ToolError) else f"internal error: {e!r}"
                return ToolResult(
                    call_id=call.id, name=call.name, content=f"ERROR: {msg}", ok=False
                )

        if self.parallel_tools and len(calls) > 1:
            return list(await asyncio.gather(*(run_one(c) for c in calls)))
        return [await run_one(c) for c in calls]
