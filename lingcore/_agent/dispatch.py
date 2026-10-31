"""Tool disclosure, authorization, validation, and execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from lingcore.errors import (
    ToolError,
)
from lingcore.message import ToolCall, ToolResult
from lingcore.plugins import (
    ToolCallEvent,
)
from lingcore.tools import Tool, ToolOutput

if TYPE_CHECKING:
    from lingcore.agent import Agent


def _effective_tool_schemas(self: Agent) -> list[dict[str, Any]]:
    """Tool schemas for the current step: the initially-enabled tools plus
    any tools granted by active skills (intersected with the profile
    allowlist by SkillState).

    Computed fresh each iteration; the base ToolRegistry is never mutated.
    Tools in the ceiling but neither initial nor granted by an active skill
    are NOT advertised — progressive disclosure (invariant 11).
    """
    base = [
        self.tools.get(name).json_schema()
        for name in self.tools.names()
        if name in self.initial_tools
    ]
    if self.skill_state is None:
        return base

    # Disclose available skills in activate_skill's description so the model
    # knows what it can invoke (progressive disclosure via the tool spec).
    if self.skill_state.skills:
        listing = "\n".join(
            f"  - {s.name}: {s.description}" for s in self.skill_state.skills.values()
        )
        for schema in base:
            fn = schema["function"]
            if fn["name"] == "activate_skill":
                fn["description"] = (
                    fn["description"] + "\nAvailable skills:\n" + listing
                )
                break

    if not self.skill_state.active:
        return base
    seen = {s["function"]["name"] for s in base}
    extra_names = self.skill_state.active_effective_tools() - seen
    for name in sorted(extra_names):
        if name in self.tools.names():
            base.append(self.tools.get(name).json_schema())
    return base


async def _dispatch(self: Agent, calls: list[ToolCall]) -> list[ToolResult]:
    # Authorization is a property of the model request that produced this
    # batch. Snapshot it before any tool runs: activate_skill/deactivation in
    # one call must affect only the next loop iteration, never sibling calls
    # whose schemas were computed under the previous state.
    authorized = self._authorized_tool_names()
    producing_request_id = self.request_ledger.last_request_id

    async def run_one(call: ToolCall) -> ToolResult:
        # Validation and the batch's authorization snapshot precede every
        # hook: plugins cannot make an invalid or forbidden call reachable.
        identity = {
            "tool.name": call.name,
            "tool.call_id": call.id,
            "request_id": producing_request_id,
        }
        with self.recorder.span("tool.dispatch", attributes=identity) as dispatch:
            if self.execution_budget is not None:
                self.execution_budget.reserve_tool()
            try:
                tool = self._resolve_tool(call.name, authorized=authorized)
                if call.invalid_arguments is not None:
                    raise ToolError(
                        "not run: tool arguments must be a complete JSON object; "
                        "submit the call again with valid arguments"
                    )
                args = tool.validate_args(call.arguments)
            except Exception as e:
                dispatch.close(
                    "denied",
                    decision="reject",
                    reason="authorization"
                    if call.name not in authorized
                    else "invalid_arguments",
                )
                msg = str(e) if isinstance(e, ToolError) else f"internal error: {e!r}"
                return ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=f"ERROR: {msg}",
                    ok=False,
                )
            event = ToolCallEvent(call.id, call.name, call.arguments)
            decision = await self._hooks.before_tool(event)
            if decision is not None and decision.action == "deny":
                dispatch.close("denied", decision="deny", reason="plugin")
                result = ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=f"ERROR: blocked by plugin {decision.plugin}: {decision.reason}",
                    ok=False,
                )
                return await self._hooks.after_tool(event, result)
            dispatch.attributes["decision"] = "allow"
        self.recorder.event(
            "policy.decision",
            attributes={
                "tool.name": call.name,
                "decision": "allow",
                "policy": "profile_and_plugins",
            },
        )
        with self.recorder.span("tool.execute", attributes=identity) as execution:
            if self.recorder.content == "redacted":
                from pathlib import Path

                try:
                    validated = args.model_dump(mode="json")
                    for key in tool.workspace_path_args:
                        if isinstance(validated.get(key), str):
                            resolved = (
                                self.tool_ctx.workspace / Path(validated[key])
                            ).resolve()
                            validated[key] = (
                                str(
                                    resolved.relative_to(
                                        self.tool_ctx.workspace.resolve()
                                    )
                                )
                                if resolved.is_relative_to(
                                    self.tool_ctx.workspace.resolve()
                                )
                                else "[outside-workspace]"
                            )
                    self._capture_committed(
                        "tool.arguments",
                        {
                            "span_id": execution.context.span_id,
                            "arguments": validated,
                            "workspace_path_args": tool.workspace_path_args,
                        },
                    )
                except Exception:
                    # Unserializable plugin arguments remain executable;
                    # their missing evidence cannot pass an argument grader.
                    self._capture_committed(
                        "tool.arguments",
                        {"span_id": execution.context.span_id, "unavailable": True},
                    )
            try:
                out = await tool.run(args, self.tool_ctx)
                if isinstance(out, ToolOutput):
                    result = ToolResult(
                        call_id=call.id,
                        name=call.name,
                        content=out.text,
                        attachments=out.attachments,
                        ok=True,
                    )
                else:
                    result = ToolResult(
                        call_id=call.id, name=call.name, content=out, ok=True
                    )
            except Exception as e:
                execution.attributes["error.type"] = type(e).__name__
                msg = str(e) if isinstance(e, ToolError) else f"internal error: {e!r}"
                result = ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=f"ERROR: {msg}",
                    ok=False,
                )
            execution.close(
                "ok" if result.ok else "error",
                ok=result.ok,
                output_bytes=len(result.content.encode(errors="replace")),
                attachments=len(result.attachments),
            )
            return await self._hooks.after_tool(event, result)

    if self.parallel_tools and len(calls) > 1:
        tasks = [asyncio.create_task(run_one(call)) for call in calls]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # A child's CancelledError makes gather return immediately;
            # explicitly reap every sibling before the turn can finalize.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    return [await run_one(c) for c in calls]


def _authorized_tool_names(self: Agent) -> frozenset[str]:
    """Currently dispatchable tools, bounded by the profile registry."""
    authorized = self.initial_tools
    if self.skill_state is not None:
        authorized |= self.skill_state.active_effective_tools()
    return authorized & frozenset(self.tools.names())


def _resolve_tool(
    self: Agent, name: str, *, authorized: frozenset[str] | None = None
) -> Tool:
    """Look up a tool by name, enforcing the permission model at dispatch.

    A tool is dispatchable only when it is currently *authorized*: in the
    initially-enabled set, or granted by an active skill (profile ∩
    requested). A ceiling tool that is neither — advertised or not — is
    refused with the same error an unknown tool would raise, so a model that
    calls a gated-but-not-active tool can't reach it.
    """
    allowed = authorized if authorized is not None else self._authorized_tool_names()
    if name in allowed:
        return self.tools.get(name)
    from lingcore.errors import ToolError

    raise ToolError(f"unknown tool: {name}")
