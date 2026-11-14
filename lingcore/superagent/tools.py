"""Host-injected router/executor tools. No global registration or approval tool."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lingcore._agent.dispatch import dispatch_authorization
from lingcore.errors import ToolError
from lingcore.tools import Tool, ToolContext


class RouteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eval_id: str
    decisions: list[dict[str, Any]] = Field(min_length=1, max_length=20)


class TicketArgs(BaseModel):
    ticket_id: str


class NoteArgs(TicketArgs):
    text: str = Field(max_length=2000)


class ReplyArgs(NoteArgs):
    purpose: Literal["ack", "answer", "result", "attention"]


class ReportArgs(BaseModel):
    status: Literal["progress", "waiting", "ready", "blocked", "failed", "dropped"]
    summary: str = Field(max_length=2000)
    question: str | None = Field(default=None, max_length=2000)
    final: bool = True


class FollowArgs(BaseModel):
    thread: str


class EmptyArgs(BaseModel):
    pass


def router_tools(runtime: Any) -> tuple[Tool, ...]:
    async def route(args: RouteArgs, ctx: ToolContext) -> str:
        snapshot = dispatch_authorization.get()
        if snapshot is None:
            raise ToolError("routing requires host authorization")
        if args.eval_id != runtime.current_eval:
            raise ToolError("evaluation is not assigned to this router turn")
        return json.dumps(
            runtime.command(
                lambda e: e.route(args.eval_id, args.decisions, snapshot, time.time())
            )
        )

    async def read(args: TicketArgs, ctx: ToolContext) -> str:
        ticket = runtime.store.read().tickets.get(args.ticket_id)
        if ticket is None:
            raise ToolError("unknown ticket")
        return str(ticket.model_dump_json())[:16000]

    async def reply(args: ReplyArgs, ctx: ToolContext) -> str:
        runtime.command(
            lambda e: e.reply(args.ticket_id, args.purpose, args.text, time.time())
        )
        return "Reply queued."

    async def note(args: NoteArgs, ctx: ToolContext) -> str:
        runtime.command(lambda e: e.note(args.ticket_id, args.text))
        return "Note saved."

    async def retry(args: TicketArgs, ctx: ToolContext) -> str:
        runtime.command(lambda e: e.retry(args.ticket_id, time.time()))
        return "Assignment queued."

    async def status(args: EmptyArgs, ctx: ToolContext) -> str:
        return json.dumps(runtime.status())

    return (
        Tool(
            "ticket_route",
            "Accept, attach, stop or ignore candidate events.",
            RouteArgs,
            route,
        ),
        Tool("ticket_read", "Read one ticket's mandate and state.", TicketArgs, read),
        Tool(
            "ticket_reply",
            "Queue an ack, answer, result or attention message.",
            ReplyArgs,
            reply,
        ),
        Tool("ticket_note", "Record a short internal note.", NoteArgs, note),
        Tool(
            "ticket_retry",
            "Retry eligible ended work under its existing mandate.",
            TicketArgs,
            retry,
        ),
        Tool(
            "ticket_status",
            "Read current ticket and service status.",
            EmptyArgs,
            status,
        ),
    )


def executor_tools(
    runtime: Any, ticket_id: str, assignment_id: str
) -> tuple[Tool, ...]:
    async def report(args: ReportArgs, ctx: ToolContext) -> str:
        runtime.command(
            lambda e: e.report(
                ticket_id,
                assignment_id,
                args.status,
                args.summary,
                args.question,
                False,
                time.time(),
                notify=not (args.final and args.status != "progress"),
            )
        )
        if args.final and args.status != "progress":
            runtime.final_reports[assignment_id] = args.model_dump()
        return "Report recorded. Finish with submit_result."

    async def read(args: EmptyArgs, ctx: ToolContext) -> str:
        return str(runtime.store.read().tickets[ticket_id].model_dump_json())[:16000]

    async def ack(args: EmptyArgs, ctx: ToolContext) -> str:
        runtime.command(
            lambda e: e.note(ticket_id, "Executor acknowledged the inputs.")
        )
        return "Acknowledged."

    async def handback(args: ReportArgs, ctx: ToolContext) -> str:
        if not args.question:
            raise ToolError("handback requires one concrete question")
        await report(args.model_copy(update={"status": "waiting", "final": True}), ctx)
        return "Handed back. Finish with submit_result."

    async def follow(args: FollowArgs, ctx: ToolContext) -> str:
        runtime.command(lambda e: e.follow(ticket_id, args.thread))
        return "Following thread as context only."

    async def unrelated(args: EmptyArgs, ctx: ToolContext) -> str:
        runtime.command(
            lambda e: e.note(ticket_id, "Executor marked the latest input unrelated.")
        )
        return "Recorded."

    return (
        Tool(
            "ticket_report",
            "Report progress or a final state, with one concrete question when blocked.",
            ReportArgs,
            report,
        ),
        Tool("ticket_read", "Read this ticket's mandate and inputs.", EmptyArgs, read),
        Tool("ticket_ack", "Acknowledge the ticket inputs.", EmptyArgs, ack),
        Tool(
            "ticket_handback",
            "Return work needing a person with one concrete question.",
            ReportArgs,
            handback,
        ),
        Tool(
            "ticket_follow",
            "Subscribe to an existing thread for context only.",
            FollowArgs,
            follow,
        ),
        Tool(
            "ticket_unrelated", "Mark the latest input unrelated.", EmptyArgs, unrelated
        ),
    )
