"""GitHub issue-comment source, enabled only through the consented GitHub plugin."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

from lingcore.errors import ConfigError, ToolError
from lingcore.superagent.sources.base import bounded_trigger


class GitHubSource:
    name = "github"

    def __init__(self, runtime: Any, config: Any, *, client: Any = None) -> None:
        if "github" not in runtime.profile.plugins:
            raise ConfigError("GitHub source requires plugins: [github]")
        self.runtime, self.config = runtime, config
        self.repo = config.options.get("repository", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo):
            raise ConfigError("GitHub repository must be owner/name")
        token = runtime.router.tool_ctx.getenv(
            config.options.get("token_env", "GITHUB_TOKEN")
        )
        if not token:
            raise ConfigError("GitHub token environment variable is missing")
        self.client = client or httpx.AsyncClient(
            base_url="https://api.github.com",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
            timeout=30,
            follow_redirects=False,
        )
        self.task: asyncio.Task[None] | None = None

    async def poll(self) -> None:
        saved_position = self.runtime.store.read().source_positions.get(self.name)
        position = int(saved_position or "0")
        comments: list[dict[str, Any]] = []
        for page in range(1, 101):
            response = await self.client.get(
                f"/repos/{self.repo}/issues/comments",
                params={
                    "sort": "created",
                    "direction": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            response.raise_for_status()
            batch = response.json()
            if saved_position is None:
                # Installation establishes a cursor; historical comments are
                # not new requests merely because a bot was just connected.
                newest = str(max((int(v["id"]) for v in batch), default=0))
                self.runtime.command(lambda e: e.checkpoint(self.name, newest))
                return
            comments.extend(v for v in batch if int(v["id"]) > position)
            if len(batch) < 100 or any(int(v["id"]) <= position for v in batch):
                break
        else:
            raise ToolError("GitHub backlog exceeds 100 pages; checkpoint retained")
        comments.sort(key=lambda v: v["id"])
        for value in comments:
            if int(value["id"]) <= position:
                continue
            actor = value.get("user", {})
            issue = str(value["issue_url"]).rsplit("/", 1)[-1]
            if not issue.isdigit():
                continue
            text = value.get("body", "")
            login = self.config.options.get("bot_login", "")
            await self.runtime.ingest(
                bounded_trigger(
                    source=self.name,
                    external_id=str(value["id"]),
                    actor=actor.get("login", ""),
                    thread=f"github:{self.repo}:{issue}",
                    address={"repository": self.repo, "issue": issue},
                    text=text,
                    addressed=bool(login) and f"@{login}" in text,
                    bot=actor.get("type") == "Bot"
                    or actor.get("login") == self.config.options.get("bot_login"),
                    timestamp=time.time(),
                )
            )

    async def start(self) -> None:
        async def run() -> None:
            while True:
                try:
                    await self.poll()
                except (httpx.HTTPError, ValueError, ToolError) as exc:
                    import logging

                    logging.getLogger(__name__).warning(
                        "GitHub polling failed: %s", type(exc).__name__
                    )
                await asyncio.sleep(self.config.interval_seconds)

        self.task = asyncio.create_task(run())

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.client.aclose()

    async def send(
        self,
        trigger: Any,
        text: str,
        *,
        key: str,
        approval: dict[str, Any] | None = None,
    ) -> None:
        if approval:
            text += "\nReview this request in the authenticated LingChat needs-person panel. Chat replies cannot grant tools."
        if trigger.address.get("repository") != self.repo:
            raise ToolError("GitHub reply repository mismatch")
        issue = trigger.address["issue"]
        if not issue.isdigit():
            raise ToolError("invalid issue address")
        marker = f"<!-- lingcore:{key} -->"
        route = f"/repos/{self.repo}/issues/{issue}/comments"
        for page in range(1, 101):
            prior = await self.client.get(route, params={"per_page": 100, "page": page})
            prior.raise_for_status()
            comments = prior.json()
            if any(marker in c.get("body", "") for c in comments):
                return
            if len(comments) < 100:
                break
        else:
            raise ToolError("GitHub reply history exceeds 100 pages; delivery deferred")
        result = await self.client.post(route, json={"body": text + "\n" + marker})
        result.raise_for_status()
