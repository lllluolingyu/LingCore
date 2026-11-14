"""Serialized, labelled approvals with no inherited always-allow decision."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingcore.tools import ConfirmFn


class ConfirmationBroker:
    def __init__(
        self,
        confirm: ConfirmFn | None,
        *,
        async_confirm: bool = False,
        current: Callable[[], ConfirmFn | None] | None = None,
    ) -> None:
        self.confirm = confirm
        self.current = current
        self.async_confirm = async_confirm
        self.lock = asyncio.Lock()

    def for_worker(
        self,
        run_id: str,
        label: str,
        cancelled: asyncio.Event,
        *,
        background: bool = False,
        deny: bool = False,
    ) -> ConfirmFn:
        async def approve(prompt: str) -> bool:
            confirm = self.current() if self.current else self.confirm
            if (
                deny
                or confirm is None
                or cancelled.is_set()
                or (background and not self.async_confirm)
            ):
                return False
            async with self.lock:
                if cancelled.is_set():
                    return False
                return await confirm(f"[worker {label} · run {run_id}] {prompt}")

        return approve
