"""Synthetic agent used only to verify Harbor's cancellation contract."""

from __future__ import annotations

import asyncio

from harbor.agents.nop import NopAgent

from lingcore_harbor_bridge.agent import BridgeAgent
from lingcore_harbor_bridge.supervision import Supervisor


class HoldingAgent(NopAgent):
    async def run(self, instruction, environment, context):
        await asyncio.Event().wait()


class FinishingTimeoutAgent(BridgeAgent):
    async def run(self, instruction, environment, context):
        sweep = Supervisor.sweep

        async def delayed(supervisor):
            # Force Harbor's five-second agent deadline into the final sweep.
            await asyncio.sleep(7)
            return await sweep(supervisor)

        Supervisor.sweep = delayed
        try:
            await super().run(instruction, environment, context)
        finally:
            Supervisor.sweep = sweep
