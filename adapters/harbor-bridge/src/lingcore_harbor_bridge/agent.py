"""Harbor agent that only relays commands; no profile or model client lives here."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage
from harbor.models.trajectories.trajectory import Trajectory

from lingcore_harbor_bridge import HARBOR_VERSION, __version__
from lingcore_harbor_bridge.describe import canonical, package_digest
from lingcore_harbor_bridge.protocol import (
    MAX_FRAME_BYTES,
    BridgeProtocolError,
    Channel,
    validate,
)
from lingcore_harbor_bridge.supervision import Supervisor


class BridgeAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)

    def __init__(
        self,
        *args: Any,
        socket_path: str = "",
        execution_id: str = "",
        task_name: str = "",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.socket_path, self.execution_id, self.task_name = (
            socket_path,
            execution_id,
            task_name,
        )

    @staticmethod
    def name() -> str:
        return "lingcore-bridge"

    def version(self) -> str:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    def _usage(self, context: AgentContext, usage: dict[str, Any]) -> None:
        context.n_input_tokens = usage["input_tokens"]
        context.n_cache_tokens = usage["cached_input_tokens"]
        context.n_output_tokens = usage["output_tokens"]
        context.cost_usd = None
        context.model_usage = {
            model: ModelUsage(
                n_input_tokens=value.get("input_tokens", 0),
                n_cache_tokens=value.get("cached_input_tokens", 0),
                n_output_tokens=value.get("output_tokens", 0),
                cost_usd=None,
            )
            for model, value in usage["per_model"].items()
        }
        assert context.metadata is not None
        context.metadata["lingcore"]["usage_coverage"] = usage["coverage"]

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        supervisor = Supervisor(environment)
        channel: Channel | None = None
        command: asyncio.Task[dict[str, Any]] | None = None
        receive: asyncio.Task[Any] | None = None
        finish: dict[str, Any] | None = None
        context.metadata = {
            "lingcore": {
                "execution_id": self.execution_id,
                "usage_coverage": "incomplete",
            }
        }
        atif = "absent"
        try:
            missing = await supervisor.probe()
            reader, writer = await asyncio.open_unix_connection(
                self.socket_path, limit=MAX_FRAME_BYTES
            )
            channel = Channel(reader, writer)
            await channel.send(
                validate(
                    dict(
                        type="hello",
                        execution_id=self.execution_id,
                        bridge={
                            "version": __version__,
                            "digest": package_digest(Path(__file__).parent),
                        },
                        harbor_version=HARBOR_VERSION,
                        task=self.task_name,
                        instruction=instruction,
                        probe={"supported": not missing, "missing": missing},
                    )
                )
            )
            while True:
                if receive is None:
                    receive = asyncio.create_task(channel.receive())
                waiting = {receive} if command is None else {receive, command}
                done, _ = await asyncio.wait(
                    waiting, return_when=asyncio.FIRST_COMPLETED
                )
                if command is not None and command in done:
                    await channel.send(validate(command.result()))
                    command = None
                if receive not in done:
                    continue
                message = receive.result()
                receive = None
                data = message.data
                if message.type == "usage":
                    self._usage(context, data["usage"])
                elif message.type == "exec" and command is None and not missing:
                    command = asyncio.create_task(supervisor.execute(data))
                    active_request = data
                elif message.type == "interrupt":
                    # A reply may have crossed the interrupt on the socket.
                    if command is not None:
                        if data["call_id"] != active_request["call_id"]:
                            raise BridgeProtocolError("interrupt call mismatch")
                        reply = await supervisor.stop_command(command, active_request)
                        command = None
                        await channel.send(validate(reply))
                elif message.type == "finish" and command is None:
                    finish = data
                    self._usage(context, data["usage"])
                    context.metadata["lingcore"].update(data["metadata"])
                    context.metadata["lingcore"].update(
                        outcome=data["outcome"], atif_reason=data["atif_reason"]
                    )
                    if data["trajectory"] is not None:
                        try:
                            trajectory = Trajectory.model_validate(data["trajectory"])
                            self.logs_dir.mkdir(parents=True, exist_ok=True)
                            with (self.logs_dir / "trajectory.json").open("xb") as file:
                                file.write(
                                    canonical(trajectory.model_dump(mode="json"))
                                )
                            atif = "written"
                        except (ValueError, OSError):
                            atif = "invalid"
                    break
                elif message.type == "abort":
                    context.metadata["lingcore"]["abort"] = data["reason"]
                    break
                else:
                    raise BridgeProtocolError("unexpected relay message")
        except asyncio.CancelledError:
            context.metadata["lingcore"]["abort"] = "harness_timeout"
            raise
        except BridgeProtocolError as exc:
            # EOF after the driving process dies still lets Harbor verify and
            # preserve its result; LingCore marks its own execution interrupted.
            context.metadata["lingcore"]["protocol_error"] = str(exc)
        finally:

            async def finalize() -> None:
                if receive is not None:
                    receive.cancel()
                    await asyncio.gather(receive, return_exceptions=True)
                if command is not None:
                    command.cancel()
                    await asyncio.gather(command, return_exceptions=True)
                clean = await supervisor.sweep()
                assert context.metadata is not None
                context.metadata["lingcore"].update(
                    sweep="clean" if clean else "failed", atif=atif
                )
                if channel is not None:
                    try:
                        if (
                            context.metadata["lingcore"].get("abort")
                            == "harness_timeout"
                        ):
                            await channel.send(
                                validate({"type": "abort", "reason": "harness_timeout"})
                            )
                        elif finish is not None:
                            await channel.send(
                                validate(
                                    dict(
                                        type="finished",
                                        sweep="clean" if clean else "failed",
                                        atif=atif,
                                    )
                                )
                            )
                    except (OSError, BridgeProtocolError):
                        pass
                    await channel.aclose()

            cleanup = asyncio.create_task(finalize())
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
                    context.metadata["lingcore"]["abort"] = "harness_timeout"
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError
