"""Bounded v1 wire messages; the bridge maintains an independent definition."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from lingcore.execution import RunnerSignal

PROTOCOL: Literal["lingcore.harbor-bridge.v1"] = "lingcore.harbor-bridge.v1"
MAX_FRAME_BYTES = 4 * 1024 * 1024
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class ProtocolFailure(RunnerSignal):
    """Runner control signal: tool error conversion must not swallow it."""


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


class BridgeInfo(WireModel):
    version: str
    digest: Digest


class Probe(WireModel):
    supported: bool
    missing: list[str] = Field(default_factory=list)


class Hello(WireModel):
    type: Literal["hello"] = "hello"
    protocol: Literal["lingcore.harbor-bridge.v1"] = PROTOCOL
    execution_id: str
    bridge: BridgeInfo
    harbor_version: str
    task: str
    instruction: str
    environment: Literal["docker"] = "docker"
    probe: Probe


class Exec(WireModel):
    type: Literal["exec"] = "exec"
    call_id: str
    command: str = Field(min_length=1)
    cwd: str | None = None
    timeout: float = Field(gt=0)
    capture_bytes: int = Field(gt=0, le=2 * 1024 * 1024)


class Interrupt(WireModel):
    type: Literal["interrupt"] = "interrupt"
    call_id: str


class ExecResult(WireModel):
    type: Literal["exec_result"] = "exec_result"
    call_id: str
    exit_code: int
    # Base64 preserves exact captured bytes and bounds JSON expansion. Tool
    # display decodes UTF-8 with replacement; binary never becomes a host path.
    output: str
    total_output_bytes: int | None = Field(ge=0)
    truncated: bool
    timed_out: bool
    duration: float = Field(ge=0)


class UsageTotals(WireModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    coverage: Literal["complete", "incomplete"] = "incomplete"
    per_model: dict[str, dict[str, Annotated[int, Field(ge=0)]]] = Field(
        default_factory=dict
    )


class Usage(WireModel):
    type: Literal["usage"] = "usage"
    usage: UsageTotals


class Finish(WireModel):
    type: Literal["finish"] = "finish"
    outcome: dict[str, Any]
    usage: UsageTotals
    metadata: dict[str, Any]
    trajectory: dict[str, Any] | None = None
    atif_reason: (
        Literal["metadata_only", "disabled", "oversized", "incomplete"] | None
    ) = None


class Finished(WireModel):
    type: Literal["finished"] = "finished"
    sweep: Literal["clean", "failed"]
    atif: Literal["written", "invalid", "absent"]


class Abort(WireModel):
    type: Literal["abort"] = "abort"
    reason: Literal["cancelled", "harness_timeout", "protocol_error"]


Frame = Annotated[
    Hello | Exec | Interrupt | ExecResult | Usage | Finish | Finished | Abort,
    Field(discriminator="type"),
]
_ADAPTER: TypeAdapter[Frame] = TypeAdapter(Frame)


def encode(frame: Frame) -> bytes:
    try:
        data = (
            json.dumps(
                frame.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise ProtocolFailure("invalid outgoing bridge frame") from None
    if (
        len(data) > MAX_FRAME_BYTES
        and isinstance(frame, Finish)
        and frame.trajectory is not None
    ):
        return encode(
            frame.model_copy(update={"trajectory": None, "atif_reason": "oversized"})
        )
    if len(data) > MAX_FRAME_BYTES:
        raise ProtocolFailure("oversized bridge frame")
    return data


def decode(data: bytes) -> Frame:
    if len(data) > MAX_FRAME_BYTES or not data.endswith(b"\n"):
        raise ProtocolFailure("oversized or incomplete bridge frame")
    try:

        def invalid_constant(value: str) -> None:
            raise ValueError("non-finite JSON number")

        return _ADAPTER.validate_python(
            json.loads(data.decode("utf-8"), parse_constant=invalid_constant)
        )
    except (ValueError, ValidationError, RecursionError):
        raise ProtocolFailure("invalid bridge frame") from None


class Channel:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self._write_lock = asyncio.Lock()

    async def send(self, frame: Frame) -> None:
        async with self._write_lock:
            try:
                self.writer.write(encode(frame))
                await self.writer.drain()
            except (ConnectionError, OSError):
                raise ProtocolFailure("bridge connection closed") from None

    async def receive(self) -> Frame:
        try:
            data = await self.reader.readline()
        except (ValueError, asyncio.LimitOverrunError, OSError):
            raise ProtocolFailure("invalid or oversized bridge frame") from None
        if not data:
            raise ProtocolFailure("bridge connection closed")
        return decode(data)

    async def aclose(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except (ConnectionError, OSError):
            pass
