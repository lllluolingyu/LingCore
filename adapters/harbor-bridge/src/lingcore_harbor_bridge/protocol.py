"""Independent stdlib-only wire model; shared fixtures check interoperability."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from lingcore_harbor_bridge import PROTOCOL

MAX_FRAME_BYTES = 4 * 1024 * 1024


class BridgeProtocolError(Exception):
    pass


@dataclass(frozen=True)
class Message:
    data: dict[str, Any]

    @property
    def type(self) -> str:
        return str(self.data["type"])


def _fields(value: Any, required: set[str], defaults: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or required - value.keys()
        or value.keys() - required - defaults.keys()
    ):
        raise BridgeProtocolError("invalid bridge fields")
    return copy.deepcopy(defaults) | value


def _string(value: Any) -> None:
    if not isinstance(value, str):
        raise BridgeProtocolError("invalid bridge string")


def _number(
    value: Any, *, minimum: float = 0, integer: bool = False, positive: bool = False
) -> None:
    if (
        type(value) not in ({int} if integer else {int, float})
        or isinstance(value, float)
        and not math.isfinite(value)
        or value < minimum
        or positive
        and value == 0
    ):
        raise BridgeProtocolError("invalid bridge number")


def _usage(value: Any) -> dict[str, Any]:
    result = _fields(
        value,
        set(),
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "requests": 0,
            "coverage": "incomplete",
            "per_model": {},
        },
    )
    for key in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "requests",
    ):
        _number(result[key], integer=True)
    if result["coverage"] not in ("complete", "incomplete") or not isinstance(
        result["per_model"], dict
    ):
        raise BridgeProtocolError("invalid bridge usage")
    for model, counts in result["per_model"].items():
        _string(model)
        if not isinstance(counts, dict):
            raise BridgeProtocolError("invalid per-model usage")
        for key, count in counts.items():
            _string(key)
            _number(count, integer=True)
    return result


def validate(value: Any) -> Message:
    if not isinstance(value, dict):
        raise BridgeProtocolError("invalid bridge frame")
    kind = value.get("type")
    if kind == "hello":
        data = _fields(
            value,
            {
                "type",
                "execution_id",
                "bridge",
                "harbor_version",
                "task",
                "instruction",
                "probe",
            },
            {"protocol": PROTOCOL, "environment": "docker"},
        )
        for key in ("execution_id", "harbor_version", "task", "instruction"):
            _string(data[key])
        bridge = _fields(data["bridge"], {"version", "digest"}, {})
        _string(bridge["version"])
        if not isinstance(bridge["digest"], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", bridge["digest"]
        ):
            raise BridgeProtocolError("invalid bridge digest")
        probe = _fields(data["probe"], {"supported"}, {"missing": []})
        if (
            type(probe["supported"]) is not bool
            or not isinstance(probe["missing"], list)
            or not all(isinstance(x, str) for x in probe["missing"])
        ):
            raise BridgeProtocolError("invalid bridge probe")
        if data["protocol"] != PROTOCOL or data["environment"] != "docker":
            raise BridgeProtocolError("unsupported bridge protocol or environment")
        data.update(bridge=bridge, probe=probe)
    elif kind == "exec":
        data = _fields(
            value,
            {"type", "call_id", "command", "timeout", "capture_bytes"},
            {"cwd": None},
        )
        _string(data["call_id"])
        _string(data["command"])
        if (
            not data["command"]
            or data["cwd"] is not None
            and not isinstance(data["cwd"], str)
        ):
            raise BridgeProtocolError("invalid bridge command")
        _number(data["timeout"], positive=True)
        _number(data["capture_bytes"], positive=True, integer=True)
        if data["capture_bytes"] > 2 * 1024 * 1024:
            raise BridgeProtocolError("invalid capture limit")
    elif kind == "interrupt":
        data = _fields(value, {"type", "call_id"}, {})
        _string(data["call_id"])
    elif kind == "exec_result":
        data = _fields(
            value,
            {
                "type",
                "call_id",
                "exit_code",
                "output",
                "total_output_bytes",
                "truncated",
                "timed_out",
                "duration",
            },
            {},
        )
        _string(data["call_id"])
        _string(data["output"])
        _number(data["exit_code"], minimum=-math.inf, integer=True)
        if data["total_output_bytes"] is not None:
            _number(data["total_output_bytes"], integer=True)
        _number(data["duration"])
        if type(data["truncated"]) is not bool or type(data["timed_out"]) is not bool:
            raise BridgeProtocolError("invalid bridge result")
    elif kind == "usage":
        data = _fields(value, {"type", "usage"}, {})
        data["usage"] = _usage(data["usage"])
    elif kind == "finish":
        data = _fields(
            value,
            {"type", "outcome", "usage", "metadata"},
            {"trajectory": None, "atif_reason": None},
        )
        data["usage"] = _usage(data["usage"])
        if data["atif_reason"] not in (
            None,
            "metadata_only",
            "disabled",
            "oversized",
            "incomplete",
        ):
            raise BridgeProtocolError("invalid ATIF reason")
        if (
            not isinstance(data["outcome"], dict)
            or not isinstance(data["metadata"], dict)
            or data["trajectory"] is not None
            and not isinstance(data["trajectory"], dict)
        ):
            raise BridgeProtocolError("invalid bridge finish")
    elif kind == "finished":
        data = _fields(value, {"type", "sweep", "atif"}, {})
        if data["sweep"] not in ("clean", "failed") or data["atif"] not in (
            "written",
            "invalid",
            "absent",
        ):
            raise BridgeProtocolError("invalid bridge finish acknowledgement")
    elif kind == "abort":
        data = _fields(value, {"type", "reason"}, {})
        if data["reason"] not in ("cancelled", "harness_timeout", "protocol_error"):
            raise BridgeProtocolError("invalid bridge abort")
    else:
        raise BridgeProtocolError("unknown bridge frame")
    return Message(data)


def encode(message: Message) -> bytes:
    try:
        value = validate(message.data)
        data = (
            json.dumps(
                value.data,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise BridgeProtocolError("invalid outgoing bridge frame") from None
    if len(data) > MAX_FRAME_BYTES:
        raise BridgeProtocolError("oversized bridge frame")
    return data


def decode(data: bytes) -> Message:
    if len(data) > MAX_FRAME_BYTES or not data.endswith(b"\n"):
        raise BridgeProtocolError("oversized or incomplete bridge frame")
    try:
        return validate(
            json.loads(
                data.decode("utf-8"),
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
            )
        )
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise BridgeProtocolError("invalid bridge frame") from None


class Channel:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer

    async def send(self, message: Message) -> None:
        self.writer.write(encode(message))
        await self.writer.drain()

    async def receive(self) -> Message:
        try:
            data = await self.reader.readline()
        except (ValueError, OSError):
            raise BridgeProtocolError("invalid or oversized bridge frame") from None
        if not data:
            raise BridgeProtocolError("bridge connection closed")
        return decode(data)

    async def aclose(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except (OSError, ConnectionError):
            pass
