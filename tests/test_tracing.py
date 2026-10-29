from __future__ import annotations

import asyncio
import json
import re
import threading

import pytest

from lingcore.tracing import NoopRecorder, TraceRecorder
from lingcore.tracing.inspect import inspect_trace
from lingcore.tracing.local import LocalTraceSink


async def test_span_closes_once_and_classifies_errors(tmp_path):
    sink = LocalTraceSink(tmp_path / "run")
    recorder = TraceRecorder(sink)
    with pytest.raises(ValueError):
        with recorder.span("agent.turn"):
            raise ValueError("secret exception")
    with pytest.raises(asyncio.CancelledError):
        with recorder.span("tool.execute") as span:
            raise asyncio.CancelledError
    span.close()
    await recorder.aclose()
    inspection = inspect_trace(tmp_path / "run")
    assert inspection.complete
    ends = [r for r in inspection.records if r.kind == "span.end"]
    assert [r.status for r in ends] == ["error", "cancelled"]
    assert all(re.fullmatch("[0-9a-f]{32}", r.trace_id) for r in ends)
    assert "secret exception" not in (tmp_path / "run/events.jsonl").read_text()


async def test_parallel_spans_inherit_parent_and_thread_emission(tmp_path):
    sink = LocalTraceSink(tmp_path / "run")
    recorder = TraceRecorder(sink)
    with recorder.span("agent.turn") as root:

        async def work():
            with recorder.span("tool.execute"):
                await asyncio.to_thread(recorder.event, "policy.decision")

        await asyncio.gather(work(), work())
    await recorder.aclose()
    result = inspect_trace(tmp_path / "run")
    assert result.complete
    children = [
        r for r in result.records if r.name == "tool.execute" and r.kind == "span.start"
    ]
    assert len(children) == 2
    assert {r.parent_span_id for r in children} == {root.context.span_id}
    assert len({r.span_id for r in children}) == 2


async def test_overflow_health_survives_unwritable_journal(tmp_path, monkeypatch):
    # Hold initialization so overflow is deterministic, independently of disks.
    started = threading.Event()
    release = threading.Event()
    original = LocalTraceSink._write

    def blocked(self):
        started.set()
        release.wait(5)
        original(self)

    monkeypatch.setattr(LocalTraceSink, "_write", blocked)
    sink = LocalTraceSink(tmp_path / "run", queue_capacity=1)
    assert await asyncio.to_thread(started.wait, 2)
    recorder = TraceRecorder(sink)
    recorder.event("policy.decision")
    recorder.event("policy.decision")
    assert sink.health.dropped_records == 1
    release.set()
    await sink.flush()
    recorder.event("policy.decision")
    await sink.aclose()
    result = inspect_trace(tmp_path / "run")
    assert not result.complete
    assert any("sequence gap" in w for w in result.warnings)


async def test_disk_error_and_symlink_fail_open(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    sink = LocalTraceSink(tmp_path / "link/run")
    recorder = TraceRecorder(sink)
    recorder.event("policy.decision")
    await sink.flush()
    health = await recorder.aclose()
    assert health.disk_errors == 1 and not health.complete
    assert not list(outside.iterdir())


async def test_crash_tail_open_span_and_unknown_schema(tmp_path):
    sink = LocalTraceSink(tmp_path / "run")
    recorder = TraceRecorder(sink)
    recorder.span("agent.turn")
    await recorder.aclose()
    journal = tmp_path / "run/events.jsonl"
    with journal.open("ab") as handle:
        handle.write(b'{"schema_version":')
    result = inspect_trace(tmp_path / "run")
    assert not result.complete and len(result.interrupted_spans) == 1
    assert any("truncated" in w for w in result.warnings)
    payload = json.loads(journal.read_bytes().splitlines()[0])
    payload["schema_version"] = "lingcore.trace.v2"
    journal.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError):
        inspect_trace(tmp_path / "run")


def test_noop_and_metadata_allowlist(tmp_path):
    with NoopRecorder().span("agent.turn"):
        pass
    assert not list(tmp_path.iterdir())
    recorder = TraceRecorder()
    # No-op has no serialization work. The enabled sink enforces the schema.
    sink = LocalTraceSink(tmp_path / "run")
    recorder = TraceRecorder(sink)
    with pytest.raises(ValueError, match="allowlisted"):
        recorder.event("tool.execute", attributes={"arguments": "secret"})
    asyncio.run(recorder.aclose())


async def test_byte_cap_preserves_prefix_and_writes_one_marker(tmp_path):
    sink = LocalTraceSink(tmp_path / "run", max_bytes=4096)
    recorder = TraceRecorder(sink)
    for _ in range(100):
        recorder.event(
            "policy.decision", attributes={"decision": "allow", "policy": "profile"}
        )
    health = await recorder.aclose()
    assert health.capped and not health.complete
    result = inspect_trace(tmp_path / "run")
    assert len([r for r in result.records if r.name == "trace.cap"]) == 1
    assert result.records[-1].name == "trace.cap"
    assert not any("sequence" in w for w in result.warnings)
    assert (
        sum(p.stat().st_size for p in (tmp_path / "run").rglob("*") if p.is_file())
        <= 4096
    )
