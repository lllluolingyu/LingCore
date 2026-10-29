"""Bounded producer queue, one confined JSONL writer, and out-of-band health."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from lingcore.paths import confined_directory
from lingcore.tracing.models import TraceHealth, TraceRecord

_LOG = logging.getLogger(__name__)
_Item = TraceRecord | tuple[str, bytes] | threading.Event | None


class LocalTraceSink:
    def __init__(
        self,
        directory: Path,
        *,
        manifest: dict[str, Any] | None = None,
        queue_capacity: int = 4096,
        flush_interval_seconds: float = 1,
        max_bytes: int = 64 * 1024 * 1024,
        record_max_bytes: int = 65536,
    ) -> None:
        if queue_capacity < 1 or max_bytes < 4096 or flush_interval_seconds <= 0:
            raise ValueError("invalid trace sink bounds")
        self.directory = directory.absolute()
        self._queue: queue.Queue[_Item] = queue.Queue(queue_capacity)
        self._health = TraceHealth()
        self._lock = threading.Lock()
        self._closed = False
        self._warned = False
        self._interval = flush_interval_seconds
        self._max = max_bytes
        self._record_max = record_max_bytes
        self._manifest = dict(manifest or {})
        self._thread = threading.Thread(
            target=self._write, name="lingcore-trace", daemon=True
        )
        self._thread.start()

    @property
    def health(self) -> TraceHealth:
        with self._lock:
            return self._health

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._health = self._health.model_copy(update=values)

    def _drop(self, *, disk: bool = False) -> None:
        with self._lock:
            h = self._health
            self._health = h.model_copy(
                update={
                    "complete": False,
                    "dropped_records": h.dropped_records + (0 if disk else 1),
                    "disk_errors": h.disk_errors + int(disk),
                }
            )
            warn = not self._warned
            self._warned = True
        if warn:
            _LOG.warning("Trace capture is incomplete; inspect trace health")

    def _enqueue(self, item: _Item) -> bool:
        if self._closed or self.health.capped:
            self._drop()
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._drop()
            return False

    def try_emit(self, record: TraceRecord) -> bool:
        return self._enqueue(record)

    def try_artifact(self, artifact_id: str, data: bytes) -> bool:
        if len(artifact_id) != 32 or any(
            c not in "0123456789abcdef" for c in artifact_id
        ):
            raise ValueError("invalid artifact identity")
        return self._enqueue((artifact_id, data))

    async def _fence(self, item: _Item, timeout: float) -> bool:
        def enqueue() -> bool:
            try:
                self._queue.put(item, timeout=max(0, timeout))
                return True
            except queue.Full:
                return False

        return await asyncio.to_thread(enqueue)

    async def flush(self, timeout_seconds: float = 2) -> TraceHealth:
        if not self._thread.is_alive():
            return self.health
        fence = threading.Event()
        try:
            async with asyncio.timeout(max(0.001, timeout_seconds)):
                admitted = await self._fence(fence, timeout_seconds)
                done = admitted and await asyncio.to_thread(fence.wait, timeout_seconds)
                if not done:
                    self._update(complete=False, flush_timed_out=True)
        except TimeoutError:
            self._update(complete=False, flush_timed_out=True)
        return self.health

    async def aclose(self, timeout_seconds: float = 2) -> TraceHealth:
        if not self._closed:
            self._closed = True
            try:
                async with asyncio.timeout(max(0.001, timeout_seconds)):
                    if not await self._fence(None, timeout_seconds):
                        self._update(complete=False, flush_timed_out=True)
                    await asyncio.to_thread(self._thread.join, timeout_seconds)
            except TimeoutError:
                self._update(complete=False, flush_timed_out=True)
        if self._thread.is_alive():
            self._update(complete=False, flush_timed_out=True)
        return self.health

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            threading.Thread(target=self._queue.put, args=(None,), daemon=True).start()

    def _write(self) -> None:
        stopped = False
        try:
            # Anchor at the filesystem root: no component of the destination,
            # including its parents, may redirect a sensitive state write.
            with confined_directory(
                Path(self.directory.anchor),
                self.directory.relative_to(self.directory.anchor),
                create=True,
                permissions=0o700,
            ) as directory:
                data = json.dumps(self._manifest, ensure_ascii=True).encode() + b"\n"
                if len(data) > min(65536, self._max // 4):
                    raise ValueError("trace manifest exceeds its bound")
                with directory.open_exclusive("manifest.json", 0o600) as manifest:
                    manifest.write(data)
                self._update(written_bytes=len(data))
                with directory.open_exclusive("events.jsonl", 0o600) as journal:
                    last: TraceRecord | None = None
                    flushed_at = time.monotonic()
                    while True:
                        try:
                            item = self._queue.get(timeout=self._interval)
                        except queue.Empty:
                            journal.flush()
                            continue
                        if item is None:
                            stopped = True
                            break
                        if isinstance(item, threading.Event):
                            journal.flush()
                            item.set()
                            continue
                        if self.health.capped:
                            continue
                        if isinstance(item, TraceRecord):
                            last = item
                            payload = item.model_dump_json().encode() + b"\n"
                            if len(payload) > self._record_max:
                                self._drop()
                                continue
                        else:
                            payload = item[1]
                        if self.health.written_bytes + len(payload) > self._max - 2048:
                            self._update(complete=False, capped=True)
                            if last is not None:
                                marker = last.model_copy(
                                    update={
                                        "kind": "event",
                                        "name": "trace.cap",
                                        "seq": last.seq
                                        + int(not isinstance(item, TraceRecord)),
                                        "event_id": str(uuid4()),
                                        "status": None,
                                        "attributes": {},
                                    }
                                )
                                payload = marker.model_dump_json().encode() + b"\n"
                                journal.write(payload)
                                journal.flush()
                                self._update(
                                    written_bytes=self.health.written_bytes
                                    + len(payload)
                                )
                            self._drop()
                            continue
                        directory.ensure_anchored()
                        if isinstance(item, TraceRecord):
                            journal.write(payload)
                            self._update(
                                written_records=self.health.written_records + 1
                            )
                        else:
                            with directory.subdirectory(
                                "artifacts", create=True, permissions=0o700
                            ) as artifacts:
                                with artifacts.open_exclusive(
                                    item[0] + ".json", 0o600
                                ) as artifact:
                                    artifact.write(payload)
                        self._update(
                            written_bytes=self.health.written_bytes + len(payload)
                        )
                        if time.monotonic() - flushed_at >= self._interval:
                            journal.flush()
                            flushed_at = time.monotonic()
                    journal.flush()
                summary_name = ".summary-" + uuid4().hex
                with directory.open_exclusive(summary_name, 0o600) as summary:
                    summary.write(self.health.model_dump_json().encode() + b"\n")
                directory.replace(summary_name, "summary.json")
        except Exception:
            self._drop(disk=True)
            if stopped:
                return
            # Producers fail open even if initialization or the disk failed.
            # Fences still complete, and health remains available without I/O.
            while True:
                item = self._queue.get()
                if item is None:
                    return
                if isinstance(item, threading.Event):
                    item.set()
