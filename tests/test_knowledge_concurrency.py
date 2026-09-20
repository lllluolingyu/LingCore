"""Concurrent index updates must preserve every successfully published source."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from lingcore.errors import ToolError
from lingcore.tools.builtin.knowledge import knowledge
from tests.test_knowledge import FakeEmbedder, _ctx, _indexed_opts

pytestmark = pytest.mark.skipif(os.name != "posix", reason="confined POSIX index")


class PausedEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, inputs):
        self.entered.set()
        await self.release.wait()
        return await super().embed(inputs)


def _prepare(workspace: Path):
    (workspace / "a.txt").write_text("feline", encoding="utf-8")
    (workspace / "b.txt").write_text("canine", encoding="utf-8")


def _documents(workspace: Path):
    connection = sqlite3.connect(workspace / ".lingcore" / "knowledge.sqlite3")
    try:
        return connection.execute("SELECT path FROM documents ORDER BY path").fetchall()
    finally:
        connection.close()


def _index(workspace: Path, path: str, provider: FakeEmbedder):
    return knowledge(
        knowledge.args_model(action="index", paths=[path]),
        _ctx(workspace, _indexed_opts(sources=["*.txt"]), embedder=provider),
    )


async def test_parallel_partial_index_updates_preserve_both_documents(tmp_path):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    second = None
    try:
        await asyncio.wait_for(provider.entered.wait(), timeout=2)
        second = asyncio.create_task(_index(tmp_path, "b.txt", FakeEmbedder()))
        # Let the competing writer publish if no lock protects the first snapshot.
        await asyncio.wait({second}, timeout=0.1)
        provider.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
        assert _documents(tmp_path) == [("a.txt",), ("b.txt",)]
    finally:
        provider.release.set()
        tasks = [task for task in (first, second) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_partial_index_updates_serialize_across_processes(tmp_path):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    process = None
    try:
        await asyncio.wait_for(provider.entered.wait(), timeout=2)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            """
import asyncio
import sys
from pathlib import Path
from tests.test_knowledge_concurrency import _index
from tests.test_knowledge import FakeEmbedder

async def main():
    print("started", flush=True)
    print(await _index(Path(sys.argv[1]), "b.txt", FakeEmbedder()), flush=True)

asyncio.run(main())
""",
            str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert await asyncio.wait_for(process.stdout.readline(), 5) == b"started\n"
        # If the child is unlocked it finishes before the parent publishes.
        output = asyncio.create_task(process.communicate())
        await asyncio.wait({output}, timeout=0.2)
        provider.release.set()
        await asyncio.wait_for(first, 2)
        stdout, stderr = await asyncio.wait_for(output, 5)
        assert process.returncode == 0, stderr.decode()
        assert b"indexed files=" in stdout
        assert _documents(tmp_path) == [("a.txt",), ("b.txt",)]
    finally:
        provider.release.set()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()


async def test_cancelled_waiter_does_not_release_another_writers_lock(tmp_path):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    waiter = None
    try:
        await asyncio.wait_for(provider.entered.wait(), 2)
        waiter = asyncio.create_task(_index(tmp_path, "b.txt", FakeEmbedder()))
        await asyncio.wait({waiter}, timeout=0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not (tmp_path / ".lingcore" / "knowledge.sqlite3").exists()
        provider.release.set()
        await asyncio.wait_for(first, 2)
        await asyncio.wait_for(_index(tmp_path, "b.txt", FakeEmbedder()), 2)
        assert _documents(tmp_path) == [("a.txt",), ("b.txt",)]
    finally:
        provider.release.set()
        tasks = [task for task in (first, waiter) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("failure", ["cancel", "error"])
async def test_failed_writer_releases_index_lock(tmp_path, failure):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    await asyncio.wait_for(provider.entered.wait(), 2)
    if failure == "cancel":
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    else:
        provider._vector = lambda value: []
        provider.release.set()
        with pytest.raises(ToolError):
            await first
    await asyncio.wait_for(_index(tmp_path, "b.txt", FakeEmbedder()), 2)
    assert _documents(tmp_path) == [("b.txt",)]


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
async def test_unsafe_lock_entry_is_rejected(tmp_path, kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _prepare(workspace)
    runtime = workspace / ".lingcore"
    runtime.mkdir()
    lock = runtime / "knowledge.sqlite3.lock"
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    if kind == "symlink":
        lock.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, lock)
    elif kind == "directory":
        lock.mkdir()
    else:
        os.mkfifo(lock)

    with pytest.raises(ToolError, match="lock"):
        await _index(workspace, "a.txt", FakeEmbedder())
    assert outside.read_bytes() == b"untouched"
    assert not (runtime / "knowledge.sqlite3").exists()


async def test_replaced_lock_is_detected_before_index_publication(tmp_path):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    try:
        await asyncio.wait_for(provider.entered.wait(), 2)
        lock = tmp_path / ".lingcore" / "knowledge.sqlite3.lock"
        lock.unlink(missing_ok=True)
        lock.touch()
        await _index(tmp_path, "b.txt", FakeEmbedder())
        provider.release.set()
        with pytest.raises(ToolError, match="lock"):
            await first
        assert _documents(tmp_path) == [("b.txt",)]
    finally:
        provider.release.set()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


async def test_custom_index_and_lock_are_excluded_from_sources(tmp_path):
    _prepare(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "custom.sqlite3.lock").touch()
    ctx = _ctx(
        tmp_path,
        _indexed_opts(index_path="state/custom.sqlite3"),
        embedder=FakeEmbedder(),
    )
    report = await knowledge(knowledge.args_model(action="index"), ctx)
    assert "indexed files=2" in report
    status = await knowledge(knowledge.args_model(action="status"), ctx)
    assert "source files=2" in status
    assert "stale files=0" in status


async def test_waiter_refuses_replaced_lock_while_original_is_held(tmp_path):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    waiter = None
    try:
        await asyncio.wait_for(provider.entered.wait(), 2)
        waiter = asyncio.create_task(_index(tmp_path, "b.txt", FakeEmbedder()))
        await asyncio.wait({waiter}, timeout=0.1)
        lock = tmp_path / ".lingcore" / "knowledge.sqlite3.lock"
        lock.unlink()
        lock.touch()
        with pytest.raises(ToolError, match="lock"):
            await asyncio.wait_for(waiter, 2)
        assert not (tmp_path / ".lingcore" / "knowledge.sqlite3").exists()
    finally:
        tasks = [task for task in (first, waiter) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_unrelated_index_does_not_wait_for_busy_index(tmp_path):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    try:
        await asyncio.wait_for(provider.entered.wait(), 2)
        ctx = _ctx(
            tmp_path,
            _indexed_opts(index_path=".lingcore/other.sqlite3"),
            embedder=FakeEmbedder(),
        )
        report = await asyncio.wait_for(
            knowledge(knowledge.args_model(action="index", paths=["b.txt"]), ctx),
            2,
        )
        assert "indexed files=1" in report
        assert not first.done()
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


async def test_parent_replacement_during_embedding_refuses_old_writer(tmp_path):
    _prepare(tmp_path)
    provider = PausedEmbedder()
    first = asyncio.create_task(_index(tmp_path, "a.txt", provider))
    try:
        await asyncio.wait_for(provider.entered.wait(), 2)
        runtime = tmp_path / ".lingcore"
        runtime.rename(tmp_path / "held")
        runtime.mkdir()
        await _index(tmp_path, "b.txt", FakeEmbedder())
        provider.release.set()
        with pytest.raises(ToolError, match="lock"):
            await first
        assert _documents(tmp_path) == [("b.txt",)]
        assert not (tmp_path / "held" / "knowledge.sqlite3").exists()
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
