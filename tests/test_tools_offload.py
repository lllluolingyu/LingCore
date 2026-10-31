"""Tools tests: offload."""

from __future__ import annotations

import hashlib

import pytest

from lingcore.errors import ConfigError
from lingcore.paths import ConfinedDirectory
from lingcore.tools import ToolContext, ToolRegistry, tool
from lingcore.tools.builtin._offload import offload_text
from lingcore.tools.builtin.fs import (
    ReadArgs,
    read_file,
)
from tests.support.tools import _IntArgs
from tests.support.tools import ctx as ctx


def test_offload_inline_below_threshold(ctx):
    assert offload_text(ctx, source="x", text="small", threshold=1000) == "small"


async def test_offload_writes_file_readable_via_read_file(ctx):
    big = "\n".join(f"row{i}" for i in range(1, 501))
    out = offload_text(ctx, source="shell", text=big, threshold=100)
    assert "full output" in out and ".lingcore/tool-output/shell-" in out
    rel = out.split("→")[1].split(";")[0].strip()
    content = await read_file(ReadArgs(path=rel, limit=3), ctx)
    assert "1\trow1" in content


async def test_read_file_pages_large_offloaded_output(ctx):
    # A >256 KiB staged log must still be pageable; the whole-file guard used
    # to reject even a three-line read.
    big = "\n".join(f"line {i}" for i in range(1, 30_001))
    assert len(big.encode()) > 256 * 1024
    out = offload_text(ctx, source="shell", text=big, threshold=100)
    rel = out.split("→")[1].split(";")[0].strip()

    window = await read_file(ReadArgs(path=rel, offset=1, limit=3), ctx)
    assert window.splitlines()[:3] == ["1\tline 1", "2\tline 2", "3\tline 3"]
    assert "more lines" in window

    tail = await read_file(ReadArgs(path=rel, offset=29_998, limit=3), ctx)
    assert tail.splitlines()[0].startswith("29998\tline 29998")


def test_read_window_does_not_consume_unrequested_tail():
    # Detecting "more lines" must peek, not read a huge newline-free line.
    import io

    from lingcore.tools.builtin import fs

    class _CountingStream(io.BytesIO):
        consumed = 0

        def read(self, size=-1):
            chunk = super().read(size)
            self.consumed += len(chunk)
            return chunk

    stream = _CountingStream(b"first\n" + b"x" * (16 * 1024 * 1024))
    window = fs._format_stream_lines(
        stream, offset=1, limit=1, max_lines=2_000, max_line_chars=2_000
    )
    assert window.splitlines()[0] == "1\tfirst"
    assert "more lines" in window
    assert stream.consumed <= 2 * fs._READ_CHUNK_BYTES


async def test_read_file_large_file_splits_lines_like_small_files(ctx):
    # CR and CRLF terminators number lines exactly like str.splitlines(), and
    # a CRLF's "\r" never counts toward a clipped line's length.
    filler = "z" * (300 * 1024)
    for sep in ("\r", "\r\n"):
        path = ctx.workspace / "big.txt"
        path.write_bytes(sep.join(["alpha", "beta", filler, "omega"]).encode())
        out = await read_file(ReadArgs(path="big.txt", limit=4), ctx)
        rows = out.splitlines()
        assert rows[0] == "1\talpha" and rows[1] == "2\tbeta"
        assert rows[2].endswith(f"(+{len(filler) - 2_000} chars)")
        assert rows[3] == "4\tomega"
        assert "more lines" not in out


def test_stream_lines_matches_splitlines_across_chunk_boundaries():
    import io

    from lingcore.tools.builtin import fs

    text = "a\r\nb\rc\n\nd\x0be\u2028f\r"
    for pad in range(0, 4):
        body = "p" * (fs._READ_CHUNK_BYTES - 1 - pad) + text
        lines = fs._StreamLines(io.BytesIO(body.encode()), max_line_chars=10**9)
        got = []
        while (line := lines.next_line()) is not None:
            got.append(line)
        assert got == body.splitlines()


def test_offload_filename_is_content_stable(ctx):
    big = "z" * 5000
    a = offload_text(ctx, source="shell", text=big, threshold=100)
    b = offload_text(ctx, source="shell", text=big, threshold=100)
    assert a == b  # identical content → identical file → identical note


def test_offload_final_symlink_is_replaced_not_followed(ctx, tmp_path):
    big = "sensitive fetched output" * 500
    digest = hashlib.sha256(big.encode()).hexdigest()[:12]
    output_dir = ctx.workspace / ".lingcore" / "tool-output"
    output_dir.mkdir(parents=True)
    dest = output_dir / f"fetch-{digest}.txt"
    outside = tmp_path.parent / f"{tmp_path.name}-outside-offload"
    try:
        dest.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this platform")

    out = offload_text(ctx, source="fetch", text=big, threshold=100)

    assert "full output" in out
    assert not outside.exists()
    assert dest.is_file() and not dest.is_symlink()
    assert dest.read_text(encoding="utf-8") == big


def test_offload_parent_swap_cannot_redirect_write(ctx, tmp_path, monkeypatch):
    """Replacing the opened runtime tree must fail closed, then truncate."""
    runtime = ctx.workspace / ".lingcore"
    moved = tmp_path.parent / f"{tmp_path.name}-runtime-held"
    real_open = ConfinedDirectory.open_exclusive
    swapped = False

    def swap_then_open(self, name, mode=0o644):
        nonlocal swapped
        if name.endswith(".part") and not swapped:
            runtime.rename(moved)
            runtime.symlink_to(moved, target_is_directory=True)
            swapped = True
        return real_open(self, name, mode)

    monkeypatch.setattr(ConfinedDirectory, "open_exclusive", swap_then_open)
    big = "z" * 5000
    out = offload_text(
        ctx,
        source="fetch",
        text=big,
        threshold=100,
        fallback_max_chars=100,
    )

    assert "full output" not in out and "truncated" in out
    assert not list(moved.rglob("*.txt"))
    assert not list(moved.rglob("*.part"))


def test_offload_disabled_truncates(ctx):
    out = offload_text(
        ctx, source="shell", text="z" * 5000, threshold=0, fallback_max_chars=100
    )
    assert "truncated" in out and len(out) < 200


def test_json_schema_shape():
    schema = read_file.json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_file"
    assert "path" in schema["function"]["parameters"]["properties"]


def test_registry_subset_and_unknown():
    reg = ToolRegistry()

    @tool(name="t1", registry=reg)
    async def t1(args: _IntArgs, ctx: ToolContext) -> str:
        return "ok"

    sub = reg.subset(["t1"])
    assert sub.get("t1").name == "t1"
    with pytest.raises(ConfigError, match="unknown tool"):
        reg.subset(["t1", "ghost"])


def test_decorator_requires_basemodel():
    reg = ToolRegistry()
    with pytest.raises(ConfigError, match="BaseModel"):

        @tool(registry=reg)
        async def bad(args: int, ctx: ToolContext) -> str:  # type: ignore[arg-type]
            return ""
