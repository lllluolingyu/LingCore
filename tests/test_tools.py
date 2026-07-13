"""Tests for the tool contract and fs tools (M2).

The path-escape suite is security-critical and intentionally exhaustive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import BaseModel

from lingcore.errors import ConfigError, ToolError
from lingcore.paths import ConfinedDirectory
from lingcore.tools import ToolContext, ToolRegistry, tool
from lingcore.tools.builtin.fs import (
    EditArgs,
    ListArgs,
    ReadArgs,
    SearchArgs,
    WriteArgs,
    _resolve,
    edit_file,
    list_dir,
    read_file,
    search,
    write_file,
)
from lingcore.tools.builtin._offload import offload_text
from lingcore.tools.builtin.pdf import Pdf2MdArgs, pdf2md


class _IntArgs(BaseModel):
    x: int


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return ToolContext(workspace=tmp_path)


# --- path escape (security-critical) -------------------------------------


def test_resolve_allows_inside(ctx):
    assert _resolve(ctx, "a.txt") == (ctx.workspace / "a.txt").resolve()
    assert _resolve(ctx, "sub/b.py").is_file()
    assert _resolve(ctx, ".") == ctx.workspace.resolve()


@pytest.mark.parametrize(
    "bad",
    [
        "../outside.txt",
        "../../etc/passwd",
        "sub/../../escape.txt",
        "/etc/passwd",
        "/tmp/abs",
    ],
)
def test_resolve_rejects_escape(ctx, bad):
    with pytest.raises(ToolError, match="escapes workspace"):
        _resolve(ctx, bad)


def test_resolve_rejects_symlink_escape(ctx, tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = ctx.workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(ToolError, match="escapes workspace"):
        _resolve(ctx, "link.txt")


# --- fs tool behaviour ----------------------------------------------------


async def test_read_file(ctx):
    # Line-numbered output: "<lineno>\t<line>" (single-line file → width 1).
    assert await read_file(ReadArgs(path="a.txt"), ctx) == "1\thello world"


async def test_read_file_offset_and_limit(ctx):
    (ctx.workspace / "big.txt").write_text(
        "\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8"
    )
    out = await read_file(ReadArgs(path="big.txt", offset=3, limit=2), ctx)
    assert out.splitlines()[:2] == ["3\tline3", "4\tline4"]
    assert "showed lines 3–4 of 10" in out


async def test_read_file_default_line_cap(ctx):
    (ctx.workspace / "big.txt").write_text(
        "\n".join(f"l{i}" for i in range(1, 21)), encoding="utf-8"
    )
    ctx.options["read_file"] = {"max_lines": 5}
    out = await read_file(ReadArgs(path="big.txt"), ctx)
    assert len([ln for ln in out.splitlines() if "\t" in ln]) == 5
    assert "of 20; pass offset/limit for more" in out


async def test_read_file_offset_past_eof(ctx):
    out = await read_file(ReadArgs(path="a.txt", offset=99), ctx)
    assert "past the end" in out


async def test_read_file_long_line_truncated(ctx):
    (ctx.workspace / "long.txt").write_text("x" * 5000, encoding="utf-8")
    ctx.options["read_file"] = {"max_line_chars": 100}
    out = await read_file(ReadArgs(path="long.txt"), ctx)
    assert "+4900 chars" in out


async def test_read_missing(ctx):
    with pytest.raises(ToolError, match="not a file"):
        await read_file(ReadArgs(path="nope.txt"), ctx)


async def test_read_file_attaches_image(ctx):
    (ctx.workspace / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nrest")
    out = await read_file(ReadArgs(path="pic.png"), ctx)
    assert out.text.startswith("attached pic.png")
    assert out.attachments[0].kind == "image"
    assert out.attachments[0].media_type == "image/png"


async def test_read_file_attaches_pdf(ctx):
    (ctx.workspace / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    out = await read_file(ReadArgs(path="doc.pdf"), ctx)
    assert out.attachments[0].kind == "file"
    assert out.attachments[0].name == "doc.pdf"


async def test_read_file_rejects_other_binary(ctx):
    (ctx.workspace / "blob.bin").write_bytes(b"abc\x00def")
    with pytest.raises(ToolError, match="binary file"):
        await read_file(ReadArgs(path="blob.bin"), ctx)


async def test_read_file_reads_text_with_unknown_extension(ctx):
    (ctx.workspace / "notes.bin").write_text("just text", encoding="utf-8")
    assert await read_file(ReadArgs(path="notes.bin"), ctx) == "1\tjust text"


async def test_read_file_rejects_escape(ctx):
    with pytest.raises(ToolError, match="escapes workspace"):
        await read_file(ReadArgs(path="../pic.png"), ctx)


# --- pdf2md ----------------------------------------------------------------


async def test_pdf2md_extracts_pages(ctx):
    from tests.test_modality import make_pdf

    (ctx.workspace / "doc.pdf").write_bytes(make_pdf("alpha beta", "gamma"))
    out = await pdf2md(Pdf2MdArgs(path="doc.pdf"), ctx)
    assert "## Page 1" in out and "alpha beta" in out
    assert "## Page 2" in out and "gamma" in out


async def test_pdf2md_rejects_escape(ctx):
    with pytest.raises(ToolError, match="escapes workspace"):
        await pdf2md(Pdf2MdArgs(path="../doc.pdf"), ctx)


async def test_pdf2md_rejects_non_pdf_content(ctx):
    (ctx.workspace / "fake.pdf").write_bytes(b"\x89PNG\r\n\x1a\nrest")
    with pytest.raises(ToolError, match="not a PDF"):
        await pdf2md(Pdf2MdArgs(path="fake.pdf"), ctx)


async def test_pdf2md_arg_caps_output(ctx):
    from tests.test_modality import make_pdf

    # Multi-line text: a single insert_text line is clipped at the page edge,
    # so newlines are what make page 1 long enough to overflow the cap.
    long_page = "\n".join(["lorem ipsum dolor sit amet"] * 12)
    (ctx.workspace / "doc.pdf").write_bytes(make_pdf(long_page, "tail page"))
    out = await pdf2md(Pdf2MdArgs(path="doc.pdf", max_chars=200), ctx)
    assert "[truncated at 200 characters" in out
    assert "tail page" not in out


async def test_pdf2md_tool_options_default(tmp_path):
    from tests.test_modality import make_pdf

    long_page = "\n".join(["lorem ipsum dolor sit amet"] * 12)
    (tmp_path / "doc.pdf").write_bytes(make_pdf(long_page, "tail page"))
    opt_ctx = ToolContext(
        workspace=tmp_path, options={"pdf2md": {"max_chars": 200}}
    )
    out = await pdf2md(Pdf2MdArgs(path="doc.pdf"), opt_ctx)
    assert "[truncated at 200 characters" in out


async def test_pdf2md_tool_options_reject_over_cap(tmp_path):
    from lingcore.media_types import FALLBACK_TEXT_MAX_CHARS
    from tests.test_modality import make_pdf

    (tmp_path / "doc.pdf").write_bytes(make_pdf("x"))
    opt_ctx = ToolContext(
        workspace=tmp_path,
        options={"pdf2md": {"max_chars": FALLBACK_TEXT_MAX_CHARS + 1}},
    )
    with pytest.raises(ToolError, match="between 200"):
        await pdf2md(Pdf2MdArgs(path="doc.pdf"), opt_ctx)


async def test_pdf2md_without_pymupdf_names_the_extra(ctx, monkeypatch):
    import lingcore.modality as modality_mod
    from lingcore.modality import PDF_INSTALL_HINT
    from tests.test_modality import make_pdf

    (ctx.workspace / "doc.pdf").write_bytes(make_pdf("x"))

    def boom():
        raise ToolError(PDF_INSTALL_HINT)

    monkeypatch.setattr(modality_mod, "_import_pymupdf", boom)
    with pytest.raises(ToolError, match=r"lingcore\[pdf\]"):
        await pdf2md(Pdf2MdArgs(path="doc.pdf"), ctx)


async def test_write_creates_nested(ctx):
    msg = await write_file(WriteArgs(path="x/y/z.txt", content="data"), ctx)
    assert "wrote" in msg
    assert (ctx.workspace / "x" / "y" / "z.txt").read_text() == "data"


async def test_write_rejects_escape(ctx):
    with pytest.raises(ToolError):
        await write_file(WriteArgs(path="../evil.txt", content="x"), ctx)


async def test_edit_unique(ctx):
    await edit_file(EditArgs(path="a.txt", old="world", new="there"), ctx)
    assert (ctx.workspace / "a.txt").read_text() == "hello there"


async def test_edit_not_found(ctx):
    with pytest.raises(ToolError, match="not found"):
        await edit_file(EditArgs(path="a.txt", old="zzz", new="q"), ctx)


async def test_edit_ambiguous(ctx):
    (ctx.workspace / "dup.txt").write_text("x x x", encoding="utf-8")
    with pytest.raises(ToolError, match="occurs 3 times"):
        await edit_file(EditArgs(path="dup.txt", old="x", new="y"), ctx)


async def test_list_dir(ctx):
    out = await list_dir(ListArgs(path="."), ctx)
    assert "a.txt" in out
    assert "sub/" in out


async def test_search(ctx):
    out = await search(SearchArgs(query="return"), ctx)
    assert "sub/b.py:2:" in out


async def test_search_no_match(ctx):
    assert await search(SearchArgs(query="zzz-nope"), ctx) == "(no matches)"


async def test_search_rejects_parent_glob(ctx, tmp_path):
    outside = tmp_path.parent / "secret-search.txt"
    outside.write_text("SECRET_SEARCH", encoding="utf-8")
    with pytest.raises(ToolError, match="glob escapes workspace"):
        await search(
            SearchArgs(query="SECRET_SEARCH", glob="../secret-search.txt"),
            ctx,
        )


async def test_search_skips_symlink_escape(ctx, tmp_path):
    outside = tmp_path.parent / "search-secret.txt"
    outside.write_text("SECRET_SEARCH", encoding="utf-8")
    link = ctx.workspace / "search-link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported on this platform")

    assert await search(SearchArgs(query="SECRET_SEARCH"), ctx) == "(no matches)"


# --- lean / deterministic output (cache-friendliness) ---------------------


async def test_list_dir_caps_entries(ctx):
    for i in range(10):
        (ctx.workspace / f"f{i}.txt").write_text("x", encoding="utf-8")
    ctx.options["list_dir"] = {"max_entries": 4}
    out = await list_dir(ListArgs(path="."), ctx)
    assert "more entries)" in out
    assert len([ln for ln in out.splitlines() if not ln.startswith("…")]) == 4


async def test_search_results_are_sorted(ctx):
    for name in ("z.txt", "a2.txt", "m.txt"):
        (ctx.workspace / name).write_text("NEEDLE\n", encoding="utf-8")
    out = await search(SearchArgs(query="NEEDLE"), ctx)
    paths = [line.split(":")[0] for line in out.splitlines()]
    assert paths == sorted(paths)


async def test_search_skips_runtime_dir(ctx):
    rt = ctx.workspace / ".lingcore" / "tool-output"
    rt.mkdir(parents=True)
    (rt / "shell-abc.txt").write_text("NEEDLE_RT\n", encoding="utf-8")
    assert await search(SearchArgs(query="NEEDLE_RT"), ctx) == "(no matches)"


def test_offload_inline_below_threshold(ctx):
    assert offload_text(ctx, source="x", text="small", threshold=1000) == "small"


async def test_offload_writes_file_readable_via_read_file(ctx):
    big = "\n".join(f"row{i}" for i in range(1, 501))
    out = offload_text(ctx, source="shell", text=big, threshold=100)
    assert "full output" in out and ".lingcore/tool-output/shell-" in out
    rel = out.split("→")[1].split(";")[0].strip()
    content = await read_file(ReadArgs(path=rel, limit=3), ctx)
    assert "1\trow1" in content


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


# --- contract / registry --------------------------------------------------


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
