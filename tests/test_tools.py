"""Tests for the tool contract and fs tools (M2).

The path-escape suite is security-critical and intentionally exhaustive.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from lingcore.errors import ConfigError, ToolError
from lingcore.paths import ConfinedDirectory, confined_directory
from lingcore.tools import ToolContext, ToolRegistry, tool
from lingcore.tools.builtin._offload import offload_text
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


def test_confined_directory_enumerates_repeatedly_and_opens_child(ctx):
    with confined_directory(ctx.workspace) as directory:
        first = list(directory.iter_entries())
        second = list(directory.iter_entries())
        with directory.subdirectory("sub") as child:
            child_names = {name for name, *_ in child.iter_entries()}

    assert first == second
    assert {name for name, *_ in first} == {"a.txt", "sub"}
    assert child_names == {"b.py"}


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
    opt_ctx = ToolContext(workspace=tmp_path, options={"pdf2md": {"max_chars": 200}})
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
    out = await search(SearchArgs(query="zzz-nope"), ctx)
    assert out.startswith("(no matches)\n")
    assert "scanned 2 files, 2 dirs" in out


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

    out = await search(SearchArgs(query="SECRET_SEARCH"), ctx)
    assert out.startswith("(no matches)\n")
    assert "skipped 1 symlink" in out


async def test_search_regex_and_invalid_regex(ctx):
    (ctx.workspace / "regex.txt").write_text(
        "ticket ABC-123\nticket ABC-nope\n", encoding="utf-8"
    )

    out = await search(SearchArgs(query=r"ABC-\d+", regex=True), ctx)

    assert "regex.txt:1: ticket ABC-123" in out
    assert "regex.txt:2:" not in out
    with pytest.raises(ToolError, match="invalid search regex"):
        await search(SearchArgs(query="[", regex=True), ctx)


async def test_search_ignore_case(ctx):
    (ctx.workspace / "case.txt").write_text("MixedCase\n", encoding="utf-8")

    assert "case.txt:1:" in await search(
        SearchArgs(query="mixedcase", ignore_case=True), ctx
    )
    assert (await search(SearchArgs(query="mixedcase"), ctx)).startswith("(no matches)")


async def test_search_matches_full_lines_beyond_render_cap(ctx):
    (ctx.workspace / "long.txt").write_text(
        "x" * 300 + " NEEDLE tail\n", encoding="utf-8"
    )
    (ctx.workspace / "anchor.txt").write_text("y" * 300 + "END\n", encoding="utf-8")

    content = await search(SearchArgs(query="NEEDLE"), ctx)
    files = await search(SearchArgs(mode="files", query="NEEDLE"), ctx)
    false_anchor = await search(SearchArgs(query=r"y$", regex=True), ctx)
    true_anchor = await search(SearchArgs(query=r"END$", regex=True), ctx)

    assert "long.txt:1:" in content
    assert files.splitlines()[0] == "long.txt"
    assert false_anchor.startswith("(no matches)")
    assert "anchor.txt:1:" in true_anchor


async def test_search_rejects_empty_query(ctx):
    with pytest.raises(ToolError, match="must not be empty"):
        await search(SearchArgs(query=""), ctx)


async def test_search_path_scope_and_path_errors(ctx):
    (ctx.workspace / "outside.py").write_text("return outside\n", encoding="utf-8")

    out = await search(SearchArgs(query="return", path="sub"), ctx)

    assert "sub/b.py:2:" in out
    assert "outside.py" not in out
    with pytest.raises(ToolError, match="escapes workspace"):
        await search(SearchArgs(query="x", path="../outside"), ctx)
    with pytest.raises(ToolError, match="not a directory"):
        await search(SearchArgs(query="x", path="a.txt"), ctx)


async def test_search_basename_and_legacy_globs(ctx):
    (ctx.workspace / "root.py").write_text("GLOB_HIT\n", encoding="utf-8")
    deep = ctx.workspace / "nested" / "deeper"
    deep.mkdir(parents=True)
    (deep / "child.py").write_text("GLOB_HIT\n", encoding="utf-8")
    (deep / "child.txt").write_text("GLOB_HIT\n", encoding="utf-8")

    basename = await search(SearchArgs(query="GLOB_HIT", glob="*.py"), ctx)
    legacy = await search(SearchArgs(query="GLOB_HIT", glob="**/*.py"), ctx)

    for out in (basename, legacy):
        assert "root.py:1:" in out
        assert "nested/deeper/child.py:1:" in out
        assert "child.txt" not in out


async def test_search_pruning_and_exclude_override(ctx):
    for dirname in (".git", "node_modules"):
        directory = ctx.workspace / dirname
        directory.mkdir()
        (directory / "secret.txt").write_text("PRUNED_HIT\n", encoding="utf-8")

    default = await search(SearchArgs(query="PRUNED_HIT"), ctx)

    assert default.startswith("(no matches)")
    assert "pruned .git, node_modules" in default
    ctx.options["search"] = {"exclude_dirs": []}
    overridden = await search(SearchArgs(query="PRUNED_HIT"), ctx)
    assert ".git/secret.txt:1:" in overridden
    assert "node_modules/secret.txt:1:" in overridden


async def test_search_does_not_follow_symlinked_dirs_or_files(ctx):
    real = ctx.workspace / "real"
    real.mkdir()
    (real / "target.txt").write_text("LINK_HIT\n", encoding="utf-8")
    try:
        (ctx.workspace / "linked-dir").symlink_to(real, target_is_directory=True)
        (ctx.workspace / "linked-file.txt").symlink_to(real / "target.txt")
    except OSError:
        pytest.skip("symlinks not supported on this platform")

    out = await search(SearchArgs(query="LINK_HIT"), ctx)

    assert "real/target.txt:1:" in out
    assert "linked-dir/target.txt" not in out
    assert "linked-file.txt:" not in out
    assert "skipped 2 symlinks" in out

    filtered = await search(SearchArgs(query="LINK_HIT", glob="*.py"), ctx)
    assert "symlink" not in filtered.splitlines()[-1]


async def test_search_per_file_match_cap(ctx):
    (ctx.workspace / "noisy.txt").write_text("\n".join(["NOISE"] * 4), encoding="utf-8")
    ctx.options["search"] = {"max_hits_per_file": 2}

    out = await search(SearchArgs(query="NOISE"), ctx)

    assert (
        len(
            [
                line
                for line in out.splitlines()
                if line.startswith("noisy.txt:") and "more matches" not in line
            ]
        )
        == 2
    )
    assert "noisy.txt: (+2 more matches in this file)" in out


async def test_search_global_match_cap(ctx):
    for i in range(3):
        (ctx.workspace / f"cap{i}.txt").write_text("CAP_HIT\n", encoding="utf-8")
    ctx.options["search"] = {"max_hits": 2}

    out = await search(SearchArgs(query="CAP_HIT"), ctx)

    assert len([line for line in out.splitlines() if ":1:" in line]) == 2
    assert "hit the 2-match cap" in out


async def test_search_file_scan_cap(ctx):
    ctx.options["search"] = {"max_files_scanned": 1}

    out = await search(SearchArgs(query="not-present"), ctx)

    assert "scanned 1 file" in out
    assert "hit the 1-file scan cap" in out


async def test_search_time_budget_returns_partial_result(ctx, monkeypatch):
    from lingcore.tools.builtin import fs as fs_module

    real_read = ConfinedDirectory.read_regular_with_stat

    def slow_read(self, name, *, max_bytes):
        result = real_read(self, name, max_bytes=max_bytes)
        time.sleep(0.005)
        return result

    monkeypatch.setattr(ConfinedDirectory, "read_regular_with_stat", slow_read)
    ctx.options["search"] = {"time_budget_ms": 1}

    out = await fs_module.search(SearchArgs(query="hello"), ctx)

    assert out.startswith("(no matches)")
    assert "stopped after the 1 ms budget" in out


@pytest.mark.parametrize(
    "mode,same_file", [("content", False), ("files", False), ("content", True)]
)
def test_search_interrupts_expensive_regex_and_keeps_prior_matches(
    tmp_path, mode, same_file
):
    from lingcore.tool_options import parse_search_options
    from lingcore.tools.builtin.fs import _search_sync

    expensive = "a" * 30 + "!\n"
    (tmp_path / "a.txt").write_text(
        "aaaa\n" + (expensive if same_file else ""), encoding="utf-8"
    )
    if not same_file:
        (tmp_path / "b.txt").write_text(expensive, encoding="utf-8")
    local_ctx = ToolContext(workspace=tmp_path)
    start = time.monotonic()

    out = _search_sync(
        SearchArgs(query=r"^(a|aa)+$", regex=True, mode=mode),
        local_ctx,
        parse_search_options({"time_budget_ms": 20}),
    )

    assert "stopped after the 20 ms budget" in out
    assert "1 match in 1 file" in out
    if mode == "content":
        assert "a.txt:1: aaaa" in out
    else:
        assert out.startswith("a.txt\n")
    assert time.monotonic() - start < 1.0


async def test_search_counts_oversized_binary_and_non_utf8_skips(tmp_path):
    (tmp_path / "large.txt").write_bytes(b"12345")
    (tmp_path / "binary.txt").write_bytes(b"x\x00")
    (tmp_path / "invalid.txt").write_bytes(b"\xff")
    (tmp_path / "text.txt").write_text("ok", encoding="utf-8")
    local_ctx = ToolContext(
        workspace=tmp_path, options={"search": {"max_file_bytes": 4}}
    )

    out = await search(SearchArgs(query="missing"), local_ctx)

    assert "skipped 1 too large, 1 binary, 1 non-UTF-8" in out


async def test_search_skips_fifo_without_opening_it(ctx):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unsupported on this platform")
    try:
        os.mkfifo(ctx.workspace / "pipe.txt")
    except OSError:
        pytest.skip("FIFOs are unsupported on this filesystem")

    out = await search(SearchArgs(query="anything"), ctx)

    assert "skipped 1 non-regular" in out


async def test_search_files_mode_with_and_without_query(ctx):
    by_name = await search(SearchArgs(mode="files", glob="*.py"), ctx)
    by_content = await search(SearchArgs(mode="files", query="return"), ctx)

    assert by_name.splitlines()[0] == "sub/b.py"
    assert by_content.splitlines()[0] == "sub/b.py"
    assert "sub/b.py:" not in by_content


async def test_search_content_mode_requires_query(ctx):
    with pytest.raises(ToolError, match="requires a query"):
        await search(SearchArgs(), ctx)


async def test_search_context_rendering(ctx):
    lines = [f"line {i}" for i in range(1, 11)]
    lines[0] = "    indented context"
    lines[1] = "CONTEXT_HIT first"
    lines[8] = "CONTEXT_HIT second"
    (ctx.workspace / "context.txt").write_text("\n".join(lines), encoding="utf-8")

    out = await search(SearchArgs(query="CONTEXT_HIT", context=2), ctx)

    assert "context.txt-1-     indented context" in out
    assert "context.txt:2: CONTEXT_HIT first" in out
    assert "\n--\n" in out
    assert "context.txt:9: CONTEXT_HIT second" in out
    assert "context.txt-10- line 10" in out


async def test_search_repeat_calls_are_byte_identical(ctx):
    args = SearchArgs(query="return", context=1)
    assert await search(args, ctx) == await search(args, ctx)


async def test_search_offloads_large_output(ctx):
    (ctx.workspace / "many.txt").write_text(
        "\n".join(f"OFFLOAD {i}" for i in range(301)), encoding="utf-8"
    )
    ctx.options["search"] = {
        "max_hits": 300,
        "max_hits_per_file": 301,
        "offload_over_chars": 50,
    }

    out = await search(SearchArgs(query="OFFLOAD"), ctx)

    assert "full output" in out and ".lingcore/tool-output/search-" in out
    rel = out.split("→")[1].split(";")[0].strip()
    staged = (ctx.workspace / rel).read_text(encoding="utf-8")
    assert "many.txt:300: OFFLOAD 299" in staged
    assert not staged.splitlines()[-1].startswith("(300 matches in 1 file;")
    assert out.splitlines()[-1].startswith("(300 matches in 1 file;")
    assert "hit the 300-match cap" in out.splitlines()[-1]


@pytest.mark.parametrize(
    "bad_options",
    [
        {"max_hits": "many"},
        {"exclude_dirs": ["ok", ""]},
        {"max_hitz": 10},
    ],
)
async def test_search_rejects_bad_options(ctx, bad_options):
    ctx.options["search"] = bad_options
    with pytest.raises(ConfigError, match=r"tool_options\.search"):
        await search(SearchArgs(query="x"), ctx)


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
    paths = [line.split(":")[0] for line in out.splitlines()[:-1]]
    assert paths == sorted(paths)


async def test_search_skips_runtime_dir(ctx):
    rt = ctx.workspace / ".lingcore" / "tool-output"
    rt.mkdir(parents=True)
    (rt / "shell-abc.txt").write_text("NEEDLE_RT\n", encoding="utf-8")
    assert (await search(SearchArgs(query="NEEDLE_RT"), ctx)).startswith("(no matches)")


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
