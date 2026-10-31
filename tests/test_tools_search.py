"""Tools tests: search."""

from __future__ import annotations

import os
import time

import pytest

from lingcore.errors import ConfigError, ToolError
from lingcore.paths import ConfinedDirectory
from lingcore.tools import ToolContext
from lingcore.tools.builtin.fs import (
    ListArgs,
    SearchArgs,
    list_dir,
    search,
)
from tests.support.tools import ctx as ctx


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
