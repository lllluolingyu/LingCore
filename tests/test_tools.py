"""Tests for the tool contract and fs tools (M2).

The path-escape suite is security-critical and intentionally exhaustive.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lingcore.errors import ConfigError, ToolError
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
    read_media,
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


# --- fs tool behaviour ----------------------------------------------------


async def test_read_file(ctx):
    assert await read_file(ReadArgs(path="a.txt"), ctx) == "hello world"


async def test_read_missing(ctx):
    with pytest.raises(ToolError, match="not a file"):
        await read_file(ReadArgs(path="nope.txt"), ctx)


async def test_read_file_rejects_known_media(ctx):
    (ctx.workspace / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nrest")
    with pytest.raises(ToolError, match="read_media"):
        await read_file(ReadArgs(path="pic.png"), ctx)


async def test_read_file_rejects_other_binary(ctx):
    (ctx.workspace / "blob.bin").write_bytes(b"abc\x00def")
    with pytest.raises(ToolError, match="binary file"):
        await read_file(ReadArgs(path="blob.bin"), ctx)


async def test_read_media_image(ctx):
    (ctx.workspace / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nrest")
    out = await read_media(ReadArgs(path="pic.png"), ctx)
    assert out.text.startswith("attached pic.png")
    assert out.attachments[0].media_type == "image/png"


async def test_read_media_pdf(ctx):
    (ctx.workspace / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    out = await read_media(ReadArgs(path="doc.pdf"), ctx)
    assert out.attachments[0].kind == "file"
    assert out.attachments[0].name == "doc.pdf"


async def test_read_media_rejects_escape(ctx):
    with pytest.raises(ToolError, match="escapes workspace"):
        await read_media(ReadArgs(path="../pic.png"), ctx)


async def test_read_media_rejects_unknown_extension(ctx):
    (ctx.workspace / "blob.bin").write_bytes(b"blob")
    with pytest.raises(ToolError, match="unsupported media type"):
        await read_media(ReadArgs(path="blob.bin"), ctx)


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
