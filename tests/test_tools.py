"""Tools tests: core behavior."""

from __future__ import annotations

import pytest

from lingcore.errors import ToolError
from lingcore.paths import ConfinedDirectory, confined_directory
from lingcore.tools import ToolContext
from lingcore.tools.builtin.fs import (
    EditArgs,
    ListArgs,
    ReadArgs,
    WriteArgs,
    _resolve,
    edit_file,
    list_dir,
    read_file,
    write_file,
)
from lingcore.tools.builtin.pdf import Pdf2MdArgs, pdf2md
from tests.support.tools import ctx as ctx


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


async def test_write_anchors_parent_against_swap(ctx, tmp_path, monkeypatch):
    import shutil

    parent = ctx.workspace / "nested"
    moved = tmp_path.parent / f"{tmp_path.name}-nested-held"
    real_open = ConfinedDirectory.open_regular
    swapped = False

    def swap_then_open(self, name, mode="rb", *, permissions=0o644):
        nonlocal swapped
        if mode == "wb" and not swapped:
            swapped = True
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        return real_open(self, name, mode, permissions=permissions)

    monkeypatch.setattr(ConfinedDirectory, "open_regular", swap_then_open)
    try:
        with pytest.raises(ToolError):
            await write_file(WriteArgs(path="nested/evil.txt", content="x"), ctx)
        assert not (moved / "evil.txt").exists()
    finally:
        shutil.rmtree(moved, ignore_errors=True)


async def test_edit_lf_text_matches_crlf_file_and_keeps_endings(ctx):
    path = ctx.workspace / "crlf.txt"
    path.write_bytes(b"first\r\nsecond\r\nthird\r\n")
    await edit_file(EditArgs(path="crlf.txt", old="first\nsecond", new="one\ntwo"), ctx)
    assert path.read_bytes() == b"one\r\ntwo\r\nthird\r\n"


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
