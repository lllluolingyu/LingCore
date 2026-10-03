"""Tests for WindowMemory (M3) — focus on block-aware trimming."""

from __future__ import annotations

import hashlib

import pytest
import tiktoken
import tiktoken.load
import tiktoken.registry

import lingcore.memory as memory_module
from lingcore.memory import WindowMemory
from lingcore.message import Message, ToolCall, ToolResult


def test_render_prepends_system():
    mem = WindowMemory(model="gpt-4o")
    mem.add(Message.user("hi"))
    rendered = mem.render("SYS")
    assert rendered[0].role == "system"
    assert rendered[0].content == "SYS"
    assert rendered[1].content == "hi"


def test_anthropic_thinking_counts_opaque_state_without_duplicate_summary():
    memory = WindowMemory(model="local/offline")
    snapshot = [
        {"type": "thinking", "thinking": "a" * 40, "signature": "s" * 80},
        {"type": "redacted_thinking", "data": "r" * 160},
        {"type": "text", "text": "ok"},
    ]
    message = Message.assistant("ok", anthropic_content=snapshot)
    with_summary = Message.assistant(
        "ok", reasoning_content="a" * 40, anthropic_content=snapshot
    )
    assert memory._tokens(message) > 70
    assert memory._tokens(with_summary) == memory._tokens(message)
    assert memory._tokens(Message.assistant("ok")) < memory._tokens(message)


def test_encoding_is_lazy_and_offline_failure_uses_estimate(monkeypatch):
    calls: list[str] = []

    def unavailable(model: str):
        calls.append(model)
        raise OSError("encoding cache unavailable")

    monkeypatch.setattr(memory_module, "_encoding", unavailable)
    mem = WindowMemory(model="local/offline")
    mem.add(Message.user("hello"))

    assert calls == []  # construction never resolves/downloads a vocabulary
    rendered = mem.render("system")
    assert calls == ["local/offline"]
    assert rendered[-1].content == "hello"
    assert mem._tokens(rendered[-1]) > 0
    assert calls == ["local/offline"]  # failed resolution is not retried


@pytest.mark.parametrize("model", ["gpt-4o", "local/offline"])
@pytest.mark.parametrize("cache_state", ["empty", "disabled", "corrupt"])
def test_cold_encoding_never_downloads(monkeypatch, tmp_path, model, cache_state):
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})
    monkeypatch.setenv(
        "TIKTOKEN_CACHE_DIR", "" if cache_state == "disabled" else str(tmp_path)
    )
    if cache_state == "corrupt":
        for name in ("o200k_base", "cl100k_base"):
            url = (
                f"https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken"
            )
            (tmp_path / hashlib.sha1(url.encode()).hexdigest()).write_bytes(b"bad")

    def forbidden_download(*args, **kwargs):
        pytest.fail("memory counting must never download a tokenizer vocabulary")

    monkeypatch.setattr(tiktoken.load, "read_file", forbidden_download)
    mem = WindowMemory(model=model)
    mem.add(Message.user("hello"))

    assert mem.render("system")[-1].content == "hello"
    assert mem._text_tokens("hello") == 2
    assert mem._text_tokens("你好世界") == 3
    assert mem._text_tokens("") == 0


def _byte_encoding():
    """A real offline tokenizer, with one token per byte and one special token."""
    return tiktoken.Encoding(
        name="test_bytes",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )


def test_cold_encoding_does_not_invoke_constructor(monkeypatch):
    def forbidden_constructor():
        pytest.fail("memory counting must not initialize tokenizer plugins")

    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})
    monkeypatch.setattr(
        tiktoken.registry,
        "ENCODING_CONSTRUCTORS",
        {"o200k_base": forbidden_constructor},
    )

    assert WindowMemory(model="gpt-4o")._text_tokens("hello") == 2


@pytest.mark.parametrize(
    ("model", "encoding_name"),
    [("gpt-4o", "o200k_base"), ("local/offline", "cl100k_base")],
)
def test_memory_reuses_loaded_encoding(monkeypatch, model, encoding_name):
    monkeypatch.setattr(
        tiktoken.registry, "ENCODINGS", {encoding_name: _byte_encoding()}
    )
    mem = WindowMemory(model=model)

    assert mem._text_tokens("hello") == 5
    assert mem._text_tokens("你好世界") == 12


def test_special_token_spelling_is_counted_as_ordinary_text(monkeypatch):
    monkeypatch.setattr(
        tiktoken.registry, "ENCODINGS", {"o200k_base": _byte_encoding()}
    )
    mem = WindowMemory(model="gpt-4o")
    literal = "<|endoftext|>"
    mem.add(Message.user(literal))
    mem.add(
        Message.assistant(
            content=literal,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": literal})],
        )
    )
    mem.add(
        Message.from_tool_result(ToolResult(call_id="c1", name="echo", content=literal))
    )

    assert mem._text_tokens(literal) == 13
    assert [message.content for message in mem.render(literal)] == [literal] * 4


def test_message_count_cap_keeps_recent():
    mem = WindowMemory(max_messages=2, model="gpt-4o")
    for i in range(5):
        mem.add(Message.user(f"m{i}"))
    rendered = mem.render("SYS")
    # system + at most 2 most-recent user blocks
    bodies = [m.content for m in rendered if m.role == "user"]
    assert bodies == ["m3", "m4"]


def test_tool_block_never_orphaned():
    """A tool result must never be kept without its assistant tool_call."""
    mem = WindowMemory(max_messages=2, model="gpt-4o")
    mem.add(Message.user("old"))
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a"})
    mem.add(Message.assistant(content="", tool_calls=[call]))
    mem.add(
        Message.from_tool_result(
            ToolResult(call_id="c1", name="read_file", content="x")
        )
    )

    rendered = mem.render("SYS")
    # If a tool message is present, the message before it must carry tool_calls.
    for i, m in enumerate(rendered):
        if m.role == "tool":
            assert rendered[i - 1].role == "assistant"
            assert rendered[i - 1].tool_calls, "tool result orphaned from its call"


def test_keeps_at_least_one_block_even_over_budget():
    mem = WindowMemory(max_messages=1, max_tokens=1, model="gpt-4o")
    mem.add(Message.user("a very long message that exceeds the tiny token budget"))
    rendered = mem.render("SYS")
    # system + the single block (we never return an empty conversation)
    assert any(m.role == "user" for m in rendered)


def test_attachment_estimates_flat_and_fallback_text():
    import base64

    from lingcore.message import Attachment

    mem = WindowMemory(model="gpt-4o")
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode("ascii")
    plain = Attachment(kind="image", media_type="image/png", data=png)
    converted = plain.model_copy(update={"fallback_text": "x" * 40_000})

    flat = mem._tokens(Message.user("m", attachments=[plain]))
    with_text = mem._tokens(Message.user("m", attachments=[converted]))
    assert flat >= 1_000  # flat per-image estimate
    # A large fallback may be what actually goes on the wire: the estimate
    # must grow with it (40k chars ≈ 10k tokens > the flat 1k).
    assert with_text >= 10_000


# --- prefix-stable (cache-aware) eviction ---------------------------------


def _render_bodies(mem, sys="SYS"):
    return [(m.role, m.content) for m in mem.render(sys) if m.role != "system"]


def test_tool_loop_renders_are_append_only():
    """Within a turn (under the cap) each render extends the previous as a
    prefix — the property that makes the tool-call loop cache-hit."""
    mem = WindowMemory(max_tokens=5_000, model="gpt-4o")  # generous: no eviction
    mem.add(Message.user("do a task"))
    prev = _render_bodies(mem)
    for i in range(6):
        call = ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"f{i}"})
        mem.add(Message.assistant(content="", tool_calls=[call]))
        mem.add(
            Message.from_tool_result(
                ToolResult(call_id=f"c{i}", name="read_file", content=f"contents {i}")
            )
        )
        cur = _render_bodies(mem)
        assert cur[: len(prev)] == prev  # prior request is a prefix of this one
        prev = cur


def _count_prefix_shifts(evict_to_ratio):
    mem = WindowMemory(
        max_messages=1000, max_tokens=80, evict_to_ratio=evict_to_ratio, model="gpt-4o"
    )
    renders = []
    for i in range(30):
        mem.add(Message.user(f"message number {i} with a few words"))
        r = _render_bodies(mem)
        # hard cap is always respected
        assert (
            sum(mem._tokens(m) for m in mem.render("SYS") if m.role != "system") <= 80
        )
        renders.append(r)
    shifts = sum(1 for a, b in zip(renders, renders[1:]) if b[: len(a)] != a)
    return shifts, mem


def test_hysteresis_reduces_prefix_shifts_vs_legacy():
    shifts_half, mem_half = _count_prefix_shifts(0.5)
    shifts_legacy, _ = _count_prefix_shifts(1.0)
    assert mem_half._floor > 0  # eviction genuinely happened
    # Chunked eviction shifts the cached prefix far less than slide-every-render.
    assert shifts_half < shifts_legacy


def test_floor_is_monotonic():
    mem = WindowMemory(
        max_messages=1000, max_tokens=60, evict_to_ratio=0.5, model="gpt-4o"
    )
    for i in range(15):
        mem.add(Message.user(f"msg {i} with padding words here"))
        mem.render("SYS")
    f1 = mem._floor
    assert f1 > 0
    mem.add(Message.user("x"))  # tiny add must never retreat the floor
    mem.render("SYS")
    assert mem._floor >= f1


def test_evicted_messages_are_physically_pruned():
    # Eviction must *release* evicted Message objects, not retain them for the
    # life of the process (bounded memory). Over many turns on a small window,
    # the retained set stays near the window size instead of growing with the
    # whole conversation.
    mem = WindowMemory(max_messages=6, max_tokens=100_000, model="gpt-4o")
    for i in range(200):
        mem.add(Message.user(f"message {i}"))
        mem.render("SYS")
    assert len(mem.messages) <= 6  # not 200
    assert mem.messages[-1].content == "message 199"  # newest kept
    assert mem._floor >= 190  # the rest were counted as evicted
