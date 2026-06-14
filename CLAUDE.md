# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What LingCore is

A config-driven async agent framework. The same runtime becomes a different
agent type (coding, role-play, teaching, psych consultant) purely by loading a
different profile YAML — **adding a new agent type should be a config file, not
a code change.** That principle is the design's whole point; preserve it.

## Commands

```bash
uv sync                       # install deps (incl. dev group)
uv run pytest -q              # run the full suite (369 tests, <10s when tiktoken is cached)
uv run pytest tests/test_config.py -q         # one file
uv run pytest tests/test_config.py::test_x -q # one test
uv run lingcore                               # launch the coding agent over CLI
uv run lingcore --profile path/to/p.yaml      # launch a specific profile
uv run lingcore --workspace /path/to/dir      # override workspace from the command line
uv run lingcore --list-sessions               # list stored sessions for the profile, exit
uv run lingcore -c                            # resume the most recent session
uv run lingcore --resume <id-prefix>          # resume a session by unique id prefix
uv run lingcore --no-session                  # run without persisting this session
```

There is no separate lint/format step configured. Match the existing style.

## Architecture (where things live)

Dependency flows bottom-up; `message.py` depends only on the stdlib-only
`media_types.py`, the loop depends on everything below it, frontends depend
only on `Agent` + events.

- `lingcore/media_types.py` — stdlib-only attachment validation rules (extension+magic-byte allowlist, size caps, name sanitization) shared by the message model; deliberately imports nothing from the rest of LingCore.
- `lingcore/message.py` — `Message`/`ToolCall`/`ToolResult`/`Attachment`/`UserInput`/`Conversation`. `Message.to_openai()` is the **only** place that knows the chat-completions wire shape — including the modality-narrowing render (`attachment_modalities=` parameter, see invariant 15).
- `lingcore/media.py` — attachment construction helpers for tools/frontends (`attachment_from_path`/`attachment_from_wire`/`classify_bytes`); converts validation failures to `ToolError`. **Any file attaches**: `classify_bytes` sorts bytes into the four `Attachment` kinds — native `image`/`file` (extension+magic agree), `text` (NUL-free UTF-8), or `binary` (everything else).
- `lingcore/ingest.py` — `ingest_attachments`: the agent-side step that copies every user attachment into `<workspace>/attachments/` (collision-safe), inlines a `text` file's content / writes a `binary` pointer note as `fallback_text`, and returns note lines announcing each path. **Never raises** (a failed copy degrades to a note); image/PDF fallbacks stay with `MediaAdapter`. Called from `Agent.run` via `asyncio.to_thread` (blocking I/O) before `MediaAdapter.prepare`.
- `lingcore/modality.py` — `MediaAdapter` + `extract_pdf_markdown`: text fallbacks for the **convertible** kinds (`image`/`file`) the model doesn't accept (`llm.modalities` registration + `media_fallback` config); `text`/`binary` are skipped (ingest owns their fallbacks). Takes a duck-typed vision client; never imports `llm.py`/the SDK. pymupdf is loaded lazily (optional `lingcore[pdf]` extra).
- `lingcore/llm.py` — `LLMClient`, the single seam over the OpenAI SDK. All streaming quirks (index-keyed tool-call fragment reassembly, partial-JSON args) live here. Transient-failure retry is **two-tier**: *opening* the stream is delegated to the SDK's header-aware policy (honors `Retry-After`/`retry-after-ms` and `x-should-retry`; retries 408/409/429/5xx + connection/timeout), bounded by `max_retries` and a per-attempt `timeout` from config. Once the stream is yielding the SDK never retries, so the client *classifies* instead: every failure raises a typed `LLMStreamError` (`lingcore/errors.py`) — `retryable=True` for mid-stream interruption/stall/truncation (incl. a stream ending without a `finish_reason`), `retryable=False` once the SDK's open budget is spent or the request is invalid. The *loop* owns recovery (`llm.stream_retries`, see invariant 5); tokens are never silently replayed — a retry discards the partial turn and announces itself via a `StreamRetry` event.
- `lingcore/events.py` — `AgentEvent` union the loop emits (incl. `SkillActivated`).
- `lingcore/agent.py` — the async run loop + `Agent.from_profile`. The core.
- `lingcore/composer.py` — `PromptComposer` protocol + `ComposeContext` + `StaticComposer`/`LayeredComposer`. The per-turn system-prompt assembly seam.
- `lingcore/config.py` — `AgentProfile` (+ `skills:` static declaration) + nested pydantic cfgs; YAML loading with `${ENV}` expansion; `_source_dir` resolution.
- `lingcore/memory.py` — `ShortTermMemory` protocol + `WindowMemory`.
- `lingcore/sessions.py` — `SessionStore` (SQLite, one db per profile dir) + `SessionMemory` (a `ShortTermMemory` wrapper that records every message) + `trim_dangling`/`attach_session`/`open_store`. Session history + resume live here; composition roots open the store, `Agent.from_profile(session_store=..., session_id=...)` wires it.
- `lingcore/skills.py` — `Skill`/`SkillState`/`load_skills`/`load_skill_tools`. The skill permission model + the loader that imports a skill's shipped tool code.
- `lingcore/guardrails.py` — `Guardrail` protocol + `NoopGuardrail`.
- `lingcore/tools/__init__.py` — `Tool`/`@tool`/`ToolRegistry`/`ToolContext` (the plugin contract).
- `lingcore/tools/builtin/{fs,patch,shell,web,memory,knowledge,skill,pdf}.py` — builtin tools (`pdf.py` = the `pdf2md` extractor, optional-dependency-gated).
- `lingcore/io/{base,cli}.py` — frontend protocol + `run_session` driver + Rich CLI.
- `lingcore/__main__.py` — the CLI entrypoint / composition root.
- `profiles/coding/` — the default profile dir (`config.yaml` + world/role/workflow `.md`). Bundled profiles live at the **repo root, outside the package tree**, so their `sessions.db`/`memory.md`/default `workspace/` stay writable (all gitignored).
- `profiles/coding_ollama/` — Ollama/vLLM variant (keyless local server).
- `profiles/daily/` — general-purpose assistant (research/notes/memory, no shell).
- `profiles/teaching/` — teaching assistant; statically engages the `canvas` skill (`skills: [canvas]`).
- `lingcore/skills/` — bundled skills (each a dir with `skill.md` frontmatter + body). A skill may also ship a Python tool module (`module:` + `provides:`), e.g. `lingcore/skills/canvas/canvas_tools.py`.

## Invariants — do not break these when extending

1. **The loop never imports `openai` directly.** `agent.py` talks only to an `LLMClient`-shaped object (duck-typed `.stream()`). This seam is what lets a different backend (or LangGraph) swap in later. Keep OpenAI specifics inside `llm.py`.
2. **Tools get context as a parameter (`ToolContext`), never via globals** — so concurrent sessions with different workspaces stay isolated.
3. **Frontends consume `AgentEvent`, never LLM internals.** `run_session` is frontend-agnostic; new frontends implement the `Frontend` protocol and change nothing else.
4. **Secrets never live in profile YAML.** `llm.api_key_env` only *names* an env var. Profiles use `extra="forbid"` so a typo is a loud `ConfigError`.
5. **Tool errors never crash the loop** — a raising tool becomes `ToolResult(ok=False)` fed back to the model. Hard `max_iters` cap emits an `Error` event rather than raising. **LLM failures don't crash it either**: a retryable `LLMStreamError` makes the loop discard the partial turn (it was never committed to memory/session), emit `StreamRetry` so frontends can void any rendered text, back off with jitter, and re-request — up to `llm.stream_retries` times per request. Terminal failures (non-retryable, budget exhausted, or a foreign exception from a duck-typed client) end the turn with an `Error` event; `run_session` never sees an exception from a model failure.
6. **`WindowMemory` trimming is block-aware** — an assistant `tool_calls` message and its `tool` results are atomic; never orphan a tool result from its call (OpenAI rejects it). Always keep ≥1 block.
7. **`run_shell` is arbitrary code execution; the workspace is NOT a sandbox.** fs tools are confined via `_resolve` + `is_relative_to` (tested against `..`/absolute/symlink escapes), but shell can escape. Mitigations: cwd-scoped, timeout-with-process-group-kill, output truncation, strict allowlist matching, and `confirm` gate. Real isolation is the deferred next step; `shell.py` is the plug point.
8. **An unset `workspace` defaults to `<profile dir>/workspace/`** (auto-created at assembly; gitignored for bundled profiles) — a bare launch never adopts whatever directory the user happened to be in. **An explicit relative `workspace` resolves against the user's CWD** (where they launched lingcore), not the profile's directory. Bundled profiles write `workspace: ${LINGCORE_WORKSPACE:-}`; the blank expansion is normalized to *unset*, so the env override stays available without changing the default.
9. **Network fetch is public-web only by default.** `fetch_url` rejects embedded credentials and non-http(s) schemes, resolves the host and rejects any answer that maps to a loopback/link-local/private/reserved address (covering alternate IP encodings — decimal/hex/octal), then **pins the connection to the vetted IP** while keeping the Host header and TLS SNI/cert verification on the hostname — so DNS can't be rebound between the check and the connection. Re-applied to every redirect hop; DNS resolution and downloaded body size are both bounded. Opt into local targets with `tool_options.fetch_url.allow_private_hosts: true`.
10. **The system prompt is composed per loop iteration, not frozen.** `agent.py` calls `composer.compose(ctx)` at the top of *every* iteration with an immutable `ComposeContext` (never mutable hidden state). This is why memory writes and skill activation take effect on the *next* model request. A `StaticComposer` (no layers/memory/skills) is the zero-overhead default; `LayeredComposer` reads memory.md fresh each call. Composer content is **read-only context — it can never unlock a tool.**
11. **Skills never grant tools beyond the profile.** Effective tools when a skill is active = `profile.tools ∩ skill.requested_tools`; the profile's `tools:` list is a hard ceiling. `activate_skill` mutates only the shared `SkillState.active`; the base `ToolRegistry` is **never** mutated. The ceiling is enforced twice: in `_effective_tool_schemas` (what's advertised) and in `_resolve_tool` (what's dispatchable). High-risk tools (`run_shell`/`write_file`/`patch_file`/`edit_file`) require `confirm` before activation. `activate_skill` only *offers* skills the profile can actually use (≥1 requested tool authorized, or instruction-only) — it never advertises a skill whose effective tool set would be empty.
12. **`memory` writes are profile-scoped and confined.** The memory file resolves against `profile_dir` (not the workspace); relative paths that escape it raise `ConfigError`, absolute paths require `allow_absolute_path: true`, and writes into the installed package tree are refused. `max_bytes` is checked on the *final* file content. Strict keys: `remember` fails if the key exists; `modify`/`forget` fail if it doesn't.
13. **Skills may ship their own tool code; the `tools:` ceiling still governs.** A skill dir may include a Python module (frontmatter `module:` + `provides:`) whose `@tool` functions are imported into the global `REGISTRY` by `load_skill_tools`, run in `from_profile` **before** `REGISTRY.subset(profile.tools)`. *Registration ≠ authorization*: a skill-provided tool is reachable only if its name is in `profile.tools` (else it fails `subset()`/`_resolve_tool` exactly like a typo — invariant 11 still binds shipped code). A profile loads a skill's code only by *engaging* it — naming it in `skills:` (static; instructions injected as an always-on prompt layer) or enabling `activate_skill` (dynamic). Importing a skill module is code execution; a broken/mismatched module, an undeclared registration, or a name collision raises `ConfigError` at load — never silently. **Loading is atomic**: a module that fails any check is rolled back (the global catalog and its `sys.modules` entry are restored), and a module may **never overwrite an existing tool** — a name it registers but doesn't declare in `provides`, or that collides with an incumbent, is refused (object-identity check, not just a name diff). The synthetic module identity is keyed by the *resolved file path*, so a profile-local skill genuinely *shadows* a bundled one of the same name rather than aliasing its already-imported code. The base subset is never mutated at runtime. (First built skill: `lingcore/skills/canvas`, used by the `teaching` profile.)
14. **`sessions.db` is profile-scoped and confined exactly like memory.** The session DB resolves against the profile dir; a relative `sessions.path` that escapes it raises `ConfigError`, an absolute path requires `sessions.allow_absolute_path: true`, and a path inside the installed package tree **gracefully disables** persistence with a one-line notice — never an error, since sessions default to enabled. Bundled profiles live at the repo root (`profiles/`) precisely so this rule doesn't bite them; LingChat surfaces the notice in its sidebar (the `notice` field of `GET /api/sessions`) instead of hiding the feature. Session rows are **lazy**: the row is created in the same transaction as the first message, so an opened-but-silent session (instant `/exit`, browser reconnect loop) leaves no empty rows behind. Loading for resume is block-aware: `trim_dangling` drops any assistant message whose `tool_calls` lack complete tool results (anywhere in the list — a crashed turn can leave one mid-list once later turns append after it) and orphaned tool results, but keeps a trailing lone user message (valid prefix, real input). Deliberate v1 scope: active skills and per-session "allow always" shell patterns are **not** restored on resume (conservative default), and attach-exclusivity (LingChat's one-tab-per-session rule) is per-process only.
15. **Attachment modalities are config-registered; degradation never crashes the loop.** Attachments have four kinds: `image`/`file` can ride **natively**; `text` always inlines its UTF-8 content; `binary` is an opaque file the model reaches only through workspace tools. `llm.modalities` (typed to the native subset — declaring `text`/`binary` is a loud `ConfigError`) declares which native kinds the model accepts (default: both `image` and `file`). **Every user attachment is copied into `<workspace>/attachments/` at ingest** (`ingest_attachments`, collision-safe) and its path announced in the user message text, so workspace-confined tools can reach it. Fallbacks are split by who can compute them: `ingest` sets `text` (inlined content, capped at `TEXT_INLINE_MAX_CHARS`) and `binary` (a workspace pointer note) fallbacks; `MediaAdapter.prepare` sets `image`/`file` ones when the model lacks that modality (PDF → extracted markdown, image → a `media_fallback.image` vision description) and **skips** `text`/`binary`. Both run **once at ingest**, before the message is committed, and **never raise** — failures (a missing pymupdf, a failed copy) become diagnostic notes the model can read. Originals are never stripped: sessions/LingChat keep full payloads, so raising `modalities` later restores native delivery, and `model_copy` keeps validators out of the hot path. The native-vs-text decision happens per render in `Message.to_openai(attachment_modalities=...)` (the `LLMClient` passes its set; `text`/`binary` are never in it); when **no native part remains the content collapses to a plain string** — text-only servers may reject a parts array. The describe call goes through the same duck-typed `stream` seam as the main model and always renders natively (config validation rejects a vision model whose own `modalities` exclude `image`).

## Conventions

- Python ≥3.11, `from __future__ import annotations` at the top of every module, full type hints, pydantic v2.
- All I/O-facing code is `async`. Tools are `async def`.
- New tests go in `tests/`; drive the loop with `tests/fakes.py:FakeLLMClient` (scripted, no network). `asyncio_mode = "auto"` is set, so no `@pytest.mark.asyncio` needed.
- Dependency versions in `pyproject.toml` are pinned exactly. Don't loosen them without reason.

## Known gaps (post-MVP)

tiktoken downloads its encoding file on first use → `WindowMemory` blocks/fails
offline (`--list-sessions` deliberately never builds an agent, so it stays
offline-safe). Real guardrail impls, summarize-memory, entry_points third-party
tools, and Discord frontends are not built yet. Session resume does not restore
active skills or per-session shell allowlists (v1 scope; see invariant 14), and
nothing prevents a CLI and a LingChat process from attaching the same session
concurrently (per-process registry only — a cross-process lease is future
work). The `knowledge` tool ships
the grep backend only; the embedding `index`/`hybrid` backends and
`auto_retrieve` prompt injection are stubbed (raise a clear `ConfigError`
pending numpy + a SQLite vec store).
