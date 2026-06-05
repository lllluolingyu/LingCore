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
uv run pytest -q              # run the full suite (68 tests, <1s when tiktoken is cached)
uv run pytest tests/test_config.py -q         # one file
uv run pytest tests/test_config.py::test_x -q # one test
uv run lingcore                               # launch the coding agent over CLI
uv run lingcore --profile path/to/p.yaml      # launch a specific profile
uv run lingcore --workspace /path/to/dir      # override workspace from the command line
```

There is no separate lint/format step configured. Match the existing style.

## Architecture (where things live)

Dependency flows bottom-up; `message.py` depends on nothing, the loop depends
on everything below it, frontends depend only on `Agent` + events.

- `lingcore/message.py` — `Message`/`ToolCall`/`ToolResult`/`Conversation`. `Message.to_openai()` is the **only** place that knows the chat-completions wire shape.
- `lingcore/llm.py` — `LLMClient`, the single seam over the OpenAI SDK. All streaming quirks (index-keyed tool-call fragment reassembly, partial-JSON args) live here.
- `lingcore/events.py` — `AgentEvent` union the loop emits.
- `lingcore/agent.py` — the async run loop + `Agent.from_profile`. The core.
- `lingcore/config.py` — `AgentProfile` + nested pydantic cfgs; YAML loading with `${ENV}` expansion.
- `lingcore/memory.py` — `ShortTermMemory` protocol + `WindowMemory`.
- `lingcore/guardrails.py` — `Guardrail` protocol + `NoopGuardrail`.
- `lingcore/tools/__init__.py` — `Tool`/`@tool`/`ToolRegistry`/`ToolContext` (the plugin contract).
- `lingcore/tools/builtin/{fs,shell}.py` — coding-agent tools.
- `lingcore/io/{base,cli}.py` — frontend protocol + `run_session` driver + Rich CLI.
- `lingcore/__main__.py` — the CLI entrypoint / composition root.
- `lingcore/profiles/coding.yaml` — the default profile (keyed provider).
- `lingcore/profiles/coding_ollama.yaml` — Ollama/vLLM variant (keyless local server).

## Invariants — do not break these when extending

1. **The loop never imports `openai` directly.** `agent.py` talks only to an `LLMClient`-shaped object (duck-typed `.stream()`). This seam is what lets a different backend (or LangGraph) swap in later. Keep OpenAI specifics inside `llm.py`.
2. **Tools get context as a parameter (`ToolContext`), never via globals** — so concurrent sessions with different workspaces stay isolated.
3. **Frontends consume `AgentEvent`, never LLM internals.** `run_session` is frontend-agnostic; new frontends implement the `Frontend` protocol and change nothing else.
4. **Secrets never live in profile YAML.** `llm.api_key_env` only *names* an env var. Profiles use `extra="forbid"` so a typo is a loud `ConfigError`.
5. **Tool errors never crash the loop** — a raising tool becomes `ToolResult(ok=False)` fed back to the model. Hard `max_iters` cap emits an `Error` event rather than raising.
6. **`WindowMemory` trimming is block-aware** — an assistant `tool_calls` message and its `tool` results are atomic; never orphan a tool result from its call (OpenAI rejects it). Always keep ≥1 block.
7. **`run_shell` is arbitrary code execution; the workspace is NOT a sandbox.** fs tools are confined via `_resolve` + `is_relative_to` (tested against `..`/absolute/symlink escapes), but shell can escape. Mitigations: cwd-scoped, timeout-with-process-group-kill, output truncation, `confirm` gate. Real isolation is the deferred next step; `shell.py` is the plug point.
8. **A relative `workspace` resolves against the user's CWD** (where they launched lingcore), not the bundled profile's directory.

## Conventions

- Python ≥3.11, `from __future__ import annotations` at the top of every module, full type hints, pydantic v2.
- All I/O-facing code is `async`. Tools are `async def`.
- New tests go in `tests/`; drive the loop with `tests/fakes.py:FakeLLMClient` (scripted, no network). `asyncio_mode = "auto"` is set, so no `@pytest.mark.asyncio` needed.
- Dependency versions in `pyproject.toml` are pinned exactly. Don't loosen them without reason.

## Known gaps (post-MVP)

tiktoken downloads its encoding file on first use → `WindowMemory` blocks/fails
offline. Real guardrail impls, summarize-memory, entry_points third-party
tools, and web/Discord frontends are not built yet.
