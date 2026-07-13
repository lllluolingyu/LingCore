# LingCore

A lightweight, config-driven async agent framework.

The same runtime becomes a different kind of agent — coding assistant,
role-play character, teaching helper, psych consultant — purely by loading a
different **profile**. Adding a new agent type is a config file, not new code.

Built directly on the OpenAI SDK (chat-completions + tool calling), so it works
against OpenAI, local servers like **Ollama** and **vLLM**, or any
OpenAI-compatible endpoint by pointing at a different `base_url`.

> Status: MVP. The coding agent runs over a CLI. Role-play / teaching / psych
> profiles are the intended next step on the same runtime.

## Features

- **Profile-driven** — model, endpoint, persona, tool list, workspace, memory,
  and sampling all live in a YAML file. Secrets stay in the environment.
- **Thin async core** — a small, owned agent loop with streaming, parallel tool
  calls, and a hard iteration cap. No heavyweight orchestration framework.
- **Pluggable tools** — a tool is an `async` function plus a pydantic args
  model and a `@tool` decorator. The coding agent ships with file read/write/
  edit, patch, directory listing, search, URL fetch, and a confirmation-gated
  shell.
- **Cache-aware context** — built so the request prefix stays stable and
  provider prompt-caching actually hits. Tool output is kept lean and
  deterministic (`read_file` is paginated + line-numbered; heavy `run_shell` /
  `fetch_url` output is offloaded to a workspace file and read back on demand);
  the conversation window evicts in stable chunks instead of sliding every turn;
  and when it nears full, the oldest history is **compacted** (summarized) rather
  than dropped. An opt-in `llm.send_prompt_cache_key` pins a session's requests
  to the same cache node, and the persistent `memory.md` can likewise
  auto-compact at a size limit instead of refusing writes.
- **Multimodal with graceful degradation** — attach any file via CLI `@path`
  or the web UI; it is copied into the workspace and announced to the model,
  and the agent can also attach a workspace image/PDF with `read_file`. A
  profile registers which modalities its model natively accepts
  (`llm.modalities`); anything else degrades to text — PDFs via markdown
  extraction, images via a description from a configured fallback vision model
  (`media_fallback`), and other text files inlined directly.
- **Session history & resume** — conversations persist to a small SQLite db
  *inside the profile directory* (each profile keeps its own history). Resume
  the latest with `-c`, a specific one with `--resume <id-prefix>`, inspect
  with `--list-sessions`, or opt out with `--no-session` /
  `sessions.enabled: false`.
- **Frontend-agnostic** — the loop emits events; the CLI renders them today, and
  a web or chat frontend can render the same events later without touching core.

## Install

Requires Python ≥3.11. Uses [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url> LingCore
cd LingCore
uv sync
```

PDF text extraction (the `pdf2md` tool and the automatic PDF→text fallback)
needs PyMuPDF, which is AGPL-3.0 while LingCore is Apache-2.0 — so it ships as
an optional extra rather than a base dependency. A dev clone already gets it
via `uv sync` (dev group); a plain install opts in with:

```bash
pip install 'lingcore[pdf]'
```

## Quick start

### Local Ollama (no API key)

```bash
ollama pull qwen2.5-coder:7b      # or any model you have
cd LingCore
uv run lingcore --profile profiles/coding_ollama --workspace /path/to/your/project
```

### Keyed provider (OpenAI, etc.)

No file edits — just environment variables:

```bash
export LINGCORE_BASE_URL=https://api.openai.com/v1
export LINGCORE_MODEL=gpt-4o
export LINGCORE_API_KEY_ENV=OPENAI_API_KEY   # names the var holding your key
export OPENAI_API_KEY=sk-...
uv run lingcore
```

Type a message; the agent streams its reply and shows each tool call. Shell
commands prompt for confirmation before running. Type `/exit` to quit.

By default the agent works in a `workspace/` folder inside the profile
directory (auto-created) — point it at a real project with
`--workspace /path/to/project` or `LINGCORE_WORKSPACE`.

Conversations are saved automatically — per profile — and can be picked up
later:

```bash
uv run lingcore -p my-agent -c                  # resume the most recent session
uv run lingcore -p my-agent --resume 3ca5       # resume by unique id prefix
uv run lingcore -p my-agent --list-sessions     # see what's stored
```

History lands in `<profile>/sessions.db` — delete the file to wipe it. The
bundled profiles live at the repo root (`profiles/`), outside the installed
package, so they keep history too (their db files are gitignored). A profile
inside an installed package can't persist and runs ephemeral with a one-line
notice.

> The first message may pause briefly while `tiktoken` downloads its tokenizer
> data (cached afterward). This step needs network access once.

## Profiles

A profile is a **directory** containing a `config.yaml` and optional Markdown
prompt-layer files. Four are shipped:

- `profiles/coding/` — default profile, targets a keyed provider via env vars.
- `profiles/coding_ollama/` — keyless variant for local Ollama/vLLM.
- `profiles/daily/` — general-purpose assistant (research, notes, persistent memory; no shell).
- `profiles/teaching/` — teaching assistant built on the Canvas skill (courses, due dates, file sync).

```
my-agent/
  config.yaml    # llm, tools, memory, loop, guardrail, sessions
  world.md       # optional — environment / setting context
  role.md        # optional — persona
  workflow.md    # optional — operating method
  memory.md      # auto-created by the memory tool (opt-in)
  sessions.db    # auto-created session history (on by default; sessions.enabled: false to opt out)
  workspace/     # default working dir for the agent's tools (auto-created; workspace: / --workspace overrides)
```

`world.md`, `role.md`, and `workflow.md` are loaded automatically if present and
composed in that order to form the system prompt. `config.yaml` may also set
`persona.system_prompt` as an inline fallback and `persona.include` for extra files.

`--profile` accepts a directory or a direct path to any YAML file. String values
support `${VAR}` and `${VAR:-default}` expansion. To create a new agent type, add
a directory — no code required.

## Skills

A **skill** is a reusable bundle in its own directory: a `skill.md` (YAML
frontmatter + an instruction body) and, optionally, a Python module that ships
its own tools. A profile engages a skill either statically (a `skills:` list,
always-on) or dynamically via the model-invoked `activate_skill` tool.

```
lingcore/skills/canvas/
  skill.md         # name, description, requested_tools, provides, module + guidance
  canvas_tools.py  # @tool functions registered when the skill is engaged
```

A code-shipping skill declares the tools it registers via `provides:` and the
module that defines them via `module:`. Crucially, **a skill cannot widen the
profile's permissions**: a shipped tool is only reachable if the profile also
lists its name under `tools:` — the `tools:` list is the single hard ceiling,
whether a tool is a builtin or skill-shipped. The bundled `canvas` skill (used
by the `teaching` profile) is the worked example: an async Canvas LMS client
exposing `canvas_courses`, `canvas_assignments`, `canvas_announcements`, and
`canvas_sync`. Its access token is read from an env var named by
`tool_options.canvas.token_env` — never stored in YAML — and downloads are
confined to the workspace.

```bash
export CANVAS_URL=https://<school>.instructure.com
export CANVAS_TOKEN=<your-canvas-token>
uv run lingcore --profile profiles/teaching   # "what's due this week?"
```

## Writing a tool

A tool is an async function whose first argument is a pydantic model (its
schema, advertised to the model) and whose second is the `ToolContext`:

```python
from pydantic import BaseModel, Field
from lingcore.tools import ToolContext, tool

class GreetArgs(BaseModel):
    name: str = Field(description="Who to greet.")

@tool(description="Return a friendly greeting.")
async def greet(args: GreetArgs, ctx: ToolContext) -> str:
    return f"Hello, {args.name}!"
```

The `@tool` decorator registers it; a profile activates it by listing `greet`
under `tools`. `ctx` carries the workspace path and a confirmation callback —
tools never reach for globals, so concurrent sessions stay isolated.

## Architecture

```
message.py   canonical Message/ToolCall/ToolResult; the only wire-format seam
llm.py       async LLMClient over the OpenAI SDK (the loop never imports openai)
events.py    AgentEvent union the loop emits
agent.py     the async run loop + Agent.from_profile  ← the core
composer.py  PromptComposer seam: per-turn system-prompt assembly
config.py    AgentProfile + YAML loading with ${ENV} expansion
paths.py     confined path validation + no-follow directory-handle writes
memory.py    ShortTermMemory protocol + WindowMemory (prefix-stable eviction) + SummarizingMemory (compaction)
sessions.py  SessionStore (SQLite per profile dir) + SessionMemory — history & resume
skills.py    Skill / SkillState / load_skill_tools — skills, incl. code-shipping
guardrails.py  Guardrail protocol + NoopGuardrail (pre/post hooks)
tools/       Tool / @tool / ToolRegistry / ToolContext, plus builtin tools
io/          Frontend protocol + run_session driver + Rich CLI
```

Two seams keep the design open: the loop talks only to an `LLMClient`-shaped
object (a different backend can drop in), and frontends consume only
`AgentEvent`s (a web/chat frontend can drop in). See `CLAUDE.md` for the full
set of invariants.

## Roadmap

The roadmap is ordered by leverage: first make the existing single-agent
experience more useful and measurable, then expand its integrations and
workflow model. The version groupings are directional rather than release
commitments.

1. **Interaction and onboarding**
   - Add an explicit stop/cancel control to LingChat, with defined behavior for
     messages submitted while a turn is running.
   - Add `lingcore profile init/list/doctor` and separate immutable profile
     templates from writable sessions, memory, and workspaces in the user's
     application-state directory. A wheel install should be useful without a
     repository checkout.

2. **v0.2 — Knowledge 1.0**
   - Finish the existing `knowledge` tool's incremental `index` and `hybrid`
     backends: stable chunks, content-hash updates, full-text plus embedding
     ranking, deletion handling, and a provider-independent embedding seam.
   - Preserve source metadata (path, page, and line range), emit retrieval
     events, and render verifiable citations in frontends.
   - Keep explicit tool-driven retrieval as the baseline, then add opt-in
     `auto_retrieve` prompt injection with a hard context budget.
   - Ship retrieval evaluations for relevance, stale-index handling, citation
     validity, and the offline grep fallback.

3. **v0.3 — Tracing and evaluations**
   - Trace model requests, tools, retrieval, retries, compaction, confirmation
     decisions, token usage, and latency, with sensitive content excluded by
     default. Start with local structured traces and offer an optional
     OpenTelemetry exporter.
   - Add a `lingcore eval` workflow for profile datasets, tool-trajectory
     assertions, quality checks, latency/cost reporting, and regression
     comparisons.

4. **v0.4 — MCP interoperability**
   - Add an MCP client with stdio and Streamable HTTP transports, initially for
     tools and later for resources and prompts.
   - Namespace discovered tools and keep every one beneath the profile's
     existing `tools` permission ceiling. Server descriptions remain untrusted;
     consent, cancellation, and progress must map through LingCore's frontend
     contracts.

5. **v0.5 — Durable workflows**
   - Let long-running turns survive a browser disconnect and replay their event
     stream on reconnect; add edit, regenerate, and session-fork controls.
   - Add schema-validated structured results so agents can participate in
     application workflows, plus an optional Responses API backend behind the
     existing `LLMClient` seam without weakening OpenAI-compatible portability.

Multi-agent handoffs, Discord/voice frontends, marketplaces, and additional
persona profiles remain later possibilities. They should follow retrieval,
tracing, and evaluations so added autonomy is observable and testable.

## Development

```bash
uv run pytest -q          # full suite (435 tests)
uv run pytest tests/test_agent.py -q
```

Tests drive the loop with a scripted fake LLM client (`tests/fakes.py`), so the
suite needs no network or API key.

### Safety note

`run_shell` executes arbitrary commands. The workspace bounds file tools
(path-escape is blocked and tested), but it is **not** a sandbox for the shell —
a command can still reach outside it. The confirmation gate, command timeout,
and output truncation/offload are the current mitigations; true isolation
(containers, seccomp) is a deliberate next step. No commands are auto-approved
by the shipped profiles. A configured multi-token allow pattern deliberately
matches trailing arguments, while shell control syntax (`;`, `&`, `&&`, pipes,
redirects, substitutions, and newlines) always falls back to confirmation and
cannot append another command to an approved prefix.

Security-sensitive workspace creation (attachment ingest, Canvas downloads,
and staged tool output) uses no-follow directory descriptors for every parent
component and keeps the validated parent open through create/rename. Swapping a
checked directory for a symlink therefore cannot redirect the write outside the
workspace; platforms without the required secure descriptor operations fail
closed.

`fetch_url` reduces SSRF risk by resolving each host (and every redirect hop)
and refusing any that maps to a loopback, link-local, or private address —
alternate IP encodings (decimal/hex/octal) and credentialed URLs are rejected
too. It then pins the connection to the vetted IP (the Host header and TLS
verification stay on the hostname), so DNS rebinding can't redirect the request
after the check. DNS resolution and downloaded body size are bounded. Profiles
can opt into private hosts with `tool_options.fetch_url.allow_private_hosts:
true` for trusted local workflows (e.g. a local Ollama or an internal API).

## License

See [LICENSE](LICENSE).
