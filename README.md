# LingCore

A lightweight, config-driven async agent framework.

The same runtime becomes a different kind of agent — coding assistant,
role-play character, teaching helper, psych consultant — purely by loading a
different **profile**. Adding a new agent type is a config file, not new code.

Built directly on the OpenAI SDK (chat-completions + tool calling), so it works
against OpenAI, local servers like **Ollama** and **vLLM**, or any
OpenAI-compatible endpoint by pointing at a different `base_url`.

> Status: MVP. The coding agent runs over the CLI or the sibling LingChat web
> app. Role-play / teaching / psych profiles use the same runtime.

## Features

- **Profile-driven** — model, endpoint, persona, tool list, workspace, memory,
  and sampling all live in a YAML file. Secrets stay in exported environment
  variables or an optional profile-local `.env`, never in YAML.
- **Thin async core** — a small, owned agent loop with streaming, parallel tool
  calls, and a hard iteration cap. No heavyweight orchestration framework.
- **Pluggable tools** — a tool is an `async` function plus a pydantic args
  model and a `@tool` decorator. The coding agent ships with file read/write/
  edit, patch, directory listing, search, URL fetch, and a confirmation-gated
  shell.
- **Knowledge retrieval** — the `knowledge` tool keeps offline grep as its
  no-index default, with opt-in incremental semantic and hybrid retrieval over
  a local SQLite index. Stable source chunks retain page/line citations,
  unchanged vectors are reused by content hash, deleted and stale sources are
  handled explicitly, and remote embedding/reranking credentials stay in the
  environment.
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
  `sessions.enabled: false`. Stable message sequences and explicit rewind/
  truncate primitives let frontends stop a turn safely or edit an earlier user
  message and regenerate its branch. Compacted working-set snapshots and
  dynamic skill state are durable too: restart restores the newest valid
  snapshot plus its raw tail and re-enables only skills still allowed by the
  current profile, without widening an old approval to newly high-risk tools.
  Snapshot tails are checked against the canonical transcript, and only two
  full snapshot bodies are retained; older event metadata remains available.
  Both are recorded as message-anchored events with monotonic replay cursors;
  rewinding a branch removes its derived events as well. A
  validated transcript prefix can also be forked atomically into a new session,
  carrying compatible snapshots/skill state with fresh cursors while preserving
  parent/root provenance.
- **Frontend-agnostic** — the loop emits events consumed by both the CLI and the
  sibling LingChat web app; another frontend can render the same contract
  without touching core.

## Install

Requires Python ≥3.11. Uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lllluolingyu/LingCore.git
cd LingCore
uv sync
```

The source checkout includes the example profiles used below. Wheels contain
the runtime package but not those repo-root, writable profile directories, so
an installed `lingcore` command must be given an external profile explicitly:

```bash
lingcore --profile /path/to/my-agent
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

For local development, put values in the selected profile's gitignored `.env`:

```dotenv
# profiles/coding/.env
LINGCORE_BASE_URL=https://api.openai.com/v1
LINGCORE_MODEL=gpt-4o
LINGCORE_API_KEY_ENV=OPENAI_API_KEY   # names the var holding your key
OPENAI_API_KEY=sk-...
```

Then launch normally:

```bash
uv run lingcore doctor --profile profiles/coding
uv run lingcore
```

`lingcore doctor` is offline and read-only. It validates the profile YAML,
reports missing/empty required variables and whether each is sourced from the
profile `.env` or the process environment, checks `.env.example` coverage, and
exits nonzero for configuration errors. It never prints values, opens sessions,
creates a workspace, builds the agent, or contacts a provider.

Exporting the same variables remains supported as a fallback when the selected
profile's `.env` does not define them, which is useful for CI and production
secret injection.

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
prompt-layer files. The source repository includes four examples:

- `profiles/coding/` — default profile, targets a keyed provider via env vars.
- `profiles/coding_ollama/` — keyless variant for local Ollama/vLLM.
- `profiles/daily/` — general-purpose assistant (research, notes, persistent memory; no shell).
- `profiles/teaching/` — teaching assistant built on the Canvas skill (courses, due dates, file sync).

```
my-agent/
  .env           # optional local variables/secrets (gitignored; never commit)
  .env.example   # secret-free setup template (commit this when env is needed)
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

`--profile` accepts a directory or a direct path to any YAML file. LingCore
loads only the `.env` beside that selected YAML—never one discovered from the
current directory or a parent—before expanding `${VAR}` and
`${VAR:-default}`. Values stay scoped to the loaded profile (they are not copied
into global `os.environ`) and override same-named variables inherited from the
launching process. Real `.env` files are gitignored; commit a secret-free
`.env.example` when a profile or skill needs setup documentation. The example
`coding`, `daily`, and `teaching` profiles include one; keyless
`coding_ollama` needs none. Run `lingcore doctor --profile <path>` after copying
or editing one. To create a new agent type, add a directory—no code required.

## Knowledge retrieval

Enable the `knowledge` tool in a profile to search a configured workspace
corpus. Its default is deliberately offline and embedding-free:

```yaml
tools:
  - knowledge

tool_options:
  knowledge:
    backend: grep
    sources: ["docs/**/*.md", "notes/**/*.txt", "papers/**/*.pdf"]
    embedding:
      enabled: false  # default even when this block is omitted
```

`action: query` then searches live files and returns `path:line` matches;
`action: index` is a no-op. To opt into semantic retrieval, switch to `index`
(embedding ranking) or `hybrid` (SQLite full-text + embedding ranking), supply
the named key, and build the index once:

```yaml
tool_options:
  knowledge:
    backend: hybrid
    sources: ["docs/**/*.md", "papers/**/*.pdf"]
    embedding:
      enabled: true
      provider: siliconflow
      base_url: https://api.siliconflow.cn/v1
      api_key_env: SILICONFLOW_API_KEY
      model: Qwen/Qwen3-VL-Embedding-8B
      dimensions: 768
      batch_size: 32
    reranker:
      enabled: false  # optional second API call; also off by default
      provider: siliconflow
      base_url: https://api.siliconflow.cn/v1
      api_key_env: SILICONFLOW_API_KEY
      model: Qwen/Qwen3-VL-Reranker-8B
```

```dotenv
# my-agent/.env (or export the same variable)
SILICONFLOW_API_KEY=...
```

```bash
uv run lingcore --profile my-agent --workspace /path/to/corpus
# Ask the agent to call knowledge with action=index, then action=query.
```

The index lives at `<workspace>/.lingcore/knowledge.sqlite3` by default and is
updated incrementally. A full index removes deleted files; `paths` on the
`index` action updates only selected workspace-relative files/directories/globs.
Queries never return changed or deleted indexed content: they show a stale-index
notice until it is rebuilt. UTF-8 text is chunked with line ranges; PDFs are
extracted page by page when the optional PDF dependency is installed. Retrieval
output is capped by `max_hits`, `max_excerpt_chars`, and `max_context_chars`.
The provider adapters follow SiliconFlow's
[embedding](https://api-docs.siliconflow.cn/docs/api/embeddings-post) and
[reranking](https://api-docs.siliconflow.cn/docs/api/rerank-post) contracts;
alternate providers can implement the small `EmbeddingProvider` and
`RerankingProvider` protocols in `lingcore/knowledge.py`.

## Skills

A **skill** is a reusable bundle in its own directory: a `skill.md` (YAML
frontmatter + an instruction body) and, optionally, a Python module that ships
its own tools. A profile engages a skill either statically (a `skills:` list,
always-on) or dynamically via the model-invoked `activate_skill` tool.

```
lingcore/skills/canvas/
  .env.example    # safe declaration of required variables; no real secrets
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
confined to the workspace. The required variables are documented beside the
skill, but the actual values belong to the profile that engages it:

```dotenv
# profiles/teaching/.env
CANVAS_URL=https://<school>.instructure.com
CANVAS_TOKEN=<your-canvas-token>
```

Start from the teaching profile's combined safe template, then edit the copied
file and check it before launch:

```bash
cp profiles/teaching/.env.example profiles/teaching/.env
uv run lingcore doctor --profile profiles/teaching
uv run lingcore --profile profiles/teaching   # "what's due this week?"
```

Do not put the real token in `lingcore/skills/canvas/`: that directory is
package code shared by every profile and may be committed or replaced during an
upgrade. The Canvas template covers the skill's variables only; also set the LLM
provider key named by the teaching profile if it is not already exported. A
`CANVAS_TOKEN` in the teaching profile's `.env` overrides an exported value, so
each profile reliably selects its own Canvas account.

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
config.py    AgentProfile + scoped profile .env + YAML ${ENV} expansion
doctor.py    offline profile/env/.env.example diagnostics (never prints values)
paths.py     confined path validation + no-follow directory-handle writes
knowledge.py provider-neutral embedding/reranking seams + SiliconFlow adapters
memory.py    ShortTermMemory protocol + WindowMemory (prefix-stable eviction) + SummarizingMemory (compaction)
sessions.py  SessionStore + SessionMemory — transcript, snapshots, replay, rewind, fork
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
   - Implemented: LingCore exposes an explicit cancellation lifecycle and
     stop-safe session truncation; LingChat adds Stop, rejects concurrent turns,
     and lets users edit any stored user message to rewind and regenerate that
     branch while preserving its attachments.
   - Implemented: the additive session-schema v2 persists compaction snapshots
     and dynamic skill state as message-anchored runtime events. Resume restores
     snapshot + raw tail, LingChat replays compaction/skill transitions after a
     restart, and monotonic cursors support incremental event consumers.
   - Implemented: atomic session-prefix forks preserve the source branch, remint
     copied event cursors, and record parent/root provenance. LingChat can fork
     and regenerate from a user message or continue from a final assistant reply.
   - Implemented: `lingcore doctor` performs offline, secret-safe profile and
     environment diagnostics. Add `lingcore profile init/list` and separate
     immutable profile templates from writable sessions, memory, and workspaces
     in the user's application-state directory. A wheel install should be useful
     without a repository checkout.

2. **v0.2 — Knowledge 1.0**
   - Implemented: the `knowledge` tool's incremental `index` and `hybrid`
     backends now provide stable chunks, content-hash vector reuse, full-text
     plus embedding ranking, deletion/stale handling, and provider-independent
     embedding/reranking seams. Embedding remains explicitly opt-in.
   - Source path/page/line metadata and verifiable tool-result citations are in
     place; add structured retrieval events and render them natively in each
     frontend.
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
   - Build a detached turn runner so an *in-flight* model/tool task can survive a
     browser disconnect. Durable completed-state replay is now in place; task
     ownership, leases, progress events, and reconnect attachment remain.
   - Build on the implemented edit/fork flows with regenerate-without-edit and
     explicit merge/export controls where real workflows need them.
   - Add schema-validated structured results so agents can participate in
     application workflows, plus an optional Responses API backend behind the
     existing `LLMClient` seam without weakening OpenAI-compatible portability.

Multi-agent handoffs, Discord/voice frontends, marketplaces, and additional
persona profiles remain later possibilities. They should follow retrieval,
tracing, and evaluations so added autonomy is observable and testable.

## Development

```bash
uv run pytest -q          # full suite
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

Security-sensitive workspace state (attachment ingest, Canvas downloads,
staged tool output, and the knowledge index) uses no-follow directory
descriptors for every parent component and keeps the validated parent open
through reads and atomic create/rename. The SQLite knowledge database is loaded
through a bounded no-follow descriptor and serialized back atomically, so it
never has to reopen an attacker-swappable workspace path. Swapping a checked
directory for a symlink therefore cannot redirect the operation outside the
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
