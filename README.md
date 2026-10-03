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
  edit, patch, directory listing, search, URL fetch, structured read-only Git
  inspection, a confirmation-gated shell, and a `todo_write` task checklist.
- **Task checklist** — `todo_write` lets the agent keep a whole-list-replaced
  todo list for multi-step work. The list persists with the session, rolls back
  with Stop/Edit, survives compaction verbatim, and frontends render it from a
  `TodoUpdated` event. It is never injected into the system prompt, so updates
  do not invalidate the cached prompt prefix.
- **Safe bounded workspace search** — recursive content and filename lookup
  supports path scoping, globs, literal/regex and case-insensitive matching,
  context lines, directory pruning, and hard time/file/hit budgets. Candidate
  files are bounded no-follow reads, symlinks are skipped, and every result
  reports scan coverage.
  Regex matching shares the scan deadline, including time spent inside a
  single expensive match; timed-out searches retain earlier results and report
  partial coverage. Patterns use Python regex syntax with the `regex` package's
  VERSION0 matching; Unicode case-insensitive matches can differ from stdlib `re`.
- **Knowledge retrieval** — the `knowledge` tool keeps offline grep as its
  no-index default, with opt-in incremental semantic and hybrid retrieval over
  a local SQLite index. Stable source chunks retain page/line citations,
  unchanged vectors are reused by content hash, deleted and stale sources are
  handled explicitly, and remote embedding/reranking credentials stay in the
  environment.
- **Cache-aware context** — built so the request prefix stays stable and
  provider prompt-caching actually hits. Tool output is kept lean and
  deterministic (`read_file` is paginated + line-numbered; `search` has stable
  ordering and coverage; heavy `search` / `run_shell` / `fetch_url` output is
  offloaded to a workspace file and read back on demand);
  the conversation window evicts in stable chunks instead of sliding every turn;
  and when it nears full, the oldest history is **compacted** (summarized) rather
  than dropped. An opt-in `llm.send_prompt_cache_key` pins a session's requests
  to the same cache node, and the persistent `memory.md` can likewise
  auto-compact at a size limit instead of refusing writes.
  Token counting reuses an encoding already loaded in the process, or uses a
  deterministic UTF-8 byte estimate. Rendering never downloads tokenizer data;
  a disk cache alone does not enable exact token counting.
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
- **Frontend-agnostic** — the loop emits events consumed by the CLI, official
  Telegram channel, and sibling LingChat web app; another
  frontend can render the same contract without touching core.
- **First-party Telegram channel** — an allowlisted private-chat Bot API
  frontend with streaming, attachments, confirmations, Stop, session
  switching, per-user workspaces/memory/history, polling, and webhook runners.

## Install

Requires Python ≥3.11. Uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lllluolingyu/LingCore.git
cd LingCore
uv sync
```

The source checkout includes the example profiles used below. Wheels include
immutable copies of the same templates; initialize one into writable user state
before the first installed-wheel run:

```bash
lingcore profile list
lingcore profile init                 # initializes the keyed `coding` default
cp ~/.local/state/lingcore/profiles/coding/.env.example \
  ~/.local/state/lingcore/profiles/coding/.env
lingcore doctor
lingcore
```

On Linux the default root follows `XDG_STATE_HOME` (falling back to
`~/.local/state/lingcore`); macOS and Windows use their normal application-state
locations. `LINGCORE_STATE_HOME` overrides the LingCore state root. Templates
are copied atomically and never overwrite an existing profile. Use
`--destination` for an explicit location, or initialize the local profile as
the installed default with `lingcore profile init coding_ollama --name coding`.

PDF text extraction (the `pdf2md` tool and the automatic PDF→text fallback)
needs PyMuPDF, which is AGPL-3.0 while LingCore is Apache-2.0 — so it ships as
an optional extra rather than a base dependency. A dev clone already gets it
via `uv sync` (dev group); a plain install opts in with:

```bash
pip install 'lingcore[pdf]'
```

Telegram support is shipped in the main wheel but its runtime dependency is an
opt-in extra:

```bash
pip install "lingcore[telegram]"
```

The extra pins `python-telegram-bot[rate-limiter,webhooks]==22.8`; it supports
the same Python 3.11–3.14 matrix and shares LingCore's `httpx==0.28.1`.

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

### Telegram

Telegram is a conversational Bot API frontend (not a broadcast-channel
publisher). Install the extra, copy the safe example into the selected profile,
and fill the *named* variable in that profile's `.env`:

```bash
pip install "lingcore[telegram]"
cp telegram.yaml.example profiles/coding/telegram.yaml
cp profiles/coding/.env.example profiles/coding/.env  # if needed
```

```dotenv
# profiles/coding/.env
TELEGRAM_BOT_TOKEN=<token from BotFather>
TELEGRAM_WEBHOOK_SECRET=<long random value>  # webhook mode only
```

Add your numeric Telegram user ID to `allowed_user_ids`, then diagnose and run:

```bash
lingcore doctor --profile profiles/coding \
  --telegram-config profiles/coding/telegram.yaml
lingcore telegram --profile profiles/coding
```

Use `--telegram-config <path>` to select a different file and
`--telegram-mode polling|webhook` for a one-run mode override. Telegram mode
does not accept the interactive CLI's `--continue`, `--resume`, `--no-session`,
`--list-sessions`, or `--workspace` flags; users switch sessions in chat with
`/new`, `/sessions`, and `/resume <id-prefix>`, and cancel with `/stop`.

Only allowlisted numeric users in private chats reach an Agent. Each user has a
separate workspace, memory file, session database, live Agent/profile copy, and
mutable tool options:

```text
<state_dir>/
  bridge.sqlite3
  users/<telegram-user-id>/
    workspace/
    memory.md
    sessions.db
```

`bridge.sqlite3` durably records the explicitly active session, so a `/resume`
selection survives a silent restart. An invalid stored selection is ignored and
replaced with a fresh session after its runtime builds successfully. Inbound
update and album deduplication is bounded in memory and therefore does **not**
survive a process restart. Telegram deployments are single-process;
cross-process leases and a durable update queue are not provided.

Polling retains pending updates:

```bash
lingcore telegram --profile /path/to/profile --telegram-mode polling
```

For webhook deployment, set `mode: webhook`, an externally reachable HTTPS
`webhook.public_url`, and `webhook.secret_token_env`. LingCore derives the
listener path from that URL, validates Telegram's secret header through PTB,
and listens on plain HTTP at `webhook.listen:webhook.port`; terminate TLS at a
trusted reverse proxy:

```bash
lingcore telegram --profile /path/to/profile --telegram-mode webhook
```

Telegram accepts text/captions with one photo or document (5 MiB image and
10 MiB file limits). Albums, voice, video, stickers, and other media are
rejected. Streaming output is plain text; tool activity shows tool names and
success/failure only, never full result bodies. PTB rate-limits requests and
retries one `RetryAfter`; if rendering still fails, LingCore keeps the completed
turn and makes a best-effort plain-message delivery instead of rolling it back.

Type a message; the agent streams its reply and shows each tool call, with a
colored diff preview for `edit_file`/`patch_file` and a dim token-usage footer
after each turn. Shell commands prompt for confirmation before running; `[A]`
allows the displayed token prefix for the rest of the session. **Ctrl-C stops
the running turn** (the submitted message is kept, its partial reply and tool
state are discarded) and a second Ctrl-C quits. In-session commands:

| Command | Effect |
|---|---|
| `/new` | start a fresh session |
| `/sessions` | list stored sessions for the profile |
| `/resume <id>` | switch to a stored session by id prefix |
| `/usage` | token usage for the current session |
| `/help` | list commands |
| `/exit` | quit (also `/quit`, `/q`, Ctrl-D) |

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

History lands in `<profile>/sessions.db` — delete the file to wipe it. Checkout
profiles live at the repo root (`profiles/`); wheel templates are copied into
writable user state by `lingcore profile init`. Both therefore keep history
without writing into installed package code. A manually selected profile inside
an installed package still runs ephemeral with a one-line notice.

`tiktoken` vocabularies are resolved lazily on the first context render. If the
cache is empty and resolution is unavailable offline, LingCore falls back to a
deterministic UTF-8 byte/token estimate; local agent assembly and Ollama use do
not require that download.

## Profiles

A profile is a **directory** containing a `config.yaml` and optional Markdown
prompt-layer files. The source repository and wheel template manifest include
four examples (`lingcore profile list` shows the installed set):

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
`persona.project_instructions` lists workspace-relative instruction files (the
coding profiles use `[AGENTS.md, CLAUDE.md]`): the first regular file found is
re-read into the prompt on every request, through no-follow confined reads and
bounded in size. It is read-only context and can never grant a tool.

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

### Native Anthropic, caching, and thinking

Select `backend: anthropic` to use the native Messages API. Its default endpoint
is `https://api.anthropic.com`; set `base_url` explicitly for a compatible proxy.

Prompt caching is enabled by default (`llm.prompt_caching: true`). The adapter
places explicit five-minute `cache_control` breakpoints on the last tool
definition, the system prompt, and the last two user/tool-result turns. Keeping
the previous turn's breakpoint lets large parallel tool batches reuse the
previous cache entry beyond the normal 20-block lookup window. This uses at
most four breakpoints and leaves signed thinking blocks unchanged.

Set `llm.prompt_caching: false` to disable these default markers for a proxy
that rejects them. Native `sampling.cache_control` (or its `extra_body`
equivalent), explicit block-level markers, and `extra_body` overrides of tools,
system, or messages take precedence over the default policy. For example,
`sampling.cache_control: {type: ephemeral, ttl: 1h}` selects Anthropic's automatic
one-hour caching on endpoints that support it. `llm.send_prompt_cache_key`
applies only to OpenAI; it does not enable Anthropic caching.

A first request writes the cache; subsequent requests can read it while the
prefix remains identical. Cache hits still depend on the provider/proxy
honoring the markers, the model's minimum cacheable length, and the cache TTL.
Changing tools or the system prompt, or compacting history, invalidates affected
prefixes. Check `UsageReported.usage.cached_input_tokens` for reported cache
reads. See [Anthropic's prompt-caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

For a model that supports adaptive thinking:

```yaml
llm:
  backend: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
  preserve_reasoning: true
  sampling:
    max_tokens: 16000
    thinking:
      type: adaptive
      display: summarized
      block_binding:
        prefix_mismatch_behavior: drop_block
    extra_headers:
      anthropic-beta: thinking-binding-controls-2026-08-01
```

The binding control and matching beta header let Anthropic drop stale thinking
after prompt/tool changes or history trimming. Keep both settings for models
with prefix-bound thinking: LingCore's skill activation and memory compaction
can change earlier context, and omitting these controls can then cause a 400.
See [Anthropic's preserved-thinking guidance](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking).

For models that support manual extended thinking, use
`thinking: {type: enabled, budget_tokens: 10000}` instead. Set `max_tokens` high
enough for both thinking and the answer. Mode availability and budget rules
depend on the model; see [Anthropic's thinking documentation](https://platform.claude.com/docs/en/build-with-claude/thinking).

Returned thinking signatures and redacted blocks are always preserved in order
through tool calls and saved sessions, including when a model thinks by default.
`preserve_reasoning: true` additionally stores readable thinking in
`Message.reasoning_content` and streams it as `LLMChunk.reasoning_delta` to direct
client callers. The agent keeps this separate from reply text. This flag does
not enable thinking; the `sampling.thinking` setting controls that.
Provider-reported thinking tokens appear in `UsageReported.usage.reasoning_tokens`.

### Guardrails

`guardrail.policy: noop` remains the default. A profile can select a third-party
implementation without editing LingCore by naming either a
`lingcore.guardrails` package entry point or a Python target; `options` are
passed to its class/factory as keyword arguments:

```yaml
guardrail:
  policy: my_safety_package.guardrails:PsychGuardrail
  options:
    crisis_message: "Contact local emergency services now."
```

The loaded object must provide async `pre_input(text)` and `post_output(text)`
methods. LingCore ships only the no-op policy; domain-specific safety behavior
belongs to the selected profile/package.

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
updated incrementally. Concurrent index updates are serialized across tasks
and processes using a persistent adjacent `.lock` file; waiting for the lock
can be cancelled. A full index removes deleted files; `paths` on the
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

The bundled instruction-only `code-review` skill is also live: the `daily`
profile exposes `activate_skill`, and its authorized `read_file`/`search` ceiling
makes `code-review` dynamically offerable. To make it always-on in another
profile, add `skills: [code-review]` and authorize the tools it should receive.

Two bundled collaboration skills connect LingCore to independently installed
coding-agent CLIs:

- `codex` provides `codex_agent` for persistent Codex CLI consultation or an
  explicitly confirmed implementation handoff.
- `claude-code` provides `claude_code_agent` for the equivalent Claude Code
  workflow.

The coding profiles authorize both tools but hide them until their skill is
activated, using the exclusion form of the initial-tool gate:

```yaml
tools: [read_file, ..., activate_skill, codex_agent, claude_code_agent]
skill_gated_tools: [codex_agent, claude_code_agent]   # everything else starts enabled
```

(`initial_tools:` is the equivalent inclusion form; declare one or the other.)
Both tools mark themselves `high_risk`, so activating either skill requires
confirmation — it sends a task and workspace context to an external agent.
Consultation is read-only; implementation mode asks again before allowing
workspace edits. Codex invocations pin the non-interactive approval boundary;
Claude Code invocations disable inherited customizations and MCP servers and
confine built-in file tools to the workspace. Each tool has a `conversation`
argument (default: `default`): reuse a name for follow-up turns, or pass
`restart: true` to point that name at a fresh external session. Aliases are
isolated by workspace and LingCore session and survive resuming the LingCore
session; a run without a persisted session (`--no-session`) falls back to a
workspace-wide alias namespace. Install and authenticate the corresponding CLI
separately; LingCore uses the executable on `PATH`, or the path configured under
`tool_options.codex_agent.executable` /
`tool_options.claude_code_agent.executable` (`~` is expanded; a blank value is
treated as unset). Runner timeouts and output limits are configurable under
those same keys, and `lingcore doctor` validates them and reports whether each
CLI resolves.

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
tools never reach for globals, so concurrent sessions stay isolated. A tool that
runs code or mutates the workspace should declare `@tool(..., high_risk=True)`:
a skill requesting it then needs user confirmation before activation, the same
gate the builtin `run_shell`/`write_file`/`edit_file`/`patch_file` get by name.

## Architecture

```
message.py   canonical Message/ToolCall/ToolResult; the only wire-format seam
llm.py       async LLMClient over the OpenAI SDK (the loop never imports openai)
events.py    AgentEvent union the loop emits
agent.py     the async run loop + Agent.from_profile  ← the core
composer.py  PromptComposer seam: per-turn system-prompt assembly
config.py    AgentProfile + scoped profile .env + YAML ${ENV} expansion
doctor.py    offline profile/env/.env.example diagnostics (never prints values)
sandbox.py   typed host/Bubblewrap/OCI shell runners + process supervision
outer_agents.py  shared runner/aliases/spec for the codex + claude-code skills
paths.py     confined path validation + no-follow directory traversal/I/O
knowledge.py provider-neutral embedding/reranking seams + SiliconFlow adapters
memory.py    ShortTermMemory protocol + WindowMemory (prefix-stable eviction) + SummarizingMemory (compaction)
sessions.py  SessionStore + SessionMemory — transcript, snapshots, replay, rewind, fork
skills.py    Skill / SkillState / load_skill_tools — skills, incl. code-shipping
guardrails.py  Guardrail protocol + NoopGuardrail (pre/post hooks)
profiles.py  immutable template manifest + writable user-state initialization
tools/       Tool / @tool / ToolRegistry / ToolContext, plus builtin tools
io/          Frontend protocol + run_session driver + Rich CLI
integrations/telegram/  PTB-light bridge/state/rendering + thin PTB adapter
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
   - Implemented: the official Telegram Bot API channel provides private-chat
     allowlisting, per-user state isolation, streaming, attachments,
     confirmations, session commands, polling, and single-process webhooks.
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
     environment diagnostics. `lingcore profile init/list` separates immutable
     wheel templates from writable sessions, memory, and workspaces in the
     user's application-state directory, so a wheel works without a checkout.
   - Implemented: `run_shell` has strict, opt-in Bubblewrap and Docker/Podman
     backends with fail-closed startup, explicit mounts/environment, network
     isolation, OCI resource budgets, and supervised cleanup. The bundled
     coding profiles select Bubblewrap; the legacy host runner remains only for
     profiles that omit the `sandbox` block.

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

`run_shell` executes arbitrary commands, so confirmation and sandboxing are
separate controls. The shipped coding profiles require confirmation and use
Bubblewrap with no network, an empty filesystem view, `/usr` read-only, the
workspace read-write, dropped capabilities, and a bounded tmpfs. Profiles may
instead select Docker or Podman; Linux containers use a read-only root,
non-root user, no-new-privileges, dropped capabilities, and mandatory CPU,
memory, PID, and temporary-storage budgets. Native Windows containers require
Docker Hyper-V isolation and CPU, memory, and storage budgets. A configured
backend fails closed and never falls back to the host runner.

Omitting `tool_options.run_shell.sandbox` deliberately preserves the legacy
unsandboxed host runner for existing custom profiles; every result names its
runner. No commands are auto-approved by the shipped profiles. A configured
multi-token allow pattern deliberately matches trailing arguments, while shell
control syntax (`;`, `&`, `&&`, pipes, redirects, substitutions, and newlines)
always falls back to confirmation. See [Sandboxed shell runner](docs/sandboxing.md)
for backend configuration, platform requirements, doctor checks, and the threat
model.

Security-sensitive workspace operations (attachment ingest, Canvas downloads,
search traversal, staged tool output, and the knowledge index) use no-follow
directory descriptors for every parent component and keep the validated parent
open through reads and atomic create/rename. The SQLite knowledge database is
loaded through a bounded no-follow descriptor and serialized back atomically,
so it never has to reopen an attacker-swappable workspace path. Swapping a
checked directory for a symlink therefore cannot redirect the operation
outside the workspace; platforms without the required secure descriptor
operations fail closed.

The coding profiles expose a structured, read-only `git` builtin for status,
diff, log, show, and branch inspection, so those routine operations do not need
shell approval. It accepts no raw flags or shell text, disables pathspec magic,
external diff/textconv helpers, hooks, filesystem monitors, credential helpers,
lazy fetching, and optional index locks. Parent checkouts, linked/separate
worktrees, and alternate object stores are refused so Git metadata stays rooted
in the workspace. Repository-changing and networked Git commands still go
through confirmation-gated `run_shell`.

`fetch_url` reduces SSRF risk by resolving each host (and every redirect hop)
and refusing any that maps to a loopback, link-local, or private address —
alternate IP encodings (decimal/hex/octal) and credentialed URLs are rejected
too. It then pins the connection to the vetted IP (the Host header and TLS
verification stay on the hostname), so DNS rebinding can't redirect the request
after the check. DNS resolution and downloaded body size are bounded. Profiles
can opt into private hosts with `tool_options.fetch_url.allow_private_hosts:
true` for trusted local workflows (e.g. a local Ollama or an internal API).

Telegram refuses to start a profile that enables `run_shell` unless
`require_confirmation` is true and `allow_patterns` is empty. Every shell call
therefore needs an inline, user-bound approval; approvals time out, cannot be
reused by another user/chat, and are denied on Stop or shutdown. This is still
consent, not sandboxing: a custom Telegram profile may still choose the legacy
host runner. Keep the bundled sandbox block or deploy such a bot as a minimally
privileged user.

## License

See [LICENSE](LICENSE).
