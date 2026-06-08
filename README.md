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
- **Frontend-agnostic** — the loop emits events; the CLI renders them today, and
  a web or chat frontend can render the same events later without touching core.

## Install

Requires Python ≥3.11. Uses [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url> LingCore
cd LingCore
uv sync
```

## Quick start

### Local Ollama (no API key)

```bash
ollama pull qwen2.5-coder:7b      # or any model you have
cd /path/to/your/project
uv run lingcore --profile lingcore/profiles/coding_ollama
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

> The first message may pause briefly while `tiktoken` downloads its tokenizer
> data (cached afterward). This step needs network access once.

## Profiles

A profile is a **directory** containing a `config.yaml` and optional Markdown
prompt-layer files. Two are shipped:

- `lingcore/profiles/coding/` — default profile, targets a keyed provider via env vars.
- `lingcore/profiles/coding_ollama/` — keyless variant for local Ollama/vLLM.

```
my-agent/
  config.yaml    # llm, tools, memory, loop, guardrail
  world.md       # optional — environment / setting context
  role.md        # optional — persona
  workflow.md    # optional — operating method
  memory.md      # auto-created by the memory tool (opt-in)
```

`world.md`, `role.md`, and `workflow.md` are loaded automatically if present and
composed in that order to form the system prompt. `config.yaml` may also set
`persona.system_prompt` as an inline fallback and `persona.include` for extra files.

`--profile` accepts a directory or a direct path to any YAML file. String values
support `${VAR}` and `${VAR:-default}` expansion. To create a new agent type, add
a directory — no code required.

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
memory.py    ShortTermMemory protocol + WindowMemory
skills.py    Skill / SkillState — the model-invoked skill permission model
guardrails.py  Guardrail protocol + NoopGuardrail (pre/post hooks)
tools/       Tool / @tool / ToolRegistry / ToolContext, plus builtin tools
io/          Frontend protocol + run_session driver + Rich CLI
```

Two seams keep the design open: the loop talks only to an `LLMClient`-shaped
object (a different backend can drop in), and frontends consume only
`AgentEvent`s (a web/chat frontend can drop in). See `CLAUDE.md` for the full
set of invariants.

## Development

```bash
uv run pytest -q          # full suite (116 tests)
uv run pytest tests/test_agent.py -q
```

Tests drive the loop with a scripted fake LLM client (`tests/fakes.py`), so the
suite needs no network or API key.

### Safety note

`run_shell` executes arbitrary commands. The workspace bounds file tools
(path-escape is blocked and tested), but it is **not** a sandbox for the shell —
a command can still reach outside it. The confirmation gate, command timeout,
and output truncation are the current mitigations; true isolation (containers,
seccomp) is a deliberate next step.

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
