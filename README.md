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
  edit, directory listing, search, and a confirmation-gated shell.
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
uv run lingcore --profile lingcore/profiles/coding_ollama.yaml
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

A profile is a YAML file describing one agent. Two are shipped:

- `lingcore/profiles/coding.yaml` — default profile, targets a keyed provider via env vars.
- `lingcore/profiles/coding_ollama.yaml` — keyless variant for local Ollama/vLLM.

The shape:

```yaml
name: coding
workspace: ${LINGCORE_WORKSPACE:-.}      # tools are confined here

llm:
  model: ${LINGCORE_MODEL:-your-model}
  base_url: ${LINGCORE_BASE_URL:-https://your-provider/v1}
  api_key_env: ${LINGCORE_API_KEY_ENV:-}  # NAMES an env var; never the key itself
  sampling:
    temperature: 0.2

persona:
  system_prompt: |
    You are a coding assistant operating inside a sandboxed workspace.

tools:                                    # selected by name from the registry
  - read_file
  - write_file
  - edit_file
  - list_dir
  - search
  - run_shell

tool_options:
  run_shell:
    timeout: 60
    require_confirmation: true

memory:
  max_messages: 50
  max_tokens: 16000

loop:
  max_iters: 30
  parallel_tools: true

guardrail:
  policy: noop
```

String values support `${VAR}` and `${VAR:-default}` expansion. Run a custom
profile with `uv run lingcore --profile path/to/profile.yaml`. To create a new
agent type, copy the file, change the persona and tool list — no code required.

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
config.py    AgentProfile + YAML loading with ${ENV} expansion
memory.py    ShortTermMemory protocol + WindowMemory
guardrails.py  Guardrail protocol + NoopGuardrail (pre/post hooks)
tools/       Tool / @tool / ToolRegistry / ToolContext, plus builtin fs + shell
io/          Frontend protocol + run_session driver + Rich CLI
```

Two seams keep the design open: the loop talks only to an `LLMClient`-shaped
object (a different backend can drop in), and frontends consume only
`AgentEvent`s (a web/chat frontend can drop in). See `CLAUDE.md` for the full
set of invariants.

## Development

```bash
uv run pytest -q          # full suite (68 tests)
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

## License

See [LICENSE](LICENSE).
