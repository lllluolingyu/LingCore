# Delegation, workflows and the superagent — design proposal

Status: **phases 0–5 implemented** (2026-11-14).
The decisions and review corrections are recorded in [§12](#12-decisions).
Operator instructions and the implemented configuration are in
[Workflows and superagents](../workflows-and-superagents.md). The sections below
retain the design rationale and phased acceptance criteria.

The implementation baseline already includes the experimental `output_rewrite`
runtime stage and the coding profile's `/init` / `world` tool. These are separate
from the proposed delegation kernel. Runtime observations live in private state
(default `.lingcore/world.md`), require explicit per-save approval, and never
rewrite the authored profile file. `UsageReported.purpose` and the persisted
`TokenUsage.purpose` distinguish rewrite requests from the main context; old
records default to `main`. The sections below describe future changes relative
to that baseline, not a claim that those existing APIs are unchanged from older
LingCore releases.

References studied:

- Claude Code's dynamic **Workflow** tool: a script the model writes that
  orchestrates subagents with `agent()` / `parallel()` / `pipeline()` /
  `phase()`, structured outputs, a shared hard token budget, background
  execution with completion notification, and journal-based resume.
- [`deepseek-ai/dsh-superagent-prototype`](https://github.com/deepseek-ai/dsh-superagent-prototype):
  a proactive team agent. Sources produce Triggers. A deterministic gate and a
  routing session turn them into Tickets with immutable Mandates. One Executor
  session and directory per Ticket does the work, and results return through
  Channels, with durable effects and restart recovery.

## 1. Summary

LingCore gets **one new primitive** and **two layers that use it**:

```mermaid
flowchart TB
  subgraph L2a[Workflows]
    W[workflow spec<br/>YAML / JSON data] --> I[interpreter<br/>map · parallel · loop · keep]
  end
  subgraph L2b[Superagent]
    S[Sources] --> G[Gate + Policy] --> R[router agent] --> T[Ticket + Mandate]
    T --> A[Assignment]
  end
  D[delegate tool] --> K
  I --> K
  A --> K
  subgraph L1[Delegation kernel]
    K[Supervisor] --> E[Envelope<br/>authority ⊆ parent]
    K --> B[Run budget<br/>chained to ExecutionBudget]
    K --> C[Confirmation broker]
    K --> X[Worker = ordinary Agent<br/>from a role overlay]
  end
  X --> L0[Agent.run — unchanged loop]
```

- **L1, the delegation kernel** (`lingcore/delegation/`). It starts a *worker*:
  an ordinary `Agent` assembled from a *role* (a profile overlay). The worker
  runs under an *envelope* whose authority can only be narrower than its
  parent's, with a run budget charged against the parent, a confirmation
  broker in place of a user, and a structured *report* as its result. It is
  exposed directly as the `delegate` tool.
- **L2a, workflows** (`lingcore/workflows/`). These are deterministic control
  flow over workers. The model may write a spec on the fly (the "dynamic"
  part), but a spec is **data** run by a bounded interpreter. It is never
  code the host executes. Profiles ship named specs, and trusted plugins may
  ship Python workflows.
- **L2b, the superagent** (`lingcore/superagent/`). A long-running service turns
  events from Lark/Feishu (v1; other sources later) into authorized Tickets. A router agent
  that only routes assigns them to executor workers, and a durable store
  survives restarts.

Each layer can ship and be tested on its own. L1 is useful alone, and L2a and
L2b are thin over it.

## 2. What we take from each reference

### 2.1 Claude Code dynamic workflows

| Idea | Keep? | LingCore form |
| --- | --- | --- |
| Control flow in a program, not in the model's turn-by-turn choices | Yes | Workflow spec interpreted by the runtime |
| The model writes the program for this task | Yes, as **data** | `workflow_run(spec=…)`, validated before anything runs |
| Program language is JavaScript run by the host | **No** | LingCore has no sandboxed interpreter. Model-written Python in-process would be arbitrary code execution outside the `run_shell` sandbox. Operator Python only via consented plugins |
| `agent(prompt, {schema})` returns a validated object | Yes | `output:` schema → worker's `submit_result` tool, validated at the tool layer so the model retries on mismatch |
| `pipeline()` by default, `parallel()` only as a real barrier | Yes | `map` (no barrier between stages) vs `parallel` (barrier) |
| Failed agent → `null`, run continues | Yes | Same, **plus** counts of dropped items always logged ("no silent caps") |
| Concurrency cap, total-agent cap, item cap | Yes | `delegation.limits` + per-spec narrowing |
| Shared hard token budget (`budget.remaining()`) | Shared accounting only in v1 | Run budget chained into `ExecutionBudget`; `reported_tokens` is a stopping threshold with possible overshoot (§4.6) |
| `isolation: 'worktree'` | Yes, **different shape** | The `git` tool deliberately refuses linked worktrees and alternates (`lingcore/tools/builtin/git.py:111`). Use a hardened local *snapshot* clone and return changes as a patch artifact |
| Runs in background, notification on completion | Yes | Background run → `BackgroundInbox` delivery turn (the `process_monitor` pattern) |
| Resume: replay cached results from a journal | Yes, by node class | Node content, dependency values, the frozen input revision and occurrence determine reuse. Live readers rerun, and effecting nodes are never rerun (§5.4) |
| Explicit user opt-in because of cost | Yes | Tool in the `tools:` ceiling (operator) + cost-preview confirmation per run (user) |
| Subagent types | Yes | **Roles**: operator-authored profile overlays |

### 2.2 DSH superagent

| Idea | Keep? | LingCore form |
| --- | --- | --- |
| Host enforces authority. Prompts and quoted text are data | **Yes, central** | Envelope (L1) and Mandate (L2b) computed in code. Briefs, reports and triggers never authorize |
| Router only routes, never designs or investigates, never waits | Yes | Router role whose tool ceiling has *no* file/shell tools. Results arrive as turns, never polled |
| Forward the original request, not the router's paraphrase | Yes | Every brief carries the originating request verbatim, quoted as data |
| One session + one directory per unit of work | Yes | Executor workers get a persistent session and a `ticket` workspace |
| Structured report states + one concrete question | Yes | `progress / waiting / ready / blocked / failed / dropped` + `question` |
| Store first, act after commit. Durable effects with backoff | Yes | Superagent store + effects table. Run journal for L1/L2a |
| Worker slots with one reserved for a person's follow-up | Yes | Executor slots + reserved human slot |
| Restart: interrupted work restarts with a recovery note | Yes | Same note: "check the directory, git state and channel before repeating outside actions" |
| Attention economy: speak only when someone must act | Yes | Workers never address the user. Questions bubble up as structured fields. Confirmations brokered and serialized |
| Few concepts. One writer per state. Pure core without I/O | Yes | `superagent/core/` is synchronous and I/O-free |
| Model-surface size tests | Yes | A test bounds every model-facing description/prompt added here |
| Workers run commands **without approval** as the OS user | **Not on the host** | Executors inherit the profile's fail-closed sandbox and run inside it without approval. The host runner is refused ([§6.6](#66-unattended-execution)) |
| GitHub PR following, Correlation, admission tables | Defer | Source-specific extensions later. The core keeps an allowlist + accept mode |

### 2.3 What LingCore already has that this builds on

| Need | Existing seam |
| --- | --- |
| Run an Agent to completion without a human | `execution.run_headless` / `ScriptedFrontend` (deadline, Stop handshake, bounded cleanup, typed `ExecutionOutcome`) |
| Request/tool admission, reported-token stopping threshold and wall deadline | `ExecutionBudget` + `InstrumentedClient(budget=…)`; tokens are charged after a request |
| Start a turn from background events, safely | `BackgroundInbox` + `run_session` (`supports_background`, paused after Stop) |
| Kill on rollback of the starting turn | `turn_rolled_back(started_at)` pattern in the process plugin |
| Delegation to an outside agent | `outer_agents.py` (Codex / Claude Code CLIs) |
| Lineage between sessions | `fork_session` provenance in `sessions.meta` |
| Per-batch authorization snapshot | `dispatch._dispatch` (`lingcore/_agent/dispatch.py:67`) |
| Channels with per-user isolation | `integrations/telegram`, `integrations/lark` |
| Approved environment observations without modifying authored prompts | `world.py` / `world` tool; private runtime files, per-save confirmation, advisory lock and revision check; channel-scoped paths |
| Optional wording pass with separate model credentials | `output_rewrite.py`; own-agent usage labelled `output_rewrite`, persisted for frontend replay |

## 3. Design principles

1. **Config, not code.** Roles, workflows and the superagent are profile YAML.
   A new kind of team or workflow is a file. Python arrives only through
   plugins named in `plugins:`.
2. **Delegation never widens authority.** Each level's tool ceiling is a
   subset of the one above, snapshotted when the delegating call was
   authorized. Roles are operator config. The model only picks among them.
   A superagent executor inherits from the router that accepted its Ticket.
   Any tool beyond that needs a person's recorded approval (§6.2).
3. **Text is data.** Briefs, worker reports, workflow values, triggers and
   quoted documents never grant a tool, a role, a budget or a mandate.
4. **Deterministic where possible.** The model chooses *which* program runs,
   and the runtime runs it the same way every time. That is what makes resume
   and audit possible.
5. **Limits and their guarantees are visible.** Agents, depth, concurrency,
   items and output size have caps; wall time has a deadline. Reported-token
   stopping thresholds can overshoot (§4.6). When a cap drops work, the drop
   is reported.
6. **Store first, act after commit.** Journals and effects are written before
   side effects and replayed after a crash. Side effects that may have
   happened come with a recovery note.
7. **Attention is the scarce resource.** Workers have no user. One concrete
   question bubbles up at a time, and confirmations are serialized and
   labelled by origin.
8. **The single-agent loop stays as it is.** A worker is an ordinary `Agent`.
   The loop gains only small, general seams ([§4.10](#410-core-seams-required)).

## 4. Layer 1 — delegation kernel

### 4.1 Concepts

| Concept | Definition | Rules |
| --- | --- | --- |
| **Role** | Operator-authored overlay on the parent profile: persona, tool list, model alias, workspace mode, iteration/memory caps, static skills | Validated at profile load. Its tools must be within the parent's ceiling. Never model-authored |
| **Envelope** | The worker's effective authority: tools, high-risk subset, workspace, budget, confirmation policy, depth | Computed in code from role ∩ (parent snapshot ∪ person-approved extras, §6.2) ∩ requested narrowing |
| **Brief** | The worker's single user turn: task text + the originating request verbatim + workspace note | Data. Bounded (task ≤ 20k chars, request ≤ 4k with a pointer beyond) |
| **Worker** | One `Agent` built from the role, run headless under its envelope | Has its own memory, session, plugin instances and usage meter |
| **Report** | The worker's typed result | `completed / blocked / failed / cancelled / budget_exceeded / timed_out` |
| **Run** | A set of workers started by one `delegate` or `workflow_run` call | Owns a run budget, a journal and a cancellation scope |
| **Supervisor** | One per parent `Agent`, owned by its `ToolContext`. In the superagent the parent is the router Agent, which lives as long as the service (§6.2) | Slots, broker, run registry, lifecycle. Closed by `Agent.aclose` |

### 4.2 Roles are profile overlays

```yaml
# profiles/coding/config.yaml (excerpt)
delegation:
  roles_dir: roles
  models:                       # aliases the model may choose; endpoints/keys stay here
    fast:
      model: ${LINGCORE_FAST_MODEL:-gpt-6-astra}   # point at a cheaper model
      base_url: ${LINGCORE_BASE_URL:-https://www.cctq.ai/v1}
      api_key_env: ${LINGCORE_API_KEY_ENV:-LLY_API_KEY}
```

```yaml
# profiles/coding/roles/reviewer.yaml   (+ optional roles/reviewer.md persona)
description: Reviews a change for one dimension. Read-only.
model: fast                     # alias, or omit to use the parent's llm
tools: [read_file, list_dir, search, git]
workspace: frozen               # shared | frozen | snapshot | scratch
context: [project_instructions] # also allowed: memory, world (read-only layers)
skills: [code-review]           # static skills from the parent's catalog
max_iters: 40
memory: {max_tokens: 60000}
```

- A built-in `general` role always exists. It uses the parent persona and the
  parent's tools minus delegation tools. It is the default when the model
  names no role, like Claude Code's general-purpose subagent.
- **Inherited and not removable:** guardrail, every *hook-providing* plugin
  enabled on the parent (their friction applies to workers too),
  `tool_options` (sandbox, network policy, allowlists), the profile `.env`
  scope and `environment_binding`. A role may not set `tool_options`, `llm`
  endpoints or `plugins` in v1.
- **Removed unless the role names them:** tools that write profile-scoped
  persistent state (`memory`, `world`, `calendar_*`). Parallel workers must
  not race on `memory.md`. Read-only `world` context means the authored
  `world.md` and approved runtime observations; inheriting that context grants
  no write tool. Granting the `world` tool still requires per-save approval
  through the broker, including stale-revision checks under its file lock.
- Private runtime state objects in `tool_options` (skill state, todo state,
  summarizer, world file) are never copied. Each worker gets fresh ones, as
  Telegram and `/new` already do with deep-copied options.
- **Role skills never widen the envelope.** A role's static skills are
  engaged inside the worker, and each unlocks only
  `skill.requested_tools ∩ envelope`. A tool that is skill-gated and inactive
  in the parent is outside the batch snapshot, so listing it in `role.tools`
  or engaging a skill that requests it does not grant it. The rule is the
  same for `delegate`, workflows and superagent executors. An executor can
  obtain such a tool only as an approved extra ([§6.2](#62-router-and-executor-roles)).
- `lingcore doctor` validates roles offline: ceiling subset, model aliases,
  skills, persona files. It warns when a role lists a tool that the profile's
  initial set plus static skills can never grant.

### 4.3 Envelope computation

```text
snapshot     = the delegating call's batch authorization (dispatch.py:67)
extras       = tools a person approved for this Ticket (§6.2); empty everywhere else
tools        = role.tools ∩ (snapshot ∪ extras) ∩ (requested_tools or role.tools)
child_depth  = depth + 1                           # the profile's own Agent is depth 0
if child_depth >= delegation.max_depth (default 1):
    tools   -= {delegate, workflow_run, workflow_*}
high_risk    = tools ∩ (parent.high_risk_set ∪ high-risk extras)   # same floor as invariant 11
workspace    = role.workspace narrowed by the request (shared → frozen or scratch allowed, not the reverse)
budget       = run budget child, ≤ role caps
confirm      = broker.for_worker(label) | deny
```

The snapshot is important. A parent's skill-gated tool that is not active in
the current batch is **not** inherited, so delegation cannot be used to reach
`codex_agent` without the `activate_skill` confirmation. A sibling
`activate_skill` in the same batch also cannot widen a worker (invariant 11's
batch rule, extended). Each `run_one` task publishes the snapshot through a
contextvar (`lingcore.tools.dispatch_authorization`), because contextvars are
copied per task and `ToolContext` is shared.

`extras` exist only for superagent executors. Host code reads them from the
Ticket's approval record. They are not a tool argument, and no `delegate`,
`workflow_run` or `ticket_route` call can supply them.

**Depth** counts Agents, not tool kinds. The profile's own Agent (the
interactive agent, or the superagent router) is depth 0. Every worker it
starts is depth 1, whether `delegate`, `workflow_run` or `ticket_route`
started it. With the default `max_depth: 1`, a depth-1 worker has no
delegation tools, so superagent executors cannot subdelegate. With
`max_depth: 2`, a depth-1 worker keeps them, and its depth-2 workers inherit
from its own batch snapshot and have none.

### 4.4 Workspace modes

| Mode | Directory | Writes | Merge-back |
| --- | --- | --- | --- |
| `shared` (default) | Parent workspace | Allowed, but a worker whose envelope holds any write-capable tool (`write_file`, `edit_file`, `patch_file`, `run_shell`, or any `high_risk` tool) holds the run's **single writer slot** for its lifetime. Readers run in parallel | None needed |
| `frozen` | `<workspace>/.lingcore/runs/<run>/input/`, one per run, shared by all of its frozen workers | None. The envelope must hold no write-capable and no network tool, or the role/spec fails validation | None. Kept with the run so resume can reuse results ([§5.4](#54-running-and-resuming)) |
| `snapshot` | `<workspace>/.lingcore/runs/<run>/<worker>/` (already excluded from search/knowledge) | Free, isolated | Diff saved as `<…>/<worker>.patch` and listed in the report. The **parent** reviews and applies it (`patch_file` / `git apply`). Never auto-merged |
| `scratch` | Empty directory under the same root | Free | Artifacts listed in the report |
| `ticket` | `<state>/work/<ticket-name>/`, persistent (L2b only) | Free | Through the ticket's own deliverable (e.g. a PR) |

Building a snapshot: for a Git workspace that passes the `git` tool's
confinement checks, run a host-side `git clone --local --no-hardlinks` (no
alternates, so the `git` tool still accepts it). Run it with a hardened
environment: `GIT_CONFIG_NOSYSTEM=1`, `GIT_CONFIG_GLOBAL=/dev/null`,
`core.hooksPath=/dev/null`, `core.fsmonitor=false`. Then apply the parent's
uncommitted tracked diff (`git diff --binary HEAD`) and commit it as the
snapshot base, so the worker's patch contains only the worker's changes.
Copy untracked, non-ignored files within a byte/count cap and list what was
skipped. A non-Git workspace gets a bounded `confined_directory` copy that
honours `search.exclude_dirs`. Snapshot clones are removed when the run ends
(the patch is enough to rebuild one), unless
`delegation.workspace.keep_snapshots` is set. Patches and scratch directories
are kept until retention prunes the run ([§4.8](#48-lifecycle)). All writes
use the existing no-follow, descriptor-relative primitives.

Building a frozen input: the host captures it once, when the run starts,
with the same hardened clone and bounded untracked copy.

- The parent's uncommitted tracked diff is applied to the working tree but
  **not committed**, so `git diff HEAD` inside the copy shows the change
  under review. The staged/unstaged split is not reproduced, and the
  workspace note in each brief says so and lists skipped files.
- The host records an **input revision**: `sha256` over the base commit id
  and a sorted manifest of `(path, mode, content sha256)` for every working
  file, plus the skipped list. A non-Git workspace uses the confined copy and
  the same manifest.
- The directory is made read-only and is kept until retention prunes the
  run. `keep_snapshots` does not apply to it. It counts toward
  `snapshot_max_bytes` and `snapshot_max_files`.

### 4.5 Confirmation broker

Workers have no frontend. The broker wraps the parent's `confirm`:

- **Serialized:** one prompt at a time per Supervisor (an `asyncio.Lock`),
  so ten parallel workers cannot flood the user.
- **Labelled:** `"[worker reviewer#2 · run r-1a2b] " + original prompt`.
  Frontends that track the turn's `run_shell` commands (invariant 9) will not
  find a worker's command in the parent stream. They therefore use generic
  framing and **never offer "always allow"**, because a worker's allowlist is
  not the parent session's. That is the safe default. A later, optional
  frontend capability `confirm_delegated(origin, prompt)` can show richer UI.
- **Denied** when the parent has no handler, when the run is cancelled, or
  when the run is in the background and the frontend lacks
  `supports_async_confirm`. The CLI cannot show a prompt while its editor owns
  the terminal. Telegram and Lark can, because confirmations there are
  messages.
- A role or a request may make the policy stricter (`deny`), never looser.

### 4.6 Budgets and usage

- `ExecutionBudget` gains an optional `parent` and an `agents` counter.
  Admission checks every ancestor and the child before incrementing any
  counter, as one synchronous operation; a child's rejection must not spend
  a parent's request or agent slot. Usage is charged once per request ID at
  each level, including usage arriving after exhaustion or cancellation.
  A **run budget** is `ExecutionBudget(limits from
  delegation.limits ∩ spec.limits, parent=agent.execution_budget)`. In an
  evaluation, workers therefore spend the trial's limits. Interactively,
  the run budget supplies the limits.
- Workers are assembled with `execution_budget=run_budget.child(role caps)`.
  `InstrumentedClient` calls `ExecutionBudget.reserve_request()`, which
  reserves a **request count**, not tokens. Dispatch already reserves tool
  calls. Agent, request and tool counts are hard admission caps; wall time
  uses the driver's deadline.
- **v1 `reported_tokens` is a reported-usage stopping threshold.** Admission
  checks previously reported input + output tokens; `record_usage` charges
  them after the request. A single request can cross the threshold, and
  concurrent requests admitted below it can all finish above it. Subsequent
  requests are refused once the shared tally reaches the threshold. In-flight
  requests remain billable, even if cancelled; their usage is still recorded.
  No maximum overshoot is promised without enforceable per-request bounds.
  `budget.remaining_tokens` is `max(0, threshold - reported_usage)` (the
  minimum across limited ancestors), and does not subtract in-flight spend.
  When no budget in the chain sets `reported_tokens` it is `null`: it renders
  as JSON `null`, and `budget_below` never fires. The validator rejects a
  spec that uses `budget_below` when the run's own effective limits
  (`delegation.limits ∩ spec.limits`) set no `reported_tokens`.
  Confirmations and progress must call this a **stopping threshold**, show
  actual usage even above it, and preserve the missing/partial usage flag.
  `usage_missing: stop` refuses subsequent requests after incomplete usage;
  it cannot prevent already-admitted requests from spending more.
- A **hard token ceiling is deferred**. It would require an atomic
  reservation across the budget chain for a conservative input bound plus
  an enforced output/reasoning limit before each provider attempt, followed
  by reconciliation against reported usage. Retries, cancellation and
  missing usage must retain any unaccounted reservation. Backends unable
  to supply these bounds must refuse hard-ceiling mode. Merely chaining
  today's admission checks cannot provide that guarantee.
- **Delegated usage is recorded separately from the agent's own usage.**
  `TurnStats.usage` continues to mean "requests this agent made", including
  its optional wording pass. The existing `TokenUsage.purpose` and
  `UsageReported.purpose` labels distinguish `output_rewrite` from `main`, and
  the label survives session replay. CLI, LingChat and AgentGUI include rewrite
  spend in totals but exclude it from the main context gauge. Missing labels
  on legacy records mean `main`. These own-agent labels do not carry worker
  identity or run lineage: delegation must not reuse the rewrite label or put
  workers into that own-agent channel. A worker's smaller context must never
  replace its parent's gauge.
  - A new record `DelegatedUsage(run_id, worker_id, role, usage: TokenUsage)`
    flows through a second channel on the parent's `UsageMeter`
    (`record_delegated` / `drain_delegated`, with its own per-turn tally).
  - **Foreground runs:** the loop drains that channel as a new event,
    `DelegatedUsageReported(records)`, before the delegating call's
    `ToolResultEvent`. `TurnStats` gains `delegated: tuple[DelegatedUsage, ...]`
    next to the unchanged `usage`, so throughput (`usage` output over
    `model_ms`) needs no exclusion rule. `turn_stats_payload` writes a
    `delegated` list only when it is non-empty, and an absent list reads back
    as `()`, so existing payloads stay valid.
  - **Background runs** are billed by no parent turn. Their usage stays on the
    run (per worker in `run_nodes`) and in each worker session's own
    `turn_stats`. The completion notice carries the run's total.
  - Frontends show the two figures apart. The CLI status bar adds a
    `delegated ↑… ↓…` item, and the turn footer lists it after the turn's own
    spend. A frontend that ignores the new event under-reports spend but never
    mis-reports context.
  - The budget chain still charges both against an evaluation trial's limits.
- `send_prompt_cache_key` uses `run_id:role` for workers, so siblings with
  identical role prefixes reach the same warm cache node.

### 4.7 Structured results: `submit_result`

Every worker gets one extra runtime tool, added through an internal
`Agent.from_profile(extra_tools=…, terminal_tools=…)` parameter. It is not
reachable from config, and it never touches the global `REGISTRY`:

```text
submit_result(status: "done" | "blocked",
              result: <compiled output schema> | string,
              question: string | null,     # required when blocked; one concrete question
              summary: string ≤ 600)
```

- **Schema subset → pydantic**, compiled once at spec load
  (`delegation/schema.py`). Allowed: `object / array / string / integer /
  number / boolean / enum / null`, `required`, `properties`, `items`,
  `maxItems`, `maxLength`, `minimum/maximum`, `description`. Depth ≤ 5,
  ≤ 64 properties in total, every array capped (`maxItems` or a default
  of 200). No `$ref`, `patternProperties` or regex `pattern`. This avoids a
  new dependency (there is no `jsonschema` in `pyproject.toml`) and keeps
  error messages short enough for the model to fix.
- **Validation at the tool layer:** an invalid call returns `ok=False` with
  pydantic's errors, and the model tries again in the same turn.
- **Ending the turn:** a successful terminal tool ends the turn after its
  batch, with `Final(content=<result JSON>)`. This saves one model request
  per worker. `ToolOutput.end_turn` is honoured only for tools listed in the
  Agent's internal `terminal_tools`. Plugin tools cannot end turns.
- **No submission:** with a schema, one nudge turn ("call submit_result
  now"), then `failed(cause="no_result")`. Without a schema, the final text
  becomes `result`.

```python
class WorkerReport(BaseModel):  # what the delegator sees
    worker_id: str
    role: str
    label: str
    status: Literal[
        "completed", "blocked", "failed", "cancelled", "budget_exceeded", "timed_out"
    ]
    result: Any | None  # validated object or text, bounded
    summary: str  # ≤ 600 chars
    question: str | None  # blocked → one concrete question
    artifacts: list[str]  # workspace-relative: patches, offloaded output
    usage: UsageSummary
    session_id: str | None
    error: str | None
```

A large `result` is offloaded like tool output (`_offload.py`). The report
carries a preview and a path.

### 4.8 Lifecycle

- **Driving a worker:** generalize `run_headless(agent, frontend=…)` and
  reuse its deadline, Stop handshake, bounded cleanup and `ExecutionOutcome`
  typing. A `WorkerFrontend` supplies the brief, at most one nudge, and the
  broker's `confirm`. Outcome → report status is a fixed table.
- **Foreground Stop:** the parent's Stop cancels the delegating tool task. The
  Supervisor then cancels, awaits and finalizes each worker
  (`cancel_turn` → await → `finalize_cancelled_turn`), and closes each one
  with a 5 s bound. The pattern is the same as `run_headless` today, and no
  task or lease leaks.
- **Background runs** belong to the Supervisor, not to the turn task:
  - When the *starting* turn rolls back, `turn_rolled_back(started_at)`
    cancels the run and discards its queued delivery (the process-plugin
    rule).
  - `workflow_stop` cancels a run explicitly.
  - `Agent.aclose` marks running runs `interrupted` in the journal (resumable)
    and reaps their workers.
- **Transient provider failures** inside a worker already go through the
  loop's `stream_retries`. Above that, a node may set
  `retry_on: [provider_transient]` (≤ 2). Agent-influenced failures (budget,
  timeout, invalid result) are **never** retried, the same rule as
  `evaluation/jobs.py`. An effecting node ([§5.4](#54-running-and-resuming))
  cannot set `retry_on`, because a retry would repeat its effects. The
  validator rejects it.
- **Tracing:** each worker turn runs under a `worker` span with
  `TraceLink(relation="delegate")`, a new literal at
  `lingcore/tracing/models.py:127`. A workflow becomes one trace tree.
- **Persistence:** worker transcripts are sessions in the parent's
  `sessions.db`, with `meta.delegation = {parent_session_id, anchor_seq,
  run_id, node_key, role, envelope}`. They are hidden from `--list-sessions`
  (shown with `--all`), listed by `lingcore run show`, and can be inspected
  with `trace show`. Each worker holds its own session lease. With sessions
  disabled, runs are in-memory and can be neither resumed nor continued.

**Workers are kept, not rolled back with the parent.** A worker's transcript,
its run journal and its artifacts record work that really happened, since its
side effects are not undone either. They are useful for debugging, and they
are what a later continuation needs.

- **Detaching:** a parent Stop or Edit that truncates past a run's anchor
  cancels the run if it is live, then marks it `detached` with the reason and
  the original anchor sequence. Nothing is deleted. The parent's transcript
  no longer refers to the run, but `workflow_status`, `lingcore run list` and
  LingChat still list it.
- **Forks:** `fork_session` does not copy runs. A fork still finds the
  source's runs through its lineage.
- **Continuing a worker:** `delegate(continue_worker=<worker_id>, prompt=…)`
  sends one more brief to an existing, idle worker session, like Claude Code's
  SendMessage to a finished agent.
  - The envelope is recomputed as *recorded envelope ∩ current role ∩
    current batch snapshot*. It can only narrow if the profile, the role or
    the parent's authorization changed since. It is refused if the role no
    longer exists or the worker's session lease is held.
  - The workspace depends on the mode. `shared` reuses the parent workspace
    and `scratch` reuses its kept directory. `snapshot` takes a fresh
    snapshot of the parent's current workspace and reapplies the worker's
    last patch with `git apply --3way`. If the patch does not apply, the
    worker continues on the fresh snapshot with a note naming the old patch
    artifact.
- **Operator continuation:** `lingcore run continue <worker-id>` opens a
  kept worker in the interactive CLI, so confirmations go to the person.
  The same envelope rule applies. Outside a turn there is no batch snapshot,
  so the profile's initial tool set plus static skills takes its place.
- **Retention:** `delegation.retention` (`max_age_days: 30`,
  `max_runs: 500`) prunes the oldest finished or detached runs at Supervisor
  start-up and through `lingcore run prune`. A pruned run's worker sessions,
  kept directories and artifacts go with it. Pruning reports what it removed,
  and `doctor` shows the count and size of retained runs. Live and
  interrupted runs are never pruned.

### 4.9 The `delegate` tool

```text
delegate(prompt: str,                    # task; the original user request is appended verbatim
         role: str = "general",
         continue_worker: str | None,    # continue a kept, idle worker instead (§4.8)
         label: str | None,
         output: str | object | None,    # schema name from the profile, or an inline subset schema
         workspace: "shared"|"frozen"|"snapshot"|"scratch" | None,   # may only narrow the role
         tools: list[str] | None,        # may only narrow the envelope
         background: bool = False)
→ WorkerReport (bounded JSON), or {run_id, status: "started"} when background
```

One worker per call. Parallelism comes from several `delegate` calls in one
batch, which `parallel_tools` already runs together, as with Claude Code's
Agent tool. The Supervisor enforces `max_concurrency`, the writer slot and
`max_workers_per_turn` across sibling calls. `@tool(high_risk=True)` keeps it
behind skill-activation confirmation when a skill grants it.

### 4.10 Core seams required

Each seam is small, general, and comes with its own test:

| Seam | Where | Why |
| --- | --- | --- |
| `ExecutionBudget(parent=…)`, `reserve_agent()` | `execution.py` | Atomic count admission and reported-usage accounting shared down the tree (§4.6) |
| `run_headless(frontend=…, confirm=…)` | `execution.py:188` | Worker driver with broker and nudge |
| `UsageMeter` delegated channel, `TurnStats.delegated`, `DelegatedUsageReported` event | `usage.py:165`, `events.py` | Record delegated spend apart from the agent's own |
| Dispatch-authorization contextvar | `_agent/dispatch.py:67` | Envelope uses the batch snapshot |
| `extra_tools` / `terminal_tools` assembly params, `ToolOutput.end_turn` | `_agent/profile.py`, `tools/__init__.py`, `_agent/iteration.py` | `submit_result` |
| `ToolContext.progress` sink → transient `ToolProgress` events, drained while a batch runs | `_agent/iteration.py:382` | Live progress for long tools (workflows, also `codex_agent`) |
| `BackgroundInbox` source kinds with per-kind framing | `background.py:97` | Run completion delivery next to monitor events |
| `TraceLink.relation = "delegate"` | `tracing/models.py:127` | Linked trace tree |

The `ToolProgress` change is the only edit to the loop's control flow. The
batch runs in a task while the loop drains the sink. Cancellation reaps the
task exactly as the existing `gather` path does. `ToolProgress` is transient
like `PluginNotice`: never stored, always before the batch's results.

## 5. Layer 2a — workflows

### 5.1 Sources of workflows

| Source | Format | Consent |
| --- | --- | --- |
| Profile `workflows/` directory | YAML spec | `workflow_run` in `tools:` |
| Inline from the model (`workflow_run(spec=…)`) | YAML/JSON spec, the same validator | `workflows.inline: true` (default) + per-run confirmation |
| Plugin `workflows:` manifest key | YAML spec or Python callable | `plugins: [name]` (trusted code) |

A bundled, instruction-only `workflows` skill holds the authoring guide
(grammar, patterns, cost guidance). `workflow_run`'s own description stays
short and lists the named workflows. Progressive disclosure keeps the cached
prefix small.

### 5.2 Spec language

A spec is a document with `name`, `description`, `when_to_use`, `inputs`
(subset schema), `limits` (may only lower `delegation.limits`), `schemas`,
`phases`, `steps` and `result`.

**Step kinds** (each step may carry `id`, `phase`, `when`):

| Kind | Value | Semantics |
| --- | --- | --- |
| `agent` | Validated `result`, or `null` on failure | One worker. `repeat: n` (≤ 5) gives a list of n independent attempts, distinguished by occurrence index |
| `map` | List, in item order, with `null`s removed | **Pipeline:** each item flows through `stages` independently, with no barrier between stages. Inside: `${<as>}` is the original item, `${prev}` the previous stage's value, `${index}` the position. A stage yielding `null` drops the item. `max_items` (default 100, hard 4096); dropped and failed counts are logged |
| `parallel` | Object keyed by child step id | **Barrier.** Use it only when a later step needs everything together |
| `loop` | Accumulated, de-duplicated list | `max_rounds` is required. `until:` any of `dry_rounds: k`, `count_at_least: [expr, n]`, `budget_below: tokens`. `collect: {from, dedupe_by}`. `${acc}` is the accumulator in the body |
| `keep` | `value` or `null` | `if:` predicate, `value:` expression. A filter inside `map` |
| `transform` | Pure result | `flatten`, `dedupe {over, by}`, `filter {over, field, equals}`, `take {over, n}`, `count`, `pick {over, fields}` |
| `log` | none | A progress line, shown to the user |
| `workflow` | The child workflow's `result` | A named workflow, one level deep. It shares the run budget and caps |

**Predicates:** `nonempty: expr`, `equals: [expr, value]`,
`count_at_least: [expr, n]`, `votes: {over, field, equals, at_least}`,
`not: pred`.

**Expressions** are only `${path}` lookups (`a.b[0].c`) over `inputs`,
`request` (the originating user text), earlier step ids, loop/map variables
and `budget` (`remaining_tokens`, `agents_left`). There are no calls, no
arithmetic, no clock and no randomness. A string that is exactly one
reference yields the value. Inside a longer string, non-strings are rendered
as JSON. References to later or unknown steps are load-time errors.

**Static caps at load:** ≤ 64 steps, nesting ≤ 4, `repeat` ≤ 5, known roles,
schemas, phases and model aliases only. The validator also computes an
*upper bound* on agents, which the confirmation shows. Sum sequential and
parallel branches; multiply map cardinality, repeat counts, maximum loop
rounds and allowed worker retries. Use the tighter of a known input/schema
cardinality and `max_items`; count conditional branches as runnable. Expand
named child workflows into the same bound. Reject a spec whose bound exceeds
the effective `max_agents` before confirmation or worker assembly. Runtime
admission remains a backstop, including for trusted Python workflows whose
dynamic agent count cannot be proved statically.

```yaml
name: review-changes
description: Review the uncommitted diff by dimension; keep findings that survive verification.
when_to_use: A thorough review of a non-trivial change.
inputs:
  type: object
  properties:
    focus: {type: string, maxLength: 500}
# The agent bound of this spec is exactly 40. Adding `retry_on` to a node
# multiplies that node's count, so it needs a higher `max_agents`.
limits: {max_agents: 40, max_concurrency: 4, reported_tokens: 600000}
schemas:
  findings:
    type: object
    required: [findings]
    properties:
      findings:
        type: array
        maxItems: 4
        items:
          type: object
          required: [file, line, claim]
          properties:
            file: {type: string}
            line: {type: integer}
            claim: {type: string, maxLength: 800}
  verdict:
    type: object
    required: [refuted, reason]
    properties:
      refuted: {type: boolean}
      reason: {type: string, maxLength: 600}
phases:
  - title: Review
  - title: Verify
  - title: Report
steps:
  - id: reviewed
    phase: Review
    map:
      over: [correctness, security, concurrency]
      as: dimension
      stages:
        - agent:
            role: reviewer
            label: review:${dimension}
            output: findings
            prompt: |
              Review the uncommitted diff for ${dimension} defects only.
              Return at most four findings, choosing the most severe.
              Extra focus from the requester: ${inputs.focus}
        - map:
            over: ${prev.findings}
            as: finding
            max_items: 4
            stages:
              - phase: Verify
                agent:
                  role: skeptic
                  repeat: 3
                  label: verify:${finding.file}
                  output: verdict
                  prompt: |
                    Try to refute this finding. Default to refuted=true if unsure.
                    ${finding}
              - keep:
                  if:
                    votes: {over: "${prev}", field: refuted, equals: false, at_least: 2}
                  value: ${finding}
  - id: confirmed
    transform:
      flatten: ${reviewed}
  - id: report
    phase: Report
    when:
      nonempty: ${confirmed}
    agent:
      role: writer
      prompt: |
        Write a concise review from these verified findings, most severe first:
        ${confirmed}
result:
  findings: ${confirmed}
  report: ${report}
```

A dimension's verification starts as soon as that dimension's review returns.
Nothing waits for the slowest reviewer, which is Claude Code's
pipeline-by-default rule.

The maximum is **3 reviewers + (3 × 4 × 3) skeptics + 1 writer = 40 agents**.
The conditional writer counts toward the bound even when no finding survives.
The schema rejects more than four findings per dimension; this example does
not enable worker retries. Increasing either findings or retries requires a
new bound within the configured cap.

The `reviewer` and `skeptic` roles use `workspace: frozen`, and the `writer`
role has no workspace or network tool. Every worker therefore reads the same
captured diff, and an interrupted run resumes from its journal
([§5.4](#54-running-and-resuming)).

### 5.3 Python workflows (trusted plugins)

```python
from lingcore.workflows import WorkflowAPI, workflow


@workflow(
    name="find-flaky-tests",
    description="Find flaky tests and propose fixes",
    phases=["Scan", "Fix"],
)
async def find_flaky(wf: WorkflowAPI, args: FlakyArgs) -> dict:
    wf.phase("Scan")
    flaky = await wf.agent(
        "Grep CI logs for retry markers.", role="explorer", output=FLAKY
    )
    fixes = await wf.pipeline(
        flaky["tests"] if flaky else [],
        lambda test, i: wf.agent(
            f"Fix flaky test {test}.",
            role="coder",
            workspace="snapshot",
            label=f"fix:{test}",
        ),
    )
    wf.log(f"{sum(f is None for f in fixes)} fixes failed")
    return {"fixes": [f for f in fixes if f]}
```

`WorkflowAPI` mirrors Claude Code: `agent`, `parallel` (barrier; failures →
`None`), `pipeline` (no barrier), `phase`, `log`, `workflow` (one level),
`args`, `budget`. It uses the same caps, journal and envelope rules as YAML.
Python cannot be made deterministic by the runtime. Content-keyed journal
entries alone cannot capture arbitrary Python reads or side effects. In v1,
Python workflows retain the journal for audit and recovery, but do not reuse
cached worker results across resumes. A future reuse contract must declare
and validate dependencies and external revisions under §5.4's rules. Resume
of a Python run reruns its workers. It is refused once the run has started
any effecting worker (§5.4).

### 5.4 Running and resuming

- **Admission:** validate the spec, compile its schemas, compute the
  agent-count bound, then **confirm** through `ctx.confirm` when the bound
  exceeds `delegation.confirm_over_agents` (default 4) or any role grants a
  high-risk tool. The prompt shows: workflow name, ≤ N agents, concurrency,
  reported-token stopping threshold (with possible overshoot), roles → tools,
  workspace modes, foreground or background.
  Without a handler the run is refused (the `activate_skill` rule), and
  evaluation `ScriptedFrontend`s deny.
- **Journal:** `runs` and `run_nodes` rows in `sessions.db` (schema v4).
  Each run row is anchored to the message sequence of the assistant message
  that issued the call. Store the whole spec digest for provenance, **not in
  every node's cache key**. A node key is
  `sha256(key_version, step_identity, node_definition_digest, rendered_brief,
  dependency_values, role_and_context_digest, schema_digest,
  resolved_model_config_digest, input_revision, occurrence)`.
  - `step_identity` uses scoped explicit IDs when provided. Anonymous steps
    use structural paths; ambiguous identity or reordering may cause misses.
    Occurrence includes the enclosing map/loop invocation and repeat index,
    so identical skeptics stay distinct regardless of completion order.
  - The node definition covers that node's semantics and referenced
    definitions, without unrelated sibling prompts. `dependency_values` are
    the **values** of the node's data and control inputs (`when`, map source,
    loop state). Upstream keys and attempt identities are not part of the
    key. Control flow is re-evaluated on resume. Editing a prompt reruns its
    own node. A consumer reruns only when the value it receives changed, so
    an upstream rerun that returns the same value leaves its consumers
    reusable. Independent nodes are unaffected.
  - Role/context identity includes the effective envelope, composed context
    (including project instructions, memory and world when used), static
    skills and relevant tool/plugin versions. Resolve model aliases before
    hashing non-secret configuration. Unknown or unversioned dependencies
    disable reuse; equal prompt text alone is insufficient.
  - `input_revision` is the run's frozen input revision
    ([§4.4](#44-workspace-modes)) for a frozen reader, and empty for every
    other class.
- **Node classes.** The validator derives each node's class from its
  envelope before anything runs, and the journal stores it:

  | Class | Envelope | On resume |
  | --- | --- | --- |
  | **pure** | No workspace, network or profile-state tool | Reused when the key matches |
  | **frozen reader** | `workspace: frozen` (which already forbids write-capable and network tools) | Reused when the key matches and the kept input still verifies against its recorded revision |
  | **live reader** | Reads a `shared`, `scratch`, `ticket` or `snapshot` workspace, or holds a network tool (`fetch_url`, `search_web`, `browser_*`), with no write-capable tool | Never reused. It runs again, which is safe because it has no effects |
  | **effecting** | Holds a write-capable tool (the writer-slot predicate of [§4.4](#44-workspace-modes)) or a profile-state writer (`memory`, `world`, `calendar_*`) | **Never run again** once it has started (see Resume) |

  A Git `HEAD` or a directory path never counts as an input revision. Only
  the frozen input does, because it is the exact, read-only tree the worker
  saw.
- **Frozen input on resume:** a run with frozen readers captures one input
  when it starts and keeps it with the run. `workflow_resume` recomputes the
  manifest digest and compares it with the recorded revision.
  - Resume always continues on that pinned input, for reused and live nodes
    alike, so one run never mixes two workspace states. The result carries
    `input_revision` and its capture time, and says so when the parent
    workspace has changed since the capture.
  - Reviewing the current workspace state is a new `workflow_run`.
  - If the kept input is missing or no longer matches, the run cannot be
    resumed. The refusal names the run and suggests a new one.
- **Resume** (`workflow_resume(run_id)` / `lingcore run resume`): revalidate
  the spec, authorization and limits, then show the same confirmation as
  `workflow_run`, with the number of nodes that will run live. Reused nodes
  charge nothing. Live nodes charge the run budget, which starts from the
  usage already recorded for the run.
  - **Pure nodes and frozen readers** return the stored result when they
    completed, the key is unchanged and the result artifact is intact.
    Otherwise they run live.
  - **Live readers** always run live.
  - **Effecting nodes are never executed twice in one run.** A completed one
    whose key is unchanged returns its recorded result, labelled `recorded`.
    For a `snapshot` writer that result is its patch artifact. The recorded
    result states what happened; it is not a prediction of what a rerun
    would do.
  - An effecting node is **blocked** instead of run when it started without
    reaching a terminal journal row, or when it completed but its key has
    changed (an edited prompt, a changed role, or a different value from a
    live reader that ran again). The block names the node, the reason and
    the earlier attempt's worker session and artifacts. Its consumers are
    skipped and counted, independent nodes still run, and the run ends
    `blocked`.
  - No argument overrides a block. Only a new `workflow_run`, with its own
    confirmation, can execute that step again. Resume therefore never
    repeats an external effect. It still gives no exactly-once guarantee
    for an effect that was interrupted midway: the journal reports which
    node may have acted partially.
- **Stop/Edit:** truncating the parent branch past a run's anchor cancels a
  live run and marks it `detached`. Its journal, worker sessions and
  artifacts are kept ([§4.8](#48-lifecycle)). `workflow_resume` also accepts
  cancelled and detached runs. After an Edit, the regenerated turn can rerun
  the same workflow and reuse eligible unchanged nodes under the rules above.
  External side effects are not undone, as today.

### 5.5 Foreground, background and progress

- **Foreground** (default): `workflow_run` blocks the parent's tool batch and
  streams `ToolProgress(call_id, RunProgress)` events. A run snapshot has
  phase, done/total/failed per phase, the latest node label and status, and
  tokens spent. The CLI renders a transient progress tree in its `Live`
  region. Telegram and Lark edit one status message.
- **Background** (`background: true`, only when the frontend has
  `supports_background` and `delegation.background` is on): the tool returns
  `{run_id, status: "started"}` at once. On completion the run publishes one
  bounded event into `BackgroundInbox` with kind `run`, for example
  `{"run": "r-1a2b", "workflow": "review-changes", "status": "completed",
  "summary": "…", "read": "workflow_result r-1a2b"}`. The frontend may start a
  delivery turn, framed as untrusted output that grants no authorization
  (invariant 18's monitor framing). A user Stop pauses delivery as it does
  today.
- The model never waits. There is no blocking-wait tool, and
  `workflow_status` is a non-blocking read. The tool description says
  results arrive by themselves, the DSH "main session does not wait" rule.

### 5.6 Model-facing surface

| Tool | Purpose |
| --- | --- |
| `workflow_run(name \| spec, args, background)` | Validate, confirm, run |
| `workflow_status(run_id?)` | Non-blocking: runs, phases, counts, spend |
| `workflow_result(run_id, node?, offset?)` | Paged result / node outputs |
| `workflow_stop(run_id)` | Cancel a run |
| `workflow_resume(run_id)` | Resume an interrupted run from the journal |

CLI additions: `/workflows` (live runs), `lingcore run list | show | resume`.
`lingcore doctor` validates `workflows/` specs offline.

## 6. Layer 2b — the superagent

A long-running LingCore service acts as a team member on Lark/Feishu (the
channel added in `60a08b6`). **v1 supports Lark only.** Telegram, scheduled
sources and GitHub come later as additional Sources over the same core. It applies
DSH's domain model with fewer concepts. Above all, authority becomes a
concrete tool envelope rather than only prompt text.

### 6.1 Domain model

| Concept | Definition | Rules |
| --- | --- | --- |
| **Source** | An event stream. v1: Lark. Later: Telegram, `schedule`, GitHub (plugin) | Holds its read position. Drops the bot's own messages. Emits Triggers with actor, `addressed` (DM or mention), thread key and reply address |
| **Trigger** | One immutable event | Idempotent by `(source, external_id)` |
| **Thread** | Triggers in order on one conversation | One Eval at a time. Holds Subscriptions |
| **Policy** | Versioned: per-source allowlists and bot lists, `accept_mode` (`off / observe / auto`), rules | Changed only by operator config or the panel, never by text. The version is a hash |
| **Gate** | Deterministic strength: `explicit` / `inferred` / not admitted | Code only |
| **Eval** | The router's decision on a batch: `accept / attach / stop / ignore` | Timeout → requeue twice → `unavailable` |
| **Mandate** | Immutable grant: policy version, strength, requester, original trigger, matched rules, **role, inherited tool set and envelope digest**. A person's approval of extra tools is a separate immutable record attached to it | Created by the host on accept. Withdrawn only by a person's stop; current policy can deny execution. Revalidated before launch and resume |
| **Ticket** | A unit of authorized work | `open → ended`, and back to open on a person's follow-up (unless a person stopped it) |
| **Assignment** | A Ticket bound to one executor worker | `pending → running → ended`. A Ticket has several over time |
| **Report** | Executor state: `progress / waiting / ready / blocked / failed / dropped` + `question` | Only the final report ends an Assignment |
| **Subscription** | Thread → Ticket fan-out of later messages as context Inputs | Grants no authority |
| **Channel** | The write side of a Thread, through a durable outbox | Ordered per (ticket, conversation) |

**Strength selects the role**, and the host enforces it. An explicit Mandate
assigns `executor.role`. An inferred (proactive) Mandate assigns
`executor.proactive_role`, which is read-only by default, unless the matched
rule names a role. DSH could only narrow proactive authority in prompt text.
Here it is a smaller tool ceiling.

```mermaid
flowchart LR
  S[Source] -->|Trigger| Th[Thread]
  Th -->|batch| Gt[Gate] -->|candidate card| R[router agent]
  Th -->|Subscription fan-out| K[Ticket]
  R -->|accept + Mandate / attach / stop / ignore| K
  K -->|Assignment| X[executor worker<br/>persistent session + ticket dir]
  X -->|ticket_report| K
  K -->|outbox| C[Channel]
```

### 6.2 Router and executor roles

- **Router** (`roles/router.yaml`): ceiling `ticket_route` (1–20 decisions per
  call, each processed independently), `ticket_read`, `ticket_reply`
  (purpose `ack / answer / result / attention`), `ticket_note`,
  `ticket_retry`, `ticket_status`. The default role has **no** file, shell or
  web tools, so "the router does not investigate" holds by construction
  until an operator adds some (see below). It is driven
  by a headless frontend whose turns are relays (candidate cards, reports,
  follow-up notices) delivered as `BackgroundInput`.
- **Executor** (`roles/executor.yaml`): the role's work tools plus
  `ticket_report`, `ticket_ack`, `ticket_handback`, `ticket_follow`,
  `ticket_unrelated` and `ticket_read`. It never gets tools that ask a person
  directly or change the profile. Questions go through
  `ticket_report(question=…)`, and the router forwards them unchanged.
- Each brief carries the original Trigger card, not the router's rewrite. A
  routing note is ≤ 30 words and adds no requirements (DSH `dispatch.md`).

**The router is the service's depth-0 Agent, and executors are its
subagents.** The router is an ordinary `Agent` built from the superagent
profile with the router role's persona. Its authority is the profile's own:
the `tools:` ceiling, the initial set and static skills, restricted to the
router role's tool list. Operator config alone defines it, and no Mandate is
involved. It owns the Supervisor and the budget root, and `Agent.aclose` at
service shutdown closes its workers.

**Ticket tools are host-injected.** The composition root registers the
`ticket_*` tools through `extra_tools`: the router set on the router Agent,
the executor set on every executor. They are not in the profile's `tools:`,
are never inherited, can never be requested as extras, and do not exist
outside `lingcore superagent`. An executor always receives its set,
whatever the envelope arithmetic below yields.

**Executors inherit by default.** On accept, host code validates the
candidate against current Policy, selects the role by Mandate strength and
matched rule, and launches the executor through the ordinary envelope rule
([§4.3](#43-envelope-computation)).

- The inherited tools are `role.tools ∩ the ticket_route batch snapshot`.
  The host records that set and the envelope digest in the Mandate. Router
  notes and proposed arguments cannot enlarge it.
- The executor inherits context like any worker: guardrail, hook plugins,
  `tool_options` (sandbox, network policy), the environment scope, budget
  ancestry and the role's declared context layers.
- It does not inherit the router's conversation. Its brief carries the
  original Trigger card.

**Extra tools need a person's approval.** The extras a role wants are
`role.tools − inherited`, limited to the profile's `tools:` ceiling and
current policy restrictions. A tool outside the ceiling cannot be approved
at all.

- While extras are wanted and unapproved, the Assignment stays `pending` in
  `awaiting_approval` and holds no slot. The host posts an approval card
  through the Channel to `policy.approvers`. The card names the requester,
  the Trigger, the role and the exact extra tools, with high-risk ones
  marked.
- An approval is a structured action (a card button, later the panel) from
  an identity in `policy.approvers`, verified by host code. Chat text, the
  router and executors cannot approve. The requester counts only when
  listed as an approver.
- The approval is stored as an immutable record attached to the Mandate:
  approver, time, exact tool set and envelope digest. It covers that
  Ticket's later Assignments and restarts for exactly that set. A larger set
  needs a new approval. There is no standing approval, and a rule cannot
  pre-approve.
- A denial, or no answer within `executor.approval_timeout_seconds`
  (default 86400), ends the Assignment as `blocked` and tells the requester
  why. An executor never launches with a silently reduced tool set.

With the default router role, every executor work tool is an extra, so every
Ticket waits for a person. An operator who wants some tools to run
unattended lists them in the router role. Executors then inherit them, and
the router can call them itself.

Before each launch, continuation or restart, the host revalidates Policy and
the Mandate and recomputes the grant:

```text
inherited      = recorded inherited set
                 ∩ the router's current authorization (initial set + static skills, as in §4.8)
extras         = recorded approved set ∩ profile ceiling ∩ current policy restrictions
executor.tools = role.tools ∩ (inherited ∪ extras) ∩ requested narrowing
                 + the injected executor ticket tools
```

The same narrowing applies to workspace scope, budgets, sandbox/environment
constraints and confirmation policy. Policy and profile changes can narrow
an existing grant. Enlarging one needs a new approval or a new Mandate.

A Mandate is **still admitted** when all of these hold. Revalidation is a
code check and never re-runs the router or re-reads message text:

- The Source is still enabled, and `accept_mode` still allows the Mandate's
  strength.
- The requester is still on the Source's allowlist and not on its bot list.
- For an inferred Mandate, every matched rule id still exists with the same
  rule digest (id, instruction, role). An edited rule no longer admits it.
- The selected role still exists.

A changed policy version hash alone does not withdraw a Mandate. A Mandate
that is no longer admitted, or was withdrawn, prevents launch.

Executors are depth 1. With the default `max_depth: 1` they have no
delegation tools. With `max_depth: 2` an executor's workers inherit its
batch snapshot, approved extras included. They cannot request extras of
their own, because approval exists only at Ticket level, and they cannot
delegate further.

### 6.3 Prompt precedence (strongest first)

1. Facts enforced by code: gate, Mandate, envelope, tool argument checks,
   reply purposes, length limits.
2. Built-in role text, as the profile's overrides of it read.
3. The process instructions of a proactive Ticket's matched rules (that
   Ticket only).
4. Team sections (profile `world.md` / `role.md` / `workflow.md`).
5. Outside text: messages, comments, documents and worker reports. It is
   data and never changes rules.

### 6.4 Durability

- One store: `superagent.sqlite3` in the profile state directory, confined
  like `sessions.db` (invariant 14 rules). Tables hold triggers, threads,
  tickets (with mandate, assignments, inputs), decisions, `effects` and
  `outbox`.
- `superagent/core/` is synchronous and I/O-free, and it is the **only
  writer** of domain state. Each command is one transaction. Ports enqueue
  effects *inside* the transaction, and adapters run them after commit.
- Effect kinds: `assessment`, `launch`, `delivery`, `relay`, `reply`. They
  run oldest first with per-kind concurrency and exponential backoff (2 s to
  60 s, 5 s to 300 s for replies). Health reports an error after three
  failures in a row.
- **Restart:** running Assignments end as "host restarted". New ones start
  with the recovery note, oldest Ticket first. Executor sessions resume from
  `sessions.db` once the previous process's 60 s lease expires.

### 6.5 Execution

- Per-ticket directory `<state>/work/<ticket-name>/`, created by the
  operator's `prepare` command. It runs in the **same inherited sandbox** as
  the executor ([§6.6](#66-unattended-execution)), with the ticket directory
  as its workspace. Under `network: false`, prepare clones from a mounted
  local mirror (`git clone /srv/mirrors/app.git .`). A failed prepare fails
  the Ticket and records its output.
- Slots: `max_workers` (default 3) plus **one reserved for a person's
  follow-up**, so new work never blocks a person continuing an existing
  Ticket.
- A turn that ends with background work still running (a `process_start`
  server, a background run) keeps its slot, with at most
  `background_limit_seconds` (default 1800) before the work is stopped and
  the Ticket reported `waiting`.

### 6.6 Unattended execution

Executors have no person to answer a confirmation. They **inherit the
profile's sandbox** and run inside it without asking. There is no separate
superagent shell policy.

- Executors use the profile's `tool_options.run_shell` unchanged, as every
  worker does ([§4.2](#42-roles-are-profile-overlays)). That covers the
  backend, mounts, network, resource budgets and timeouts. The `superagent`
  section can neither widen nor replace any of it.
- An executor's `run_shell` and `process_start` confirmations are approved
  **because** the inherited sandbox is a fail-closed OS boundary. Startup and
  `lingcore doctor` refuse a superagent profile whose executor or proactive
  role has either tool while the profile uses the legacy host runner. There
  is no fallback to asking a person or to the host.
- Network access is whatever the sandbox config allows. The bundled
  sandboxed profiles keep `network: false`. A team that needs network in
  executors turns it on in the sandbox, which then applies to every executor.
- Every other confirmation is denied, because nobody can answer it. That
  covers a private-host `fetch_url`, high-risk skill activation and browser
  approval. Skills an executor needs are engaged statically in its role.
- Tool approval ([§6.2](#62-router-and-executor-roles)) is not a
  confirmation. It happens before launch, holds no slot, and decides which
  tools the executor has. Once a tool is in the envelope, the rules above
  apply to it without further asking.
- There is no `ask` mode. A confirmation that waits hours for a chat reply
  would hold a slot, so a question becomes `ticket_report(blocked,
  question=…)` instead. Pushing branches or opening PRs belongs to a later
  GitHub plugin tool with its own credential (`ctx.getenv`), not to shell
  access with ambient credentials.

### 6.7 Composition root

`lingcore superagent -p profiles/team --lark-config …` (v1 is Lark-only).
The Lark integration splits into a *transport* (SDK I/O, rendering,
attachment conversion) and its current per-user *runtime*. The superagent
reuses only the transport, as a Source plus a Channel. Telegram gets the same
split when it becomes a Source. `lingcore superagent status` prints health, and a
later LingChat page shows "needs a person" first (DSH's panel principle).

## 7. Configuration reference

```yaml
tools: [..., delegate, workflow_run, workflow_status, workflow_result, workflow_stop]

delegation:
  roles_dir: roles
  default_role: general
  max_depth: 1
  max_workers_per_turn: 8
  confirm_over_agents: 4
  background: true
  limits:                        # per run; specs and roles may only lower them
    max_agents: 50
    max_concurrency: 4
    reported_tokens: 1000000     # stopping threshold; in-flight usage can overshoot
    usage_missing: ignore        # ignore | stop; specs and roles may only tighten it to stop
    model_requests: 2000
    wall_seconds: 3600
  models: {}                     # alias → LLMCfg (no secrets: api_key_env names a variable)
  workspace:
    snapshot_max_bytes: 268435456
    snapshot_max_files: 20000
    keep_snapshots: false
  retention:                     # kept runs, worker sessions and artifacts
    max_age_days: 30
    max_runs: 500

workflows:
  dir: workflows
  inline: true
```

```yaml
# profiles/team/config.yaml (superagent excerpt)
superagent:
  store: superagent.sqlite3
  sources:
    lark: {config: lark.yaml}     # v1: Lark only
  policy:
    accept_mode: observe          # off | observe | auto
    approvers: []                 # identities whose card action approves extra executor tools
    rules:
      - id: reproducible-bug
        instruction: When someone in the team chat reports a bug with steps to reproduce, investigate and propose a fix.
        role: executor
  router: {role: router, eval_timeout_seconds: 300}
  executor:
    role: executor
    proactive_role: observer
    max_workers: 3
    approval_timeout_seconds: 86400   # unanswered extra-tool approval → blocked
    prepare: git clone /srv/mirrors/app.git .   # runs in the inherited sandbox
    background_limit_seconds: 1800
# No shell or sandbox keys here: executors inherit tool_options.run_shell.
```

All new sections use `extra="forbid"`. `delegation.models` entries follow
invariant 4: they name variables and never hold keys.

## 8. Events and persistence changes

Existing baseline: `UsageReported(usage, purpose="main")` and
`TokenUsage.purpose="main"` already support `output_rewrite`; non-default
purposes are saved in each `turn_stats.usage` entry, with no database schema
change and `main` as the default when absent. The table lists future
delegation/workflow/superagent additions; purpose metadata belongs to the
existing wording feature.

| Change | Kind |
| --- | --- |
| `ToolProgress(call_id, payload)` | New transient `AgentEvent`, never stored. Ignored by frontends that don't render it |
| `RunProgress` payload (phases, counts, latest node, spend) | Data carried by `ToolProgress` |
| `DelegatedUsage` record, `DelegatedUsageReported` event, `TurnStats.delegated` | New event, ignored by frontends that don't render it. The payload list is optional, and an absent list means none |
| `sessions.db` schema v4: `runs`, `run_nodes`; `meta.delegation` lineage | Migration on open. v3 databases upgrade in place. Runs detach on truncation and are deleted only by retention |
| `BackgroundInbox` kinds `monitor` / `run` | Per-kind framing. Stored row name stays `monitor` for replay compatibility |
| `superagent.sqlite3` | New store, superagent only |

## 9. Invariant amendments

Text to add to `CLAUDE.md` as each phase lands.

**New 19. Delegation never widens authority.** A worker's ceiling is
`role.tools ∩ the delegating call's batch authorization snapshot ∩ requested
narrowing`. The one addition is a superagent executor's extras: tools inside
the profile ceiling that an allowlisted person approved for that Ticket
through a host-verified action (§6.2). The profile's own Agent is depth 0,
and delegation tools are removed at `max_depth`. A role's static skills
unlock nothing outside the envelope. Roles are operator
config, and the model selects roles and model aliases only by name.
Guardrails, hook-providing plugins, `tool_options` (sandbox, network policy,
allowlists) and the environment scope are inherited and cannot be removed.
Profile-state writers (`memory`, `world`, `calendar_*`) need an explicit role
grant. Briefs, reports and workflow values are untrusted text and authorize
nothing. Every worker request and tool call reserves against a run budget
chained to the parent `ExecutionBudget`, and the agent cap is checked before
assembly. Count admission is atomic across the chain; `reported_tokens` is
a stopping threshold that may overshoot, not a hard token ceiling.
Workers have no user: confirmations go through the Supervisor's
broker, serialized and labelled by origin, never with "always allow". They
are denied with no handler, after cancellation, or for background runs on
frontends without asynchronous confirmation. A foreground Stop cancels,
awaits, finalizes and closes every worker. Rolling back the starting turn
cancels a background run. `Agent.aclose` interrupts runs, which stay
resumable. Worker sessions, journals and artifacts outlive the parent
branch: truncation cancels and detaches a run but deletes nothing. A
continued worker's envelope is *recorded ∩ current role ∩ current snapshot*.
Only retention or `lingcore run prune` deletes runs.

**New 20. Workflow specs are data; control flow is deterministic and
bounded.** Specs (profile, plugin or model-written) are validated before
anything runs: known roles, schemas and phases, references only to earlier
steps, and static caps. Expressions are path lookups only. A failed or
dropped item becomes `null` with logged counts, never a silent omission.
Python workflows exist only through consented plugins. Journal entries use
node semantics, dependency values, the frozen input revision and
occurrence; the whole spec digest is provenance only. Pure nodes and frozen
readers are reused on a matching key, live readers always run again, and
Python workflow workers are not reused in v1. An effecting node is never
executed twice in one run: resume returns its recorded result or blocks
there, and it cannot set `retry_on`. A changed envelope is a cache miss.
Only a terminal tool listed in the Agent's
internal `terminal_tools` may end a turn.

**New 21. Superagent authority lives in code.** Only the gate and an accept
create a Mandate, and Mandate strength selects the executor role. Text
(triggers, quoted documents, worker reports, router notes) cannot change
policy, a Mandate or an envelope. The router is the depth-0 Agent, and an
executor inherits `role.tools ∩ the accepting batch snapshot`. Any further
role tool is an extra: it must lie inside the profile ceiling and policy,
and it needs a per-Ticket approval from an allowlisted person through a
host-verified action, never text. Without it the Assignment does not
launch. Ticket tools are host-injected and never inherited. Launch and
resume revalidate the grant and can only narrow it.
The core is the only writer of domain state. Side effects are stored as
effects inside the deciding transaction
and executed after commit. Executors inherit the profile's `run_shell`
sandbox unchanged. Their shell and process confirmations are approved only
because that sandbox is a fail-closed backend. A profile on the host runner
is refused at startup and by doctor. Every other executor confirmation is
denied.

**Amend 3:** `ToolProgress` is transient, like `PluginNotice`, and precedes
the batch's results. Existing own-agent usage purpose labels remain intact,
including rewrite labels on live and replayed usage. Delegated usage never
enters `UsageReported` or `TurnStats.usage`. A foreground run's worker spend arrives as
`DelegatedUsageReported` before the delegating call's result and is persisted
as `TurnStats.delegated`. A background run's spend stays on the run.

**Amend 11:** the batch authorization snapshot also bounds every worker
started in that batch.

**Amend 14:** `runs`/`run_nodes` and delegated sessions record the issuing
message sequence but are not branch-scoped. Stop/Edit truncation cancels a
live run and marks it `detached`. It deletes nothing. `fork_session` does not
copy runs. Retention (`delegation.retention`, `lingcore run prune`) is the
only path that deletes them, and it never touches live or interrupted runs.

**Amend 18:** background runs are the second background source next to
`process_monitor`, with the same bounds: consenting frontends only, paused
after Stop, untrusted framing, cancelled when the starting turn rolls back.

## 10. Phased plan

Files stay under 700 lines. The public entry points (`agent.py`,
`execution.py`, `background.py`) keep their names and delegate inward.

| Phase | Delivers | Main files | Exit criteria |
| --- | --- | --- | --- |
| **0. Seams** | §4.10 table, no user-visible feature | `execution.py`, `usage.py`, `_agent/dispatch.py`, `_agent/iteration.py`, `background.py`, `tracing/models.py` | Each seam unit-tested. The full suite, ruff and mypy are green in LingCore and LingChat |
| **1. Kernel + `delegate`** | Roles, envelope, Supervisor, broker, `submit_result`, `shared`/`scratch` workspaces, foreground only, in-memory journal, delegated sessions | `delegation/{roles,envelope,supervisor,worker,schema,broker}.py`, `tools/builtin/delegate.py` | Authorization, cancellation, budget and broker tests below. A `profiles/coding/roles/` example. Doctor checks |
| **2. Isolation, background, kept workers** | `snapshot` + patch artifacts, background `delegate`, sessions schema v4, detach on Stop/Edit, `continue_worker`, `lingcore run continue`, retention | `delegation/{workspaces,journal,retention}.py`, `_sessions/runs.py` | Snapshot never yields a linked worktree. Rollback cancels background runs. Stop/Edit detaches without deleting. Continuation never widens the envelope. Migration test from v3 |
| **3. Workflows** | Spec + interpreter, `workflow_*` tools, progress, resume, plugin Python workflows, `workflows` skill, CLI `/workflows`, `lingcore run …` | `workflows/{spec,expressions,interpreter,api,catalog}.py`, `tools/builtin/workflow.py`, `io/_cli/` | Pipeline-ordering, determinism, resume-cache and cap tests. `review-changes` example passes on fakes |
| **4. Superagent (Lark)** | Core engine, store, effects, outbox, Lark source/channel, router/executor tools, inherited-sandbox execution, recovery | `superagent/core/*`, `superagent/{store,effects,outbox,passes,runtime}.py`, `superagent/sources/lark.py` | Domain tests on in-memory repositories (DSH `core-engine.spec` style), restart recovery, slot reservation, host-runner profile refused, a model-surface budget |
| **5. More sources** | Telegram source/channel, `schedule` source, GitHub source plugin, LingChat "needs a person" panel | `superagent/sources/*`, `integrations/telegram/` | Each source passes the shared Source contract tests |

## 11. Testing strategy

- **Fakes:** a `FakeLLMFactory` keyed by role/label hands each worker its own
  scripted `FakeLLMClient` (`tests/fakes.py`), so tests stay offline and
  deterministic. Fixtures go in `tests/support/`, and tests are grouped by
  feature (`test_delegation_envelope.py`, `test_workflow_interpreter.py`, …).
- **Authorization:** role tools outside the ceiling → `ConfigError`. A
  parent's skill-gated, inactive tool is not inherited. A sibling
  `activate_skill` cannot widen a worker. A worker calling a tool outside its
  envelope gets the unknown-tool error. Delegation tools are absent at depth.
  A role's static skill unlocks nothing outside the envelope. An executor
  inherits exactly `role.tools ∩ the router's snapshot` plus its injected
  ticket tools. A role wanting more waits in `awaiting_approval` without a
  slot, launches with the extras only after a listed approver's card action,
  and ends `blocked` on denial or timeout. Chat text, the router, a
  non-approver and a forged tool argument cannot approve. A tool outside the
  profile ceiling is never approvable. An approval does not carry to another
  Ticket or to a larger set. Policy narrowing survives restart, and policy
  widening never enlarges an existing grant. An edited matched rule stops
  admitting its inferred Mandate. Executors have no delegation tools at
  `max_depth: 1`; at 2 their workers inherit and cannot request extras.
- **Cancellation:** Stop during a 10-worker fan-out leaves no running task,
  no held lease and no snapshot clone. Worker sessions and the run journal
  remain, marked `detached`. `finalize_cancelled_turn` returns
  `TurnCancelled`.
- **Kept workers:** a continued worker gets a narrower envelope after its
  role loses a tool. A snapshot worker's patch is reapplied, and a
  conflicting patch produces the note. Retention prunes only finished or
  detached runs and reports what it removed.
- **Usage:** delegated spend never changes the CLI context gauge or
  `TurnStats.usage`, and `turn_stats` payloads without `delegated` still
  load. Tests cover both `main` and `output_rewrite` own-agent usage and legacy
  entries without purpose labels; worker spend never updates a parent's context.
- **Unattended execution:** a superagent profile on the host runner is
  refused. Executor shell confirmations are approved under Bubblewrap, and a
  private-host `fetch_url` confirmation is denied.
- **Budget:** worker spend exhausts the parent's evaluation budget with
  `budget_exceeded`. `max_agents` is enforced before assembly.
  A child admission failure leaves ancestor counters unchanged. Admit two
  requests below `reported_tokens`, complete both above the threshold and
  assert that all usage is recorded and the next request is refused. Cover
  single-request overshoot, duplicate usage IDs, late usage after cancellation
  and `usage_missing: stop`. `remaining_tokens` clamps to zero and
  `budget_below` stops a loop; neither promises unspent in-flight capacity.
- **Broker:** prompts are serialized and labelled, never offer always-allow,
  and are denied in background on a CLI-like frontend.
- **Interpreter:** `map` lets item A reach stage 3 while item B is in stage 1.
  `parallel` is a barrier. A `null` drops an item and is counted. Output is
  identical across runs with shuffled completion order. Static validation
  rejects forward references and over-cap specs. The documented example
  admits exactly 40 agents at its upper bound; the former 20-findings variant
  computes 184 and is rejected before any worker starts. Worker retries and
  child workflows count toward the bound.
- **Resume:** pure nodes and frozen readers hit the cache on a verified
  frozen input, so the `review-changes` example resumes without rerunning
  completed reviewers. Editing one prompt reruns that node; a consumer hits
  when the rerun returns the same value and reruns when it differs.
  Changing a role or resolved model configuration invalidates its nodes.
  Live readers always run again, even with identical `HEAD`, spec and text.
  A missing or altered frozen input refuses the resume. Shuffled completion
  order preserves occurrence identity. An effecting node is never executed
  twice: completed with an unchanged key it returns the recorded result,
  and interrupted or with a changed key it blocks, its consumers are
  skipped and counted, and the run ends `blocked`. `retry_on` on an
  effecting node and `budget_below` without a token limit fail validation.
  Resume shows the confirmation again. Python workflows do not replay
  cached results in v1 and cannot resume after an effecting worker started.
- **Model surface:** one test bounds the total characters of every new tool
  description, role persona and skill shipped by default (DSH
  `model-surface.spec`), so the cached prefix cannot grow unnoticed.

## 12. Decisions

Initial decisions recorded 2026-11-10; review corrections 7–11 recorded
2026-11-14. The sections above reflect both.

| # | Question | Decision | Where it shows |
| --- | --- | --- | --- |
| 1 | Builtin tools or a bundled plugin? | **Builtins** (`tools/builtin/delegate.py`, `tools/builtin/workflow.py`), gated by `tools:` and `high_risk=True`. They keep plain names such as `delegate` instead of plugin-prefixed ones | §4.9, §5.6 |
| 2 | Inline, model-written specs? | **Allowed, on by default** (`workflows.inline: true`) for flexibility. The envelope bounds authority, and the run confirmation bounds cost. Operators can turn inline specs off | §5.1, §5.4 |
| 3 | Worker sessions after Stop/Edit? | **Kept** for debugging and later continuation. Truncation cancels and detaches a run. `continue_worker` and `lingcore run continue` pick a worker up again, and retention is the only deletion path | §4.8, §5.4, amendments 14 and 19 |
| 4 | Delegated usage in `TurnStats`? | **Recorded separately**: `DelegatedUsageReported` and `TurnStats.delegated`. `usage` and throughput stay the agent's own | §4.6, §8, amendment 3 |
| 5 | Unattended shell for executors? | **Inherit the profile's sandbox.** There is no separate policy key. Shell and process confirmations are approved only under a fail-closed backend, the host runner is refused, and network follows the sandbox config | §6.5, §6.6, invariant 21 |
| 6 | Superagent scope for v1? | **Lark only.** Telegram, schedule, GitHub and the LingChat panel move to phase 5 | §6, §6.7, §7, §10 |
| 7 | Safe and selective resume? | **Node-local semantics, dependency values and a frozen input revision.** Keep the spec digest for provenance. A `frozen` workspace mode makes readers reusable; live readers always rerun; consumers rerun only when a value changed; Python worker reuse is off in v1 | §4.4, §5.3, §5.4, §11 |
| 8 | Hard token ceiling in v1? | **No.** Preserve `reported_tokens` as a stopping threshold with explicit overshoot and missing-usage semantics. Hard ceilings need a separate reservation/backend contract | §2.1, §4.6, §4.10 |
| 9 | Who authorizes executors? | **The router by inheritance, a person for anything more.** Executors are the router's depth-1 subagents and inherit its context and tool permissions. Extra role tools need a per-Ticket approval from a listed approver. Ticket tools are host-injected | §4.2, §4.3, §6.2, §6.6, invariants 19 and 21 |
| 10 | Does `review-changes` fit its cap? | **Four findings per dimension.** Three reviewers, 36 skeptics and one writer fit 40; static validation rejects larger bounds | §5.2, §11 |
| 11 | Rerun a node that had effects? | **Never.** Resume returns a completed effecting node's recorded result, or blocks at it when it was interrupted or its inputs changed. Only a new confirmed run executes it again. No `retry_on` for effecting nodes | §4.8, §5.3, §5.4, invariant 20 |

Still open, and not blocking phase 0:

- In a background `delegate`/workflow run on the CLI, worker confirmations are
  still denied, because the editor owns the terminal. Decision 5 covers
  superagent executors only. Applying the same inherited-sandbox
  auto-approval to background L1 runs would be a separate, explicit opt-in.
