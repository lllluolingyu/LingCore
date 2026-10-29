# Milestone 3: Harbor benchmark adapter

Status: implementation plan, 2026-10-30. It expands
[milestone 3 of the tracing and evaluation design](tracing-and-evaluation.md#later-milestones)
and replaces the in-process adapter proposed in the
[Terminal-Bench integration design](evaluation-terminal-bench.md). It builds on
the milestone 2 [locked plans](evaluation-locked-plans.md). Harbor facts were
read from Harbor 0.25.0 (source commit `ed3f332`, 2026-10-29), and
Terminal-Bench facts from release v4.0.0 (commit `452bf305`, 66 tasks). Step
3.1 re-checks each one against the pinned releases.

## Goal

Run a locked plan whose tasks come from a Harbor dataset, Terminal-Bench 4.0
first. Harbor builds each task environment and runs its verifier. The agent is
the ordinary LingCore loop with a terminal toolset bound to that environment.
LingCore keeps everything milestone 2 built: the lock, the job ledger, resume,
infrastructure retries, attempt selection, traces, and reports.

Out of scope: Harbor installed agents, environment providers other than local
Docker, parallel trials, GPU, multi-container, and multi-step tasks, a
persistent PTY, a direct `harbor run -a …` route without `lingcore eval`, the
five-repetition benchmark track, leaderboard submission, and `eval compare`
(milestone 4).

## Vocabulary

| Term | Meaning |
| --- | --- |
| Harbor environment | A Python environment containing Harbor and the bridge, separate from LingCore's |
| Bridge | The `lingcore-harbor-bridge` distribution: a command-line tool plus a Harbor agent class. It never imports LingCore |
| Task environment | The container Harbor builds for one trial; the agent's commands run there |
| Harbor trial | One Harbor run of one task; always exactly one LingCore execution |
| Reward | The named numbers a task's verifier writes, as Harbor recorded them |

## Workflow

```sh
# Once: Harbor in its own environment. Installing the bridge installs its pinned Harbor.
uv tool install --python 3.12 ./adapters/harbor-bridge

# Once: export the dataset to a local directory. The only network step.
harbor download terminal-bench/terminal-bench@4.0.0 --export -o datasets/

# Freeze the plan: describes, selects, and hashes tasks through the bridge.
# No model call, no container, no network.
lingcore eval resolve docs/examples/evaluation/terminal-bench.yaml --lock tb.lock.json
lingcore eval validate --lock tb.lock.json

# Run, resume, and report exactly as in milestone 2.
lingcore eval run --lock tb.lock.json --output /tmp/lingcore-evals
lingcore eval resume /tmp/lingcore-evals/<job-id>
lingcore eval report /tmp/lingcore-evals/<job-id>
```

`uv tool` exposes only the bridge's own command. Run `harbor download` through
`uvx --python 3.12 harbor@0.25.0` or any other installation of the same version.

Export from the Hub rather than checking out the release's Git tag. In the
Hub copy, the agent and verifier environments of every task reference
digest-pinned prebuilt images. The Git source builds from Dockerfiles, and only
13 of its 140 `FROM` lines are pinned.

## Decisions

- **Two Python environments.** Harbor 0.25.0 requires Python ≥3.12 and,
  through litellm, installs its own openai (≥2.20) and pydantic, all with
  lower bounds. LingCore supports 3.11 and pins exact versions. They are never
  installed together, and LingCore's dependencies do not change. The bridge is
  a separate distribution in this repository. It pins `harbor==0.25.0` and
  depends on nothing from LingCore.
- **LingCore drives; Harbor runs one trial at a time.** For each execution,
  LingCore starts the bridge, which runs one Harbor job containing one trial
  (`-k 1 -n 1 --max-retries 0`) and writes it inside the execution directory.
  Every Harbor trial is exactly one execution in the milestone 2 ledger, so
  resume, selection, coverage, usage, and exit codes are reused rather than
  re-implemented.
- **The agent loop stays in the LingCore process.** The bridge's Harbor agent
  only relays terminal operations between LingCore and Harbor's
  `BaseEnvironment`. The profile, model credentials, trace, and budgets never
  enter the Harbor process or the task environment. This is stronger than the
  design doc's in-process adapter, which put credentials in Harbor's process.
- **LingCore owns retries.** Harbor's retries default to 0, and LingCore always
  passes `--max-retries 0`: before retrying, Harbor deletes the failed trial
  directory, which would erase evidence. The milestone 2 policy classifies
  Harbor outcomes; see [Outcome mapping](#outcome-mapping).
- **Harbor owns rewards; LingCore owns validity.** A reward is read only from
  the trial result the Harbor process writes, never from a file in the task
  environment. The bridge evaluates Harbor's own `reward_valid` and failure
  cause with the pinned error-policy table (`reward_valid` is a computed
  property, so only Harbor's code can evaluate it). LingCore combines them with
  its execution outcome.
- **Tasks come from a local export and are frozen at resolve.** The user
  exports the dataset once. `resolve` and `validate` read only that directory
  and the bridge, with no network and no containers. `run` passes each task
  directory to Harbor with `-p`, so what runs is what was hashed.
- **Terminal tools are LingCore tools.** `terminal_exec` and
  `terminal_read_output` live in LingCore and reach the task environment through
  `ToolContext.environment_binding`. `resolve` measures their schemas offline
  with a binding that refuses every call, as it already does with model clients.
- **Commands are supervised in the task environment.** Harbor's Docker `exec`
  buffers output without a cap and, on timeout, kills only the host-side
  client. The bridge caps output in the container and enforces each command's
  deadline there. Before Harbor collects artifacts or runs a shared verifier,
  it kills every process the agent started.
- **Regrade goes through Harbor.** A task whose verifier runs in a separate
  environment, against artifacts collected from the agent's environment, can
  be verified again without the agent. Harbor's regrade does this, and it
  checks that the new verifier's inputs were collected. Every Terminal-Bench
  4.0 task is verified this way. `eval regrade` delegates to Harbor's regrade,
  which needs Docker but no agent, model, or terminal tool. It refuses tasks
  with a shared verifier.
- **Task digests split like case digests.** Each task's agent-visible input
  and its verifier have separate digests. `run` refuses a change to either,
  and `regrade` refuses only an input change, as in milestone 2.

## Architecture

```text
LingCore environment                      Harbor environment (Python ≥3.12)
──────────────────────────────            ───────────────────────────────────
lingcore eval run --lock                  lingcore-harbor-bridge trial …
  Agent, model client, budgets              Harbor CLI, called in-process
  terminal_exec → binding ── Unix socket ──→ BridgeAgent.run()
                                              └→ BaseEnvironment.exec → task container
  trace, ledger, result.json                Harbor verifier and trial result
        ↑                                           │
        └──── lingcore-harbor-bridge summarize ←────┘ executions/<id>/harbor/
```

Per execution:

1. LingCore writes `trial.json`, creates a private `0700` directory holding a
   listening Unix socket, and starts `lingcore-harbor-bridge trial` in its own
   process group with an allowlisted environment.
2. The bridge runs Harbor with the task directory, the bridge agent
   (`-a lingcore_harbor_bridge.agent:BridgeAgent`), the socket path and
   execution ID as agent kwargs, and the locked timeout multipliers.
3. Harbor builds the environment and calls the agent's `setup` and then `run`.
   The agent probes the container's prerequisites, records the processes
   already running, connects, and sends `hello`. LingCore accepts one
   connection, then closes and removes the listener, as the `process` plugin's
   relay does.
4. LingCore checks `hello` against the lock (execution ID, task, instruction
   digest, bridge and Harbor versions). It then runs the Agent through
   `run_headless`, with the instruction as the only user turn. Each terminal
   tool call is one `exec` request.
5. When the turn ends, LingCore sends `finish`. The agent kills every process
   the agent started, verifies that none remain, fills Harbor's
   `AgentContext`, writes the ATIF trajectory if one was sent and it validates,
   and returns. Harbor collects the declared artifacts, runs the verifier, and
   removes the environments.
6. After the bridge exits, LingCore runs `lingcore-harbor-bridge summarize` on
   the Harbor output. It maps that summary and its own execution outcome to a
   `TrialResult` and writes `result.json`.

LingCore finds the bridge through `--bridge <path>` on `resolve`, `validate`,
`run`, and `resume`, then `LINGCORE_HARBOR_BRIDGE`, then `PATH`. The path is
machine-specific and never enters the lock. The versions and digest the bridge
reports do.

The bridge process environment contains only `PATH`, `HOME`, `USER`, `LANG`,
`LC_ALL`, `TMPDIR`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `DOCKER_HOST`,
`DOCKER_CONFIG`, `DOCKER_CONTEXT`, and the names in
`execution.harbor.pass_env`. It never contains the profile's `.env` values or a
credential the profile names; resolve refuses a `pass_env` entry that is one.

## Bridge protocol

`lingcore.harbor-bridge.v1` is newline-delimited JSON over the Unix socket. A
frame is at most 4 MiB. An oversized or malformed frame, an unknown message
type, or an unknown field ends the trial as a runner failure. Each package
defines its own models. Example frames in `adapters/harbor-bridge/protocol/v1/`
are parsed by both test suites, so the two definitions cannot drift apart.

| Message | Direction | Fields |
| --- | --- | --- |
| `hello` | bridge → LingCore | protocol, execution ID, bridge version and digest, Harbor version, task name, instruction, environment type, probe result |
| `exec` | LingCore → bridge | call ID, command, cwd, timeout, capture bytes |
| `exec_result` | bridge → LingCore | call ID, exit code, output, total output bytes, truncated, timed out, duration |
| `usage` | LingCore → bridge | Running totals and per-model usage, sent after each model request so a harness timeout still leaves them in `AgentContext` |
| `finish` | LingCore → bridge | Execution outcome, final usage, metadata, optional ATIF trajectory |
| `finished` | bridge → LingCore | Sweep status (`clean` or `failed`) and ATIF status (`written`, `invalid`, or `absent`) |
| `abort` | either | Reason: `cancelled`, `harness_timeout`, or `protocol_error` |

Requests are sequential. Harbor profiles must set `loop.parallel_tools:
false`, and the binding serializes calls regardless. EOF from LingCore, after
a crash or kill, counts as `abort`: the agent sweeps and returns, so Harbor
still verifies and cleans up, and its output survives for resume to inspect.

`AgentContext` receives input tokens (including cached, which matches
LingCore's convention), cached tokens, output tokens, per-model usage, `null`
cost, and `metadata.lingcore` with the execution ID, trial ID, plan digest,
outcome, and usage coverage.

## Terminal binding `harbor-terminal-v1`

| Tool | Arguments | Result |
| --- | --- | --- |
| `terminal_exec` | `command`; optional `cwd` (absolute, in the task environment) and `timeout_sec` | Exit code, duration, the first `display_bytes` of output, total size, and a handle when output was truncated |
| `terminal_read_output` | `handle`, `offset`, `limit` | A bounded slice of that command's captured output |

- Options under `tool_options.terminal_exec`: `timeout` (default 120 s),
  `max_timeout` (900 s; larger requests are clamped, as with `run_shell`),
  `capture_bytes` (1 MiB per command), and `display_bytes` (16 KiB). They are
  profile values, so the candidate digest covers them.
- Harbor's Docker environment merges stderr into stdout. The tool reports one
  combined stream and its description says so.
- Handles are random and scoped to the trial. They index an in-memory store
  bounded at 32 MiB in total, which evicts the oldest output first. An unknown,
  evicted, or foreign handle is an ordinary tool error. A handle never names a
  file.
- The description says that files persist between calls, and that the working
  directory and shell variables do not.
- Without a bound environment, both tools fail with a tool error. They never
  fall back to `run_shell`, the workspace, or any host path.
- `Agent.from_profile` gains `environment_binding=`, which it places on
  `ToolContext`. The tools are registered into a trial-local registry passed
  through the existing `tool_registry=`, never into the global registry.

Command supervision in the bridge agent:

- Each command runs through a small bash wrapper. Harbor's Docker `exec`
  already requires bash. The wrapper starts the command in a new session,
  captures the first `capture_bytes` while draining the rest, and counts the
  total bytes. Like `run_shell`, it never blocks the command or sends it
  SIGPIPE. It enforces `timeout_sec` inside the container by killing the
  command's process group.
- Harbor's own `exec` timeout is set 30 seconds longer, as a backstop. If the
  backstop fires, the bridge kills the process group with a second `exec` and
  checks that it is gone.
- Before the first command, the agent records the PIDs already present.
  `finish`, `abort`, cancellation, and EOF each kill every process outside that
  baseline, repeat until none remain (bounded), and remove the wrapper's
  runtime directory. All of this happens before `run()` returns, so no agent
  process can still change files while Harbor collects artifacts or runs a
  shared verifier. A sweep that cannot confirm an empty set reports `failed`,
  which invalidates the trial as a cleanup failure.
- The setup probe checks for `/proc`, `bash`, `setsid`, `kill`, `head`, `cat`,
  and `wc`. A missing prerequisite ends the trial as `unsupported` before the
  agent starts.
- The wrapper's runtime directory under `/tmp` is visible to the agent while
  it runs. It is removed before verification.

## Suite schema

`docs/examples/evaluation/terminal-bench.yaml` becomes the acceptance fixture.
Changes from the proposed shape:

| Field | Change |
| --- | --- |
| `source.path` | New: the exported dataset directory, relative to the suite |
| `source.selection` | See [Task selection](#task-selection); `interaction` is removed |
| `source.training_use` | New: `prohibited` for Terminal-Bench, whose tasks carry a training-data canary. Locks and reports repeat it so later exporters can exclude these jobs |
| `execution.harbor` | New: `pass_env` (names only) and `timeout_multipliers` (`agent`, `verifier`, `setup`, `build`; default 1) |
| `execution.retry_on` | As in milestone 2; Harbor environment failures map to `environment` |
| `grading` | `authority: harbor` plus `reward: {name, pass_at, values}`; `reward_policy` and `preserve_raw_results` are removed (raw results are always kept) |
| `report.headline` | `mean_reward` |

Preflight for `runner: harbor` refuses:
- a ceiling that differs from the binding's tools;
- any plugin, skill, or offerable skill;
- enabled sessions, non-empty `persona.project_instructions`, or a memory tool;
- `loop.parallel_tools: true`;
- `approval_policy` other than `benchmark_environment` (which is valid only
  for this runner);
- `trace.exports: [atif]` with `trace.content: metadata`.

It also refuses `limits.wall_seconds` that is not at least 60 seconds below
every selected task's agent timeout after multipliers, so LingCore's deadline
ends the turn before Harbor cancels it.

Terminal-Bench 4.0 instructions tell the agent it has 28800 seconds. When
`wall_seconds` is lower, the lock records the `reduced_limits` limitation,
because the instruction overstates the real budget. The prebuilt images are
`linux/amd64`. On any other host architecture, Docker emulates them, which is
slower and changes timing, so reports record `emulated_platform`.

## Task selection

`describe` reads each task with Harbor's own configuration model, so it
migrates deprecated fields exactly as a run would. Selection uses only these
derived properties:

| Property | Derived from | Terminal-Bench 4.0 |
| --- | --- | --- |
| `os` | `[environment].os`, or Harbor's default when omitted | All Linux |
| `gpus` | `[environment].gpus` and `[verifier.environment].gpus`; omitted means 0 | 3 tasks need a GPU |
| `containers` | Services in `environment/docker-compose.yaml`; otherwise 1 | 11 tasks use compose |
| `cpus`, `memory_mb`, `storage_mb` | `[environment]` | 2–16 CPUs, 4–32 GiB, storage up to 1,024,000 MB |
| `task_tools` | Non-empty `[environment].mcp_servers` or `skills_dir` | One task has MCP servers, one a skills directory |
| `steps` | `[[steps]]` | None |
| `verifier_mode` | `[verifier].environment_mode` as Harbor resolves it | All `separate` |

`selection.required` sets limits on these properties:
- `os` must match.
- `gpus` and `containers` are maximums.
- `max_cpus`, `max_memory_mb`, and `max_storage_mb` are maximums sized to the
  local Docker host.
- `task_tools: false` excludes task-provided tools.
- Multi-step tasks are always excluded in this milestone.

The bridge may be unable to read a property. That task is then excluded with
reason `metadata_unreadable`; the value is never guessed from the task name or
instruction.

Interaction mode and network needs are not in the metadata, so they are not
selection keys:
- Terminal-Bench is an open-internet benchmark. Every task runs with Harbor's
  default public network, and its CI forbids tasks from declaring otherwise.
- The tasks that need a GUI or an MCP server are already excluded by
  `containers` or `task_tools`.

With `os: linux`, `gpus: 0`, and `containers: 1`, 52 of the 66 tasks are
eligible before the resource limits apply.

Two policies:
- `sorted_capability_subset` sorts eligible task IDs and takes the first
  `count`.
- `explicit` takes a listed set and fails if any listed task is ineligible.

Both store the full included and excluded lists, with reasons, and too few
eligible tasks fails resolve. Alphabetical selection is only an integration
smoke test; scientific comparisons use `explicit` with a justified list.

## Lock additions

The lock stays `lingcore.eval-lock.v1`. `source` is discriminated by `kind`,
and `execution` and `grading` gain variants for `runner: harbor` and
`authority: harbor`. A native lock's bytes and digests do not change.

```json
{
  "source": {
    "kind": "harbor",
    "digest": "sha256:…",
    "dataset": {
      "ref": "terminal-bench/terminal-bench@4.0.0",
      "path": "../datasets/terminal-bench",
      "release_source": {
        "repository": "https://github.com/harbor-framework/terminal-bench",
        "commit": "452bf305c6daa62fc59061d22133a7cbc7c1572e"
      },
      "training_use": "prohibited"
    },
    "selection": {
      "policy": "sorted_capability_subset",
      "count": 3,
      "required": {"os": "linux", "gpus": 0, "containers": 1, "task_tools": false,
                   "max_cpus": 4, "max_memory_mb": 8192, "max_storage_mb": 20480},
      "included": ["…"],
      "excluded": [{"task": "…", "reasons": ["gpus"]}]
    },
    "tasks": [{
      "id": "wal-recovery-ordering",
      "name": "terminal-bench/wal-recovery-ordering",
      "path": "wal-recovery-ordering",
      "input_digest": "sha256:…",
      "verifier_digest": "sha256:…",
      "support_digest": "sha256:…",
      "harbor_digest": "sha256:…",
      "verifier": {"mode": "separate", "timeout_sec": 1800, "artifacts": ["/app/"]},
      "agent_timeout_sec": 28800,
      "resources": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240},
      "images": {"agent": {"kind": "image", "pinned": true},
                 "verifier": {"kind": "image", "pinned": true}}
    }]
  },
  "execution": {
    "runner": "harbor",
    "approval_policy": "benchmark_environment",
    "harness": {
      "digest": "sha256:…",
      "harbor": "0.25.0",
      "python": "CPython 3.12.…",
      "bridge": {"version": "0.1.0", "digest": "sha256:…",
                 "protocol": "lingcore.harbor-bridge.v1"},
      "environment": "docker",
      "error_policy_digest": "sha256:…",
      "atif": "ATIF-v1.7",
      "timeout_multipliers": {"agent": 1, "verifier": 1, "setup": 1, "build": 1}
    }
  },
  "grading": {
    "digest": "sha256:…",
    "authority": "harbor",
    "reward": {"name": "reward", "pass_at": 1, "values": [0, 1]}
  }
}
```

Unchanged sections (`suite`, `candidate`, `runtime`, `trace`, the rest of
`execution`) are omitted above.

| Field | Contents and drift meaning |
| --- | --- |
| `source.dataset.path` | Relative to the lock, like case paths |
| `source.dataset.release_source` | Informational. An export cannot be proven offline to match the release commit, so `limitations` records `release_unverified` |
| `source.tasks[].input_digest` | What shapes the agent's run: `instruction.md`, `environment/`, and Harbor's parsed task configuration without `[verifier]` and `artifacts`. File sets are sorted `(relative path, file digest)` pairs; configuration is canonical JSON, with deprecated fields migrated as the bridge prints it |
| `source.tasks[].verifier_digest` | `tests/`, the parsed `[verifier]` section, and `artifacts` (what is collected for the verifier) |
| `source.tasks[].support_digest` | Every other file, such as `solution/`, `README.md`, and licences. The agent never sees them, but they belong to the task |
| `source.tasks[].harbor_digest` | Harbor's own task digest, as the bridge reports it. Each trial's Harbor lock is compared with it after the run |
| `source.tasks[].images` | For the agent and verifier environments, `kind` is `dockerfile`, `compose`, or `image`. `pinned` is true only for an image reference with a `sha256` digest; an unpinned task adds the `image_unpinned` limitation |
| `execution.harness` | Everything the bridge reports about itself and Harbor. Any change is drift |
| `grading.reward` | The named reward that decides success, its pass threshold, and the allowed values. Any other value makes the grade invalid |

A task ID is the task's directory name in the export, and `path` is relative to
the dataset path. The ID must match `[a-z0-9][a-z0-9._-]*`, or resolve fails.
`name` records Harbor's `[task].name`. Trial IDs are `<task-id>--candidate--<n>`,
in sorted task order.

## Job directory additions

```text
executions/<execution-id>/
  harbor/                  # Harbor --jobs-dir; written only by Harbor
  harbor.lock              # held by the bridge process while it runs
  harbor-summary.json      # the bridge's summary of Harbor's trial; exclusive create
  regrades/<grade-set>/    # Harbor's regrade output for one pass; written only by Harbor
```

`evidence.json` indexes the Harbor files with digests and sizes: the trial
result, configuration, and lock; the verifier's reward and test output; the
exception text; and the trajectory. LingCore reads any of them only as bounded
bytes, limited by `trace.artifact_max_bytes`, and never modifies or moves them.

`result.json` adds `rewards` (verbatim), `reward_valid`,
`harbor_failure_cause`, `harbor_exceptions` (type names with their pinned
policy), per-phase `timing`, and the task digest from Harbor's trial lock.

## Outcome mapping

The bridge summary supplies Harbor's failure cause and `reward_valid`. LingCore
supplies whether `hello` arrived and its own `ExecutionOutcome`. Rows are
checked in order.

| When | Execution outcome | Grade | `retry_on` cause |
| --- | --- | --- | --- |
| Bridge or Harbor exits without a trial result, or the protocol fails before `hello` | failed, `runner` | not run | `runner` |
| Harbor cause `sandbox_failed` before `hello` (build, start, setup, healthcheck) | failed, `environment` | not run | `environment` |
| Setup probe failed | failed, `environment` | not run, `unsupported` | none; counted as unsupported |
| Harbor cause `harbor_failed` | failed, `runner` | not run | none; stops the invocation like `configuration` |
| Sweep failed, or LingCore cleanup failed | as recorded | invalid, `cleanup_failed` | never |
| Protocol failure after `hello` | failed, `runner` | invalid | `runner` |
| LingCore provider `transient` or `interrupted` | failed, `provider` | invalid (the reward is kept, not counted) | `provider_transient` when opted in |
| LingCore provider `auth` or `configuration` | failed, `provider` | invalid | none; stops the invocation |
| Harbor cancelled `run()` (`AgentTimeoutError`) | `timed_out`, limit `harness_agent` | from reward | none |
| Harbor cause `verifier_failed` (timeout, missing, empty, or unparsable reward) | as LingCore | invalid, `verifier_failed` | none: the agent can influence it |
| A reward outside `grading.reward`, or `reward_valid` false for any other cause | as LingCore | invalid, `reward_invalid` | none |
| Otherwise | as LingCore (completed, timed out, budget, agent error) | valid; passed when reward ≥ `pass_at` | none |

Unsupported trials resolve with reason `unsupported` and count in
`coverage.unsupported`, which milestone 1 reserved. A `sandbox_failed` cause
after `hello`, such as running out of memory, falls to the `reward_valid` row
and is not retried, because the agent's commands may have caused it. Harbor's
`model_provider_failed` classes never occur, because Harbor does not run the
model.

## Cancellation, crashes, and resume

- Ctrl-C or external cancellation during a Harbor trial: LingCore stops the
  Agent with the normal Stop handshake and sends `abort`. It then sends
  SIGTERM to the bridge's process group (Harbor turns SIGTERM into its own
  interrupt handling and tears down the environment), waits up to 120
  seconds, then sends SIGKILL. A SIGKILL records `environment_cleanup:
  unknown` in the result, because the container may remain. All of this runs
  under `finish_io` before cancellation propagates; only the CLI converts it to
  exit 130.
- If LingCore itself dies, the bridge sees EOF, sweeps, and returns. Harbor
  verifies, cleans up, and writes its result into the execution directory.
- Resume first checks each unfinished Harbor execution's `harbor.lock` without
  blocking. If a bridge from an earlier invocation still holds it, resume
  refuses with `harness_running` and names the execution. Otherwise orphaned
  executions are marked interrupted as in milestone 2, with Harbor's files
  kept. They are never imported as results, because the agent's turn did not
  finish under LingCore's control.
- Harbor's own retries stay off, so its output directory is never deleted.

## Steps

### 3.1 Harbor environment and bridge package

This step gates the rest. New `adapters/harbor-bridge/` with its own
`pyproject.toml`: distribution `lingcore-harbor-bridge`, Python ≥3.12,
`harbor==0.25.0`, and console script `lingcore-harbor-bridge`. Module
`lingcore_harbor_bridge` may import only the standard library and Harbor. The
script has four subcommands, each printing bounded JSON:
- `describe` (this step);
- `trial` (3.3, 3.5);
- `summarize` (3.5);
- `regrade` (3.7).

This step implements `describe` and settles the open facts.

- `lingcore-harbor-bridge describe --dataset <dir>`: prints JSON with the
  protocol version, the bridge version and package digest, the Harbor and
  Python versions, the error-policy table and its digest, and the supported
  ATIF versions. For each task, it also prints Harbor's task digest, the parsed
  task configuration, the environment kind with its compose service count, and
  the instruction digest.
- Confirm against 0.25.0, and record the answers in this plan:
  - Harbor's CLI entry point can be called in-process with the bridge agent
    importable.
  - How Harbor computes `task_checksum`, and whether `describe` can reproduce it.
  - How `ExecResult.stdout` is decoded.
  - That `--max-retries 0`, `-k 1`, and `-n 1` produce exactly one trial
    directory.
  - That SIGTERM removes the task container.
  - That 0.25.0 parses all 66 Terminal-Bench 4.0 task files (most declare
    `schema_version = "2.0"`) and reproduces the eligibility counts above.
  - How a regrade of one local source trial is started (the job
    configuration's `SourceTrialConfig` with `action: regrade`), and where its
    output goes.
- The bridge has its own Ruff and mypy configuration. LingCore's `mypy`
  remains `files = ["lingcore"]`.

Tests: a separate CI job installs the bridge into a fresh Python 3.12
environment and runs the bridge tests there. Those tests check:
- importing the bridge with `lingcore` blocked;
- `describe` over a fixture dataset;
- constructing `BridgeAgent` through Harbor's agent loader.

LingCore's own suite installs neither.

### 3.2 Bridge protocol and terminal binding (LingCore side)

New `lingcore/evaluation/harbor/` with `protocol.py` (message models and
framing), `listener.py` (private directory, one connection, execution ID
check), `binding.py` (`HarborTerminal`, implementing `EnvironmentBinding`),
and `tools.py` (the two tools and the trial-local registry). `Agent.from_profile`
gains `environment_binding=`.

Tests, with a fake bridge (a stdlib script speaking the protocol over an
in-memory environment):
- Without a binding, the tools fail and the workspace is untouched.
- `display_bytes`, `capture_bytes`, and the store cap hold; evicted, unknown,
  and foreign handles are tool errors.
- `max_timeout` clamps.
- Cancellation sends `abort`.
- Oversized, malformed, and unknown frames end the trial as `runner`.
- A second connection and a wrong execution ID are refused.
- The socket path stays below the macOS limit.
- No frame and no bridge environment contains a fixture credential.
- Offline identity measures both schemas with a refusing binding.

### 3.3 Bridge agent and command supervision

`BridgeAgent(BaseAgent)` covers the setup probe, baseline, `hello`, the
request loop, the wrapper, per-command timeout, the sweep, incremental
`AgentContext` population, and handling of `CancelledError` and EOF. Exceptions
it raises are ordinary types, so Harbor classifies them as `unclassified`. The
bridge never names an exception after an entry in Harbor's table to change its
treatment.

Tests:
- In the bridge suite, with a fake `BaseEnvironment`: message sequencing, the
  sweep on each exit path, and `AgentContext` contents.
- In opt-in Docker tests (`LINGCORE_HARBOR_E2E=1`):
  - `yes` is capped inside the container, and the bridge's memory stays bounded.
  - `sleep 1000 &` is gone before verification.
  - A timed-out command's process group is killed, checked with a second `exec`.
  - An image without `setsid` is `unsupported`.
  - A Harbor agent timeout produces `abort` and a valid graded result.

### 3.4 Harbor source, selection, and lock

Suite and lock models for `source.kind: harbor`, `runner: harbor`, and
`authority: harbor`. `resolve` runs `describe`, applies the selection, hashes
the task directories, records the harness, and writes the lock. `validate`
repeats `describe` and hashing and reports drift by task ID and field.

Tests, with the fake bridge and a fixture dataset of tiny tasks (one plain,
one GPU, one compose, one multi-step, one with MCP servers, one too large, and
one with unreadable metadata):
- Selection includes and excludes them with reasons, deterministically.
- Too few eligible tasks fails resolution.
- Drift is named for each task digest (an instruction, an environment file,
  a test, a verifier timeout, a solution file), the harness version, the
  bridge digest, and the selection.
- Changing `tests/` moves only `verifier_digest`, and changing `instruction.md`
  moves only `input_digest`.
- The preflight refusals above.
- The milestone 2 native lock fixture still validates byte-for-byte.
- No network and no Docker are used (both are made unavailable in the test).

### 3.5 Harbor runner, outcomes, and resume

The executor gains a Harbor trial path beside the native one. It covers
listener setup, the bridge subprocess with the allowlisted environment and
Harbor flags, the `hello` check, `run_headless`, `finish`, a bounded wait,
`summarize`, the outcome mapping, and `result.json`. Cancellation and resume
behave as described above.

Tests, with a fake bridge that writes Harbor-shaped results:
- One test per outcome-mapping row.
- Retry selection with a fresh execution directory per attempt.
- The flags never enable Harbor retries.
- The environment allowlist holds.
- A LingCore crash mid-trial followed by resume.
- `harness_running` refusal while a stale bridge holds `harbor.lock`.
- No record is modified after it is written.

### 3.6 Rewards and reports

Terminal-Bench rewards are binary: `tests/test.sh` writes exactly 0 or 1 to
`reward.txt`. Its suite therefore declares
`reward: {name: reward, pass_at: 1, values: [0, 1]}`.

The report stays `lingcore.eval-report.v2` and gains additive fields:
- Per trial: `rewards` and `reward_valid`.
- `mean_reward`: per case, then overall, over valid selected trials.
- `harbor_accuracy`: successes divided by planned trials, with missing and
  invalid trials counted as 0. This matches Harbor's default mean and the
  leaderboard's denominator (the official run uses five repetitions; this
  milestone does not claim that score).
- Harbor phase durations beside LingCore's latency.
- Harness provenance, `training_use`, and `coverage.unsupported`.

Exit codes follow milestone 2.

Tests:
- Binary and named rewards.
- A reward outside the declared values.
- A missing reward name.
- Both headline metrics with invalid and missing trials.
- Report fields for each outcome row.

### 3.7 Regrade through Harbor

For a Harbor job, `eval regrade <job> [--lock <path>]` keeps the milestone 2
steps and records the same grade sets and reports:

1. Acquire `.lock` and load the current tasks. Refuse, by task ID, any task
   whose `input_digest` changed; a changed `verifier_digest` is the point and
   is recorded.
2. Refuse, by task ID, any task whose verifier mode is not `separate`. Check
   that every finished execution has a Harbor trial, then start no regrade
   until the whole pass is checked.
3. For each finished execution, run `lingcore-harbor-bridge regrade` with the
   source trial and the current task. The output goes to
   `executions/<id>/regrades/<grade-set>/`. Harbor seeds the saved artifacts
   into a fresh verifier environment, with its no-op agent in place of an
   agent.
4. Map each regrade summary to a grade with the outcome-mapping rows for
   verification. A Harbor `RegradeError` (a verifier input that was never
   collected) refuses that task, as a missing evidence need does in
   milestone 2.

Regrade starts no LingCore Agent, model client, terminal tool, or bridge
socket. It does need Docker.

Tests, with the fake bridge:
- A corrected test produces a new grade set while every original record stays
  byte-identical.
- A changed instruction is refused.
- A shared-mode task is refused.
- A `RegradeError` is refused by task.
- A factory that raises proves no Agent or model client is built.

### 3.8 ATIF export

LingCore builds an `ATIF-v1.7` trajectory from the trace journal, following the
mapping table in the design doc, and sends it in `finish`. The bridge
validates it with Harbor's own trajectory model before writing
`agent/trajectory.json`, and declares `AgentCapabilities(atif=True)`. Without
captured content, no trajectory is sent and the report says `atif: absent`.

Tests: LingCore golden trajectories cover parallel calls, compaction,
transport retries, incomplete responses, cancellation, and missing provider
usage. The bridge suite validates the same golden files with Harbor's model.

### 3.9 Synthetic end-to-end, then the development subset

A synthetic disposable task in `docs/examples/evaluation/harbor-synthetic/`,
shaped like a Terminal-Bench task:
- an instruction to write a file;
- an environment `FROM` a digest-pinned small image;
- `environment_mode = "separate"`, with `artifacts = ["/app/"]` and a
  `tests/Dockerfile`;
- a `tests/test.sh` that writes `reward.txt`;
- a reference solution.

- Opt-in end-to-end test, with Docker and the bridge:
  - Harbor's own oracle and no-op agents give rewards 1 and 0.
  - `lingcore eval run --lock` with a scripted fake model client passes.
  - The same run with a failing script gives a valid failure.
  - Correcting the test and running `eval regrade` changes only the new grade
    set.
  - Ctrl-C mid-command leaves no container.
- Manual step: resolve and run the Terminal-Bench development subset with a
  real model. Inspect every trace and compare the normalized rewards with
  Harbor's originals. This step is not a CI gate.

## Documentation updates

- `docs/evaluation-terminal-bench.md`: point to this plan. Replace the
  in-process adapter with the bridge, record LingCore-owned retries, and note
  that `ExecutionOutcome` and provider categories already exist.
- `docs/tracing-and-evaluation.md`: milestone 3 status, and a link to this plan.
- `docs/examples/evaluation/terminal-bench.yaml` and `terminal-agent.yaml`: the
  implemented schema.
- `docs/examples/evaluation/README.md`: installing the bridge, exporting the
  dataset, and the Harbor workflow.
- `CLAUDE.md`: architecture entries for `lingcore/evaluation/harbor/` and
  `adapters/harbor-bridge/`, and the rule that the bridge never imports
  LingCore and Harbor never sees credentials.
- `README.md` roadmap.

## Acceptance for the milestone

- The synthetic task resolves, validates, runs locked, resumes after a killed
  LingCore process, regrades a corrected test, and reports, with a fake model
  client and real Docker in the opt-in job.
- `resolve` and `validate` use no network, no containers, and no model call.
- No lock, job record, report, journal, bridge frame, or Harbor file contains
  a fixture secret. The Harbor process environment contains none.
- Harbor never retries, and no Harbor output directory is deleted.
- Attempt selection never depends on a reward.
- No agent-started process survives into verification on the synthetic task.
- LingCore's dependencies are unchanged, and LingCore's test suite needs
  neither Harbor nor Docker.
- Full LingCore test suite, `ruff check`, `ruff format --check`, and `mypy`
  pass, and so do the bridge's own checks.
