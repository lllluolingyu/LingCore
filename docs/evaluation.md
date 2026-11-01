# Tracing and evaluation

LingCore supports native cases with deterministic graders and terminal tasks
through an isolated Harbor bridge. Both use fresh execution state, bounded
traces, locked plans, immutable attempt records, resume, regrading, and offline
comparison. Native cases run sequentially; distributed execution and
OpenTelemetry export are not implemented.

## Create a native suite

Keep suite, case, profile, lock, and job files outside the model-visible
workspace. YAML paths resolve relative to the declaring file; lock locators
resolve relative to the lock. Create these three files in a directory such as
`my-eval/`.

`agent.yaml` uses the normal agent profile schema:

```yaml
name: file-evaluation
llm:
  backend: openai
  model: ${LINGCORE_MODEL}
  base_url: ${LINGCORE_BASE_URL}
  api_key_env: LINGCORE_API_KEY
  modalities: []
persona:
  system_prompt: Complete the task using the available tools and verify the result.
  project_instructions: []
tools: [read_file, write_file]
sessions:
  enabled: false
```

Set the three environment variables in your shell or a private `.env` beside
the profile. Credentials never belong in suites, cases, or locks. Select
`backend: openai-responses` for the native Responses API; `openai` uses Chat
Completions. See the [backend configuration](../README.md#native-openai-responses).

`case.yaml` contains the task and evaluator-only assertions:

```yaml
schema_version: lingcore.eval.case.v1
id: write-greeting
revision: 1
requires:
  tools: [write_file, read_file]
input:
  turns:
    - user: 'Create hello.txt containing exactly "Hello, LingCore!" and one newline. Read it to verify.'
fixture:
  workspace: empty
graders:
  - id: artifact
    kind: artifact_text
    path: hello.txt
    equals: "Hello, LingCore!\n"
  - id: tools
    kind: tool_policy
    require:
      - tool: write_file
        arguments: {path: hello.txt}
      - tool: read_file
        arguments: {path: hello.txt}
    causal_order: [[write_file, read_file]]
    forbid: [run_shell, terminal_exec]
```

Only the input turns reach the agent; graders remain outside model context.
Cases may seed relative text files through `fixture.files` and provide several
scripted turns. `artifact_text` checks saved files; `tool_policy` checks required
successful calls, forbidden attempts, argument predicates, and causal ordering.
Argument predicates require redacted trace content. Arbitrary fixture copying,
external native-case sources, and model-based graders are not supported.

`suite.yaml` selects the cases, profile, limits, and trace policy:

```yaml
schema_version: lingcore.eval.suite.v1
name: file-smoke
track: development
source:
  kind: local
  cases: [case.yaml]
  lock: {required: false}
agent:
  profile: agent.yaml
execution:
  runner: native
  repetitions: 2
  concurrency: 1
  approval_policy: deny
  limits:
    wall_seconds: 60
    model_requests: 8
    tool_calls: 8
    reported_tokens: 32000
    usage_missing: stop
trace:
  required: true
  content: redacted
grading:
  authority: local
report:
  headline: mean_case_success
```

Every execution receives a fresh private workspace and separate profile state,
removed after evidence collection. Native defaults apply independently to each
omitted limit: 60 seconds, 20 requests, and 40 tool calls. Explicit `null`
disables a bound. Token admission occurs between requests, so the last response
can exceed its threshold; token limits are not monetary spending caps. Shell
tools require an explicitly configured sandbox.

## Run, lock, and resume

For quick development, run the suite directly:

```sh
lingcore eval run my-eval/suite.yaml --output my-eval/jobs
```

Direct native runs remain unlocked development evaluations. They can be
regraded but cannot resume. Set `source.lock.required: true` to require a lock.
Resolve and validate are offline and need no provider credentials:

```sh
lingcore eval resolve my-eval/suite.yaml --lock my-eval/suite.lock.json
lingcore eval validate --lock my-eval/suite.lock.json
lingcore eval run --lock my-eval/suite.lock.json --output my-eval/jobs
lingcore eval resume my-eval/jobs/<job-id>
lingcore eval report my-eval/jobs/<job-id>
```

A lock pins ordered cases, effective prompts/tool schemas, profile/model,
runtime files, dependencies, execution policy, limits, and graders. Drift
refuses execution before creating a job. Resolve a fresh lock for intentional
changes; `--replace` atomically replaces an existing lock. YAML comments and
key ordering do not change identity; prompt/tool source assets are hashed as
bytes. Credential availability and Git status are informational. Model aliases
may still change behind an endpoint; reports retain provider-reported model IDs.

Resume holds the job lock, preserves existing attempts, and replaces interrupted
ones from fresh state. If the suite and lock moved together, pass
`--lock /new/location/suite.lock.json` to resume with the identical saved lock.
Unreadable results count as interrupted; unreadable trial records remain
preserved and appear in `ledger_issues`. Their usage coverage stays incomplete.

Optional infrastructure retries use `execution.infrastructure_retries` and
`execution.retry_on: [runner, environment]`; add `provider_transient` to opt in
to transient provider retries. Attempt selection is independent of scores.
Failed grades, agent errors, budgets, timeouts, and cleanup failures never
trigger retries. Auth/configuration errors stop the invocation; after fixing
credentials, resume restarts the affected trial without spending the
infrastructure retry allowance. Changing a locked model still needs a new lock.

## Traces, evidence, and regrading

Tracing is opt-in through a profile's `tracing` settings; evaluation suites
declare their own trace policy. Metadata capture is separate from bounded
redacted conversation/tool content. The append-only journal records lifecycle
spans and usage for each request. Inspect an execution with:

```sh
lingcore trace show my-eval/jobs/<job-id>/executions/<execution-id>/trace
```

A failed/cancelled request may have billed usage without a committed assistant
message. Usage accounting stays independent of conversation rollback. Missing
usage and unknown costs remain unknown rather than estimated. Provider
reasoning, signed blocks, encrypted Responses state, credentials, and environment
values are excluded from content capture. Identical sanitized prompts/tool
schemas are captured once per turn. Truncation preserves already admitted
evidence and records omitted content explicitly.

Each job retains immutable `job.json`, `plan.json`, and, when locked, `lock.json`.
The `executions/` ledger holds start records, results, traces, and evidence.
Original grades remain in `report.json`; new grade sets have separate reports.
`--grades latest` selects the newest completed grading pass, falling back to the
original report if none completed. Failed or interrupted passes remain available
by their grade-set ID.
V1 reports remain readable, but V1 jobs cannot resume or regrade.

```sh
lingcore eval regrade my-eval/jobs/<job-id>
lingcore eval report my-eval/jobs/<job-id> --grades latest
lingcore eval report my-eval/jobs/<job-id> --grades original
```

Native regrading reads saved evidence without agents, tools, or provider calls.
Case inputs and grader identities have separate digests: changed inputs refuse
regrade, while corrected graders produce new grades. Declare additional saved
paths in `evidence.artifacts` before a run. Missing files remain valid failures;
evidence that was never collected cannot be reconstructed. Incomplete trace
evidence remains `not_evaluable`.

Run/report exit codes are 0 for complete passing trials, 1 for complete valid
task failures, 2 for refused/invalid/incomplete evaluations, and 130 for
interruption. The CLI prints a job path whenever one exists. Library cancellation
propagates after finalization, cleanup, and best-effort result persistence.

## Compare jobs

Resolve a separate lock for each changed candidate, keeping case inputs,
repetitions, execution/trace policy, and grader identities compatible:

```sh
lingcore eval compare my-eval/jobs/<baseline> my-eval/jobs/<candidate> \
  --output my-eval/comparison.json
lingcore eval compare my-eval/jobs/<baseline> my-eval/jobs/<candidate> \
  --baseline-grades latest --candidate-grades latest
```

Comparison is offline and works without the original suite files. It reads V2
ledgers rather than trusting stale report summaries. Model, profile, and runtime
differences are allowed. Grade selectors accept `original`, `latest`, or a saved
grade-set ID. `--output` refuses an existing file; omit it for JSON on stdout.

Results include paired per-case means, regressions/improvements, latency
differences, and selected/all-attempt usage. Deltas are candidate minus baseline.
Missing or invalid pairs suppress the overall delta and confidence interval.
A deterministic 95% percentile bootstrap resamples whole cases with their
repetitions together (2,000 draws, seed 0 by default). One case has no interval;
repeated trials are not independent tasks. Small-suite intervals are descriptive.

Comparison exits 1 if any case's mean decreases, 2 for incompatible/incomplete
evidence, and 0 otherwise. It never rewrites source jobs or grants public
benchmark eligibility.

## Harbor and Terminal-Bench

The [Harbor bridge](../adapters/harbor-bridge/README.md) is an independent
Python 3.12+ package pinned to Harbor 0.25.0. Keep it outside LingCore's environment.
Docker runs need Compose and Buildx, even with prebuilt task images. Download
the public dataset separately, then set the suite's `source.path` to the export
directory containing the task folders:

```sh
uv tool install --python 3.12 ./adapters/harbor-bridge
uvx --python 3.12 harbor@0.25.0 download terminal-bench/terminal-bench@4.0.0 \
  --export -o datasets/
```

Use an absolute dataset path without symlink components or a dataset beneath
the suite directory; parent-relative paths containing `..` are refused. Set
`LINGCORE_HARBOR_BRIDGE` or pass `--bridge` if the executable is not on PATH.
Resolve and validate need neither Docker nor model access. Run/resume need both;
Harbor regrade runs only a separate verifier against preserved artifacts.

For a small CPU development suite, create `harbor.yaml` with:

```yaml
schema_version: lingcore.eval.suite.v1
name: terminal-development
track: development
source:
  kind: harbor
  dataset: terminal-bench/terminal-bench@4.0.0
  path: dataset
  training_use: prohibited
  selection:
    policy: sorted_capability_subset
    count: 3
    required:
      os: linux
      gpus: 0
      containers: 1
      task_tools: false
      max_cpus: 4
      max_memory_mb: 8192
  lock: {required: true}
agent:
  profile: terminal-agent.yaml
  tool_binding: harbor-terminal-v1
execution:
  runner: harbor
  repetitions: 1
  concurrency: 1
  approval_policy: benchmark_environment
  limits:
    wall_seconds: 600
    model_requests: 20
    tool_calls: 40
    reported_tokens: 64000
    usage_missing: stop
trace:
  required: true
  content: redacted
  exports: [atif]
grading:
  authority: harbor
  reward: {name: reward, pass_at: 1, values: [0, 1]}
report:
  headline: mean_reward
```

Create `terminal-agent.yaml` from the earlier agent profile, replacing its tools
with `[terminal_exec, terminal_read_output]`. Its system prompt should direct all
task work to the assigned terminal environment. Resolve to a lock and use the
same run/resume/report/regrade commands as native suites. Setup, build, and
verifier deadlines remain task-defined; reduced agent budgets produce
development results, not a full benchmark score.

LingCore owns the model loop, credentials, budgets, traces, usage, and immutable
attempt ledger. The bridge never imports LingCore or receives model credentials.
It runs one Harbor trial per execution and relays terminal operations through a
private Unix socket. Harbor provisions environments and runs verifiers.
Commands have bounded output and process lifetimes; agent-started processes are
killed before artifact collection/verification. Cleanup failures invalidate
grading. Harbor retries are disabled; LingCore owns retry selection.

Harbor supplies named rewards and its error policy. `mean_reward` averages valid
trials within tasks, then across tasks. `harbor_accuracy` counts missing/invalid
trials as zero over the planned denominator. Reports retain phase timings and
locked agent/verifier resource requests. Docker enforces CPU/memory limits, but
not storage quotas; requested resources are not measured utilization.

Collected artifacts exceeding the size limit are marked `oversized` without
invalidating a valid verifier result; unsafe paths still fail collection.
Regrade refuses changed agent inputs and shared verifiers. Missing artifacts
skip the affected attempt and appear in `regrade_skipped`. Non-default verifier,
setup, or build timeout multipliers refuse regrade; locked `pass_env` is forwarded.
ATIF v1.7 export requires captured content and labels incomplete/redacted data.
It never synthesizes reasoning or missing observations.

Terminal-Bench 4.0.0 exports are verified against the
[publisher catalog](../lingcore/evaluation/harbor/releases/README.md). Locks record
the Hub version, package hashes, release source commit, and catalog digest.
Partial exports are `verified_subset`; modified tasks require a distinct local
dataset name. Other unverified datasets remain usable for development.

Full benchmark policy requires all 66 tasks, five repetitions, published rewards,
original grading, task-defined budgets, pinned images, required traces, and
`training_use: prohibited`. Evaluator wall/request/tool/token caps must be
explicitly `null`, `usage_missing: ignore`, and `automatic_turns: 0`. Harbor
enforces task timeouts. Current CPU/single-container/no-task-tool capabilities
refuse unsupported tasks; GPU and multi-step workflows are not implemented.

`benchmark_score` requires a complete eligible original run, valid grades/traces,
successful cleanup, and no recorded reproducibility limitations. Development
subsets and regrades cannot claim compatibility. Local eligibility is not
upstream leaderboard approval.

## Automated checks

Evaluation test data lives in [tests/fixtures/evaluation](../tests/fixtures/evaluation).
The native regression cases cover file work, cited retrieval, source updates,
calendar recurrence across DST, and multi-turn invoice updates with deterministic
graders and scripted clients. The nightly Harbor job also uses fake model clients
and a synthetic task; public model evaluations remain manual.

```sh
uv run pytest tests/test_evaluation.py tests/test_regression_suite.py -q
LINGCORE_HARBOR_E2E=1 LINGCORE_HARBOR_BRIDGE=/path/to/lingcore-harbor-bridge \
  uv run pytest tests/test_harbor_e2e.py -q
```
