# Evaluation examples

The native example supports unlocked runs and the complete
[locked evaluation workflow](../../tracing-and-evaluation.md). The Harbor adapter
runs locked local exports with a separate Python environment and Docker.

| File | Purpose |
| --- | --- |
| [terminal-bench.yaml](terminal-bench.yaml) | Public Terminal-Bench 4.0 development subset through Harbor |
| [terminal-agent.yaml](terminal-agent.yaml) | Dedicated environment-only agent profile for that adapter |
| [local.yaml](local.yaml) | Executable native suite with two repetitions of a file-writing task |
| [local-agent.yaml](local-agent.yaml) | Minimal profile for the native example |
| [regression.yaml](regression.yaml) | Five native cases, three repetitions each |
| [regression-agent.yaml](regression-agent.yaml) | Offline retrieval and calendar tools for the regression suite |
| [terminal-bench-cpu-10/smoke.yaml](terminal-bench-cpu-10/smoke.yaml) | CPU-10 with bounded agent time, requests, tools, and tokens |
| [cases/write-greeting.yaml](cases/write-greeting.yaml) | Agent input, required tools, and evaluator-only assertions |
| [.env.example](.env.example) | Secret-free environment names for both profiles |

Suite and case YAML use the `lingcore.eval.*.v1` schemas. Agent YAML uses the
current profile shape. Keep suites, cases, locks, and output jobs outside the
model-visible workspace. Paths resolve relative to the declaring YAML file;
lock locators resolve relative to the lock file.

Set `LINGCORE_MODEL`, `LINGCORE_BASE_URL`, and `LINGCORE_API_KEY` in the process
environment, or a private `.env` beside a copied profile. See
[.env.example](.env.example). Credentials never belong in a suite or lock.
Resolution and validation are offline; keys may be unset when resolving, but
running and resuming require the profile's credentials.

Freeze the native example and run it:

```sh
lingcore eval resolve docs/examples/evaluation/local.yaml --lock docs/examples/evaluation/local.lock.json
lingcore eval validate --lock docs/examples/evaluation/local.lock.json
lingcore eval run --lock docs/examples/evaluation/local.lock.json --output /tmp/lingcore-evals
lingcore eval report /tmp/lingcore-evals/<job-id>
lingcore trace show /tmp/lingcore-evals/<job-id>/executions/<execution-id>/trace
```

Commit your lock beside the suite, cases, and profile. A lock pins the measured
prompt, tool schemas, source assets, models, runtime files, dependencies,
limits, graders, and ordered trials. Changes are named without printing their
values. Drift refuses execution before creating a job. Resolve a new lock for
an intentional change; `--replace` atomically replaces an existing lock.
Parsed YAML comments and key order do not change identity. Prompt and tool
source assets are hashed as bytes. Credential availability and git status are
informational, so adding credentials or committing unchanged code keeps the
plan identity. Model aliases may still change behind an endpoint; reports
record models reported by providers.

After Ctrl-C or a worker crash, continue the same job:

```sh
lingcore eval resume /tmp/lingcore-evals/<job-id>
# If the suite and lock moved together, point to the identical saved lock:
lingcore eval resume /tmp/lingcore-evals/<job-id> --lock /new/location/local.lock.json
```

Completed executions remain immutable. Resume replaces interrupted attempts
from fresh state, keeps their partial traces and usage, and runs only pending
trials in frozen order. Run, resume, and regrade refuse a held job lock.
Unreadable results count as interrupted. Unreadable trial records are preserved
and listed in `ledger_issues`; their directories remain unassigned and usage
coverage stays incomplete. Resume persists this decision under `quarantined/`
without moving or changing the original files. Stray files such as `.DS_Store` are skipped, while
symlinks and inconsistent execution identities are refused. Writes flush and
sync records and directory entries to disk.
Sequential infrastructure retries are optional:

```yaml
execution:
  infrastructure_retries: 1
  retry_on: [runner, environment]  # add provider_transient to opt in
```

Failed grades, agent errors, budgets, timeouts, and cleanup failures never
trigger retries. Selection uses the first eligible execution regardless of its
grade. Auth and configuration errors stop the current invocation. After fixing
credentials, `eval resume` restarts the stopped trial and then the remaining
ones, preserving every failed attempt. A fresh auth/configuration failure stops
the resumed invocation too. These restarts and interrupted attempts do not
consume the infrastructure retry allowance. Provider HTTP 404 is a configuration
stop; changing a locked model ID still requires a new lock.

After correcting a case's grader, regrade saved evidence:

```sh
lingcore eval regrade /tmp/lingcore-evals/<job-id>
lingcore eval report /tmp/lingcore-evals/<job-id> --grades latest
lingcore eval report /tmp/lingcore-evals/<job-id> --grades original
```

Regrading writes a new grade set without changing any existing result, running
an Agent, or calling models/tools. Changed case inputs or an artifact that was
never preserved are refused. Declare `evidence.artifacts: [path, ...]` in a
case before running to preserve additional paths for future graders. A missing
file at a collected path remains a valid failure. Tool argument assertions
require `trace.content: redacted`; incomplete trace evidence remains
`not_evaluable`.

Each job contains immutable `job.json`, `plan.json`, the exact `lock.json`
when locked, and an append-only `executions/` ledger. Each execution has its
start record, direct trace journal, evidence manifest, content-addressed binary
artifacts, and final result or interruption marker. New grades live beside
results. `report.json` describes the original grading; `reports/<grade-id>.json`
describes a regrade. Report v2 includes lock provenance, attempts, coverage,
selected/all-execution usage, and served models. V1 reports remain readable;
v1 jobs cannot resume or regrade.
The model comparison accepts all configured main/vision models and a dated
snapshot of an unversioned alias; explicitly pinned dates remain exact.

For quick development, the native suite keeps `lock.required: false`:

```sh
lingcore eval run docs/examples/evaluation/local.yaml --output /tmp/lingcore-evals
```

These results stay `locked: false` and can be regraded, but cannot resume.
Set `lock.required: true` to require locked runs. Both native modes remain
`benchmark_compatible: false` on the development track.

Omitted limits default independently to 60 seconds, 20 model requests, and
40 tool calls; explicit `null` disables a limit. Every execution has a fresh
private workspace and separate profile state, removed after evidence
collection. Cases may seed relative text files with `fixture.files`. The
native runner supports sequential scripted text turns and the two example
graders. Arbitrary fixture copying and external sources remain later work.

Exit codes: 0 for complete passing trials, 1 for complete valid task failures,
2 for refused, invalid, or incomplete evaluations, and 130 for interruption.
The CLI prints the job path whenever one exists. Library cancellation
propagates after finalization and cleanup.

Run the deterministic acceptance tests without provider calls:

```sh
uv run pytest tests/test_eval_identity.py tests/test_eval_lock.py tests/test_eval_jobs.py tests/test_eval_retries.py tests/test_eval_regrade.py tests/test_eval_cli.py -q
```

For a ready-to-prepare CPU sample, use the
[ten-task Terminal-Bench subset](terminal-bench-cpu-10/README.md). It pins ten
packages, runs one trial at a time, defaults to `gpt-6.1-sol`, and uses the relaxed
task-scale budgets. Its downloader fetches
only those ten packages, and its adjacent dataset path works without editing
the suite.

Install the bridge separately, export Terminal-Bench, and set `source.path` to
the export directory containing the task folders:

```sh
uv tool install --python 3.12 ./adapters/harbor-bridge
uvx --python 3.12 harbor@0.25.0 download terminal-bench/terminal-bench@4.0.0 --export -o datasets/
lingcore eval resolve docs/examples/evaluation/terminal-bench.yaml --lock tb.lock.json
lingcore eval validate --lock tb.lock.json
lingcore eval run --lock tb.lock.json --output /tmp/lingcore-evals
lingcore eval resume /tmp/lingcore-evals/<job-id>
lingcore eval regrade /tmp/lingcore-evals/<job-id>
```

Use `--bridge /path/to/lingcore-harbor-bridge` or `LINGCORE_HARBOR_BRIDGE` if it
is not on PATH. Resolve and validate need no Docker or model access. Run,
resume, and Harbor regrade need Docker; regrade starts no agent or model.
Install both Docker Compose and [Buildx](https://github.com/docker/buildx#installing).
Harbor builds a helper image for tasks with restricted verifier networking, even
when the task's main images are prebuilt.
Harbor retries are always disabled. The local ledger selects attempts
independently of rewards. A running orphan bridge blocks resume until it
releases its execution lock.

`mean_reward` averages valid trials per task, then tasks. `harbor_accuracy`
counts missing and invalid trials as zero over the planned denominator.
Named rewards, validity, phase timings, harness provenance, training-use
restrictions, and limitations remain in each report. Correcting a verifier
creates a new grade set; changing agent input refuses regrade.
Selected attempts with valid original grades or `verifier_failed` / `reward_invalid`
grades can be reverified; other attempts keep their original grades. Missing
artifacts skip only the affected attempt, preserve its original grade, and appear
in `regrade_skipped`. Regrade forwards the locked `pass_env`. Non-default verifier,
setup, or build timeout multipliers refuse regrade because the pinned Harbor
command cannot preserve them. Agent timeout multipliers are allowed.

Terminal-Bench 4.0.0 exports are checked offline against the bundled publisher
catalog. The lock records the immutable Hub dataset version, each task package
digest, the release's source commit, and the catalog digest. Modified or extra
tasks refuse the published dataset name; use a distinct local name for modified
development tasks. Partial exports are marked `verified_subset`. Other dataset
names remain usable for development with `release_unverified`.

The [benchmark example](terminal-bench-benchmark.yaml) declares the full 66-task,
five-repetition policy with task-defined budgets. It currently refuses tasks
requiring GPU, multi-container, or task-tool support. Keep using the development
example until those capabilities are implemented. A benchmark report exposes
`benchmark_eligibility` reasons and only supplies `benchmark_score` when the
original locked run is complete, valid, trace-complete, and free of recorded
reproducibility limitations. This is not a claim of upstream submission approval.

`resources.tasks` and each case's `resources` show the locked agent and separate
verifier requests, even for cases that never started. Shared verification uses
the agent environment. `resources.enforcement` describes backend policy: Docker
limits CPU and memory, but does not enforce storage quotas. These fields contain
resource requests, not measured utilization or forwarded environment values.

The [synthetic task](harbor-synthetic/write-file/) exercises separate-verifier
artifact collection without a public benchmark or paid model. Opt-in checks:

```sh
LINGCORE_HARBOR_E2E=1 LINGCORE_HARBOR_BRIDGE=/path/to/lingcore-harbor-bridge uv run pytest tests/test_harbor_e2e.py -q
```

The public suite's reduced budgets and alphabetical selection are development
checks, not a full benchmark score. Public model runs are manual; CI uses
scripted clients and synthetic tasks.

Known preflight issue: Harbor dataset paths containing `..` currently raise
`PathEscapeError`. Until normalization is fixed, set `source.path` to an absolute
dataset path without symlink components, or place the export beside the suite
and use `path: dataset`. The example's parent-relative path needs that adjustment.

## Native regression suite

`regression.yaml` runs five cases three times each with fresh state: the file
smoke, source-cited retrieval with an outdated distractor, source updates across
two turns, calendar recurrence across a daylight-saving transition, and an
invoice recalculation across two turns. `regression-agent.yaml` enables the
offline knowledge grep backend (no embeddings) and calendar tools. Dates and
timezones are explicit; no live search or clock is required.

The deterministic graders check exact saved artifacts, successful required tool
calls, and causal ordering. They do not judge free-form prose. The refresh case's
final artifacts and tool policy measure recovery after a source change; the CI
fixture additionally checks the actual retrieved text before and after the
update. Semantic-index evaluation requires an explicitly configured embedding
provider and is outside this offline suite. Each calendar repetition starts
with its own empty state. Its policy checks the requested event arguments,
including duration and recurrence, so a correct agenda cannot hide a wrong event.

```sh
lingcore eval resolve docs/examples/evaluation/regression.yaml --lock /tmp/regression.lock.json
lingcore eval run --lock /tmp/regression.lock.json --output /tmp/lingcore-evals
```

The run uses the model configured in the example environment and makes model
calls. CI drives the real tools with scripted fake clients and verifies both
passing cases and incorrect artifacts, without a paid provider. The original
`local.yaml` remains the small two-trial installation smoke example.

## Compare preserved jobs

Run the same case set and execution policy for each candidate, resolving a new
lock when the model, profile, or runtime changes. Then compare the saved jobs:

```sh
lingcore eval compare /tmp/lingcore-evals/<baseline> /tmp/lingcore-evals/<candidate> \
  --output /tmp/comparison.json
lingcore eval compare /tmp/lingcore-evals/<baseline> /tmp/lingcore-evals/<candidate> \
  --baseline-grades latest --candidate-grades latest
```

Comparison is offline and still works if the original suite files have moved.
Cases, inputs/revisions, repetitions, execution/trace settings, and selected
graders must agree. Model/profile/runtime changes are allowed and recorded.
Reports separate missing/invalid pairs, and suppress the aggregate delta and
uncertainty when incomplete. Usage includes all attempts separately from the
selected attempts; unknown usage deltas remain null. The bootstrap resamples
cases rather than treating repeated attempts as independent tasks.

Exit status 1 means at least one case's mean score decreased, including when
other cases improved. Status 2 means incompatible or incomplete evidence;
otherwise it is 0. A comparison is a development regression check, not a new
public benchmark eligibility claim. `--output` refuses an existing file;
omit it for JSON on stdout. No source job records are rewritten.
