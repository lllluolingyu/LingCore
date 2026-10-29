# Milestone 2: locked evaluation plans

Status: implementation plan, 2026-10-30; step 2.1 is done. It expands
[milestone 2 of the tracing and evaluation design](tracing-and-evaluation.md#milestone-2-locked-plans)
and builds on the milestone 1 native evaluator in `lingcore/evaluation/`.

## Goal

Milestone 1 runs whatever the suite files say at the moment of execution. Two
runs of the same suite may silently use different cases, prompts, models, or
LingCore code, so their scores cannot be compared. A crashed job has to start
over, and a corrected grader means running the agent again.

Milestone 2 freezes the plan into a lock file before execution. A locked run
refuses to start if anything the lock describes has changed. An interrupted
job resumes with only its unfinished trials. A corrected grader rescores the
saved evidence without running the agent.

Out of scope: the Harbor adapter (milestone 3), `eval compare` and statistics
(milestone 4), parallel trials, cost estimates, and exporters. The lock format
leaves room for milestone 3 through a source discriminator.

## Vocabulary

| Term | Meaning |
| --- | --- |
| Plan | Everything that determines a job: cases, candidate, runtime, limits, graders, ordered trials |
| Lock | A file recording the plan's digests and versions; never secret values |
| Job | One output directory created by `eval run`, possibly continued by `eval resume` |
| Trial | One planned `(case, repetition)`; its ID comes from the lock |
| Execution | One attempt at a trial; a trial may have several |
| Selected execution | The execution that counts for the trial under the retry policy |
| Grade set | The grades of every execution from one grading pass; the run's own pass is `original` |

## Workflow

```sh
# Freeze the plan. Prints the planned trials and limits; no model call.
lingcore eval resolve docs/examples/evaluation/local.yaml --lock local.lock.json

# Later, or on another machine: check that nothing drifted. Offline.
lingcore eval validate --lock local.lock.json

# Run exactly that plan. Refuses drift before creating any state.
lingcore eval run --lock local.lock.json --output /tmp/lingcore-evals

# After a crash or Ctrl-C: run only the unfinished trials.
lingcore eval resume /tmp/lingcore-evals/<job-id>

# After correcting a grader: rescore saved evidence; the agent never runs.
lingcore eval regrade /tmp/lingcore-evals/<job-id>
lingcore eval report /tmp/lingcore-evals/<job-id> --grades latest
```

A lock pins one candidate on one LingCore runtime. To measure a LingCore or
profile change, resolve one lock before the change and one after. Milestone 4's
comparison requires their task, grading, execution, and trace sections to match,
and reports the candidate and runtime differences.

The unlocked `lingcore eval run <suite>` from milestone 1 keeps working for
quick development, and its results stay `locked: false`.

## Decisions

- **Identity is always computed.** Resolution, locked runs, and unlocked runs
  all derive the same identity. Only locked runs enforce it, but an unlocked
  job still records it in `plan.json`, so it can be regraded.
- **Digests cover parsed values, not file bytes.** Suite and case digests use
  the validated model with defaults filled in. A comment or reordered key is
  therefore not drift, while a changed default is. Source files that assembly
  reads verbatim (prompt layers, skills, commands) are hashed as bytes.
- **Candidate identity is measured.** Resolution assembles the Agent offline
  and hashes what the model would actually receive: the initial system prompt
  and every tool schema in the ceiling.
- **Cases have two digests.** `input_digest` covers what the agent sees
  (turns, fixture, requirements, revision). `graders_digest` covers the
  evaluator-only graders. `run` refuses a change to either, and `regrade`
  refuses only an input change.
- **Two lock digests.** `digest` covers the whole file for integrity.
  `plan_digest` covers only identity, excluding paths, timestamps, and
  limitations. Moving a suite directory or re-resolving unchanged inputs
  therefore keeps `plan_digest`.
- **Interrupted executions are replaced on resume** without using up the
  infrastructure retry allowance, and are counted in every report.
- **Cleanup failure is never retried.** An agent can make cleanup fail, for
  example by leaving undeletable files, so a retry would give a failing agent
  a second chance. Such a trial is invalid.
- **Grader versions are integers declared in code.** Any change to grading
  semantics bumps the version. Lock and grade records store them.

## Lock file

Canonical JSON: UTF-8, sorted keys, separators `,` and `:`, no NaN, written
with a trailing newline. Every digest is `sha256:<hex>` over canonical JSON or
raw bytes. Paths are relative to the lock file's directory (they may start
with `..`). Absolute paths and home directories never appear.

```json
{
  "schema_version": "lingcore.eval-lock.v1",
  "digest": "sha256:…",
  "plan_digest": "sha256:…",
  "created_at": "2026-10-30T09:00:00Z",
  "suite": {"name": "native-file-smoke", "track": "development",
            "path": "local.yaml", "digest": "sha256:…"},
  "source": {
    "kind": "local",
    "digest": "sha256:…",
    "cases": [{
      "id": "write-greeting",
      "revision": 1,
      "path": "cases/write-greeting.yaml",
      "input_digest": "sha256:…",
      "graders_digest": "sha256:…",
      "graders": [{"id": "artifact", "kind": "artifact_text", "version": 1},
                  {"id": "tool_policy", "kind": "tool_policy", "version": 1}],
      "evidence": {"artifacts": ["hello.txt"], "content": true}
    }]
  },
  "candidate": {
    "digest": "sha256:…",
    "profile": {"path": "local-agent.yaml", "config_digest": "sha256:…"},
    "model": {"backend": "openai", "id": "deepseek-chat",
              "endpoint": "api.deepseek.com"},
    "vision_model": null,
    "sampling": {"temperature": null, "max_tokens": null},
    "prompt_digest": "sha256:…",
    "tools": {
      "ceiling": [{"name": "list_dir", "schema_digest": "sha256:…"}],
      "initial": ["list_dir", "read_file", "write_file"]
    },
    "sources": {"local-agent.yaml": "sha256:…"},
    "credentials": [{"name": "LINGCORE_API_KEY", "set": true}]
  },
  "runtime": {
    "digest": "sha256:…",
    "lingcore": "0.5.1",
    "package_digest": "sha256:…",
    "git": {"commit": "26fb02d…", "dirty": false},
    "python": "CPython 3.13.15",
    "dependencies": {"anthropic": "…", "openai": "…", "pydantic": "…"}
  },
  "execution": {
    "digest": "sha256:…",
    "runner": "native", "state": "fresh_per_execution",
    "approval_policy": "deny", "repetitions": 2, "order": "declared",
    "concurrency": 1,
    "retry": {"infrastructure_retries": 0, "retry_on": ["runner", "environment"]},
    "limits": {"wall_seconds": 60, "model_requests": 8, "tool_calls": 8,
               "reported_tokens": 32000, "usage_missing": "stop",
               "automatic_turns": 0}
  },
  "trace": {"required": true, "content": "redacted",
            "max_bytes": 8388608, "artifact_max_bytes": 1048576},
  "grading": {"digest": "sha256:…", "authority": "local",
              "success": "all_required"},
  "trials": [
    {"trial_id": "write-greeting--candidate--1", "case_id": "write-greeting", "repetition": 1},
    {"trial_id": "write-greeting--candidate--2", "case_id": "write-greeting", "repetition": 2}
  ],
  "limitations": []
}
```

Section rules:

| Section | Contents and drift meaning |
| --- | --- |
| `suite` | Name, track, and the parsed suite digest |
| `source` | Discriminated by `kind`; `local` lists cases in declaration order. Milestone 3 adds `harbor` with task, image, and verifier identity |
| `candidate.profile.config_digest` | Resolved profile `model_dump(mode="json")` without runner-owned `workspace` and `tracing`, with each `base_url` replaced by its endpoint |
| `candidate.model` | Backend, resolved model ID, endpoint. The endpoint is the URL host and port, or the suite's `agent.endpoint_alias`; query, userinfo, and path are dropped because proxies put tokens there |
| `candidate.prompt_digest` | Turn-0 system prompt from offline assembly with an empty user message |
| `candidate.tools` | `Tool.json_schema()` digest per ceiling tool, plus the initially enabled names |
| `candidate.sources` | Bytes of every source file assembly reads: the profile YAML, `world.md`/`role.md`/`workflow.md`, `persona.include`, engaged profile-local skill directories, `commands/`, and engaged profile-local plugin roots. Installed plugins record their distribution version and root digest; bundled ones are covered by the runtime |
| `candidate.credentials` | Names from `api_key_env`, `token_env`, and retrieval provider settings, and whether each is set. Never values |
| `runtime.package_digest` | Sorted `(relative path, file digest)` pairs for every file under the installed `lingcore` package, excluding `__pycache__` |
| `runtime.git` | Best effort, informational: commit and dirty flag when the package lives in a git checkout. `null` otherwise. The package digest is what detects drift |
| `runtime.dependencies` | Installed versions of the distributions `lingcore` requires |
| `execution`, `trace`, `grading` | Validated suite policy with defaults filled in |
| `trials` | Explicit ordered list; resume and reports use it, never re-derive it |
| `limitations` | `runtime_dirty`, `git_unavailable`, `credential_unset`, `plugin_unversioned`; informational, not drift |

`resolve` refuses a plan in which any known secret value appears in the
canonical lock body, which invariant 4 already forbids. Credentials unset at
resolve time produce `credential_unset` rather than an error, so a lock can be
made on a machine without keys. `run` and `resume` require them.

## Job directory

```text
<output>/job-<id>/
  .lock                     # held by run, resume, and regrade; refuses a second user
  job.json                  # immutable: mode, lock digests, original lock location
  lock.json                 # exact bytes of the lock used (locked jobs only)
  plan.json                 # immutable identity; unlocked jobs record it too
  executions/<execution-id>/
    trial.json              # written at start
    trace/                  # journal written here directly, not copied afterwards
    evidence.json           # written at collection: what was preserved
    artifacts/<digest>.bin  # collected artifact bytes, content-addressed
    result.json             # written at the end; never modified
    interrupted.json        # written by resume when result.json is missing
    grades/<grade-set>.json # one per regrade pass
  grade-sets/<grade-set>.json
  report.json               # derived view of the original grade set; atomically replaced
  reports/<grade-set>.json  # derived view per regrade pass
```

Every write goes through `confined_directory` with private permissions.
Immutable records use exclusive create; derived files use descriptor-relative
atomic replace. Nothing in `executions/` is rewritten after it is written.

`trial.json` gains `execution_id`, `attempt` (1-based within the trial,
assigned while holding `.lock`), `started_at`, the lock's `plan_digest`, and
the scratch directory path so that resume can remove it.

`evidence.json` lists trace completeness, content mode, and every collected
artifact: `{path, status: present|missing|unsafe, digest, bytes}`. A missing
file is evidence (the agent never wrote it); a path never collected is not.

`result.json` keeps the milestone 1 shape and adds `attempt`, `locked`,
`lock_digest`, `plan_digest`, `infrastructure_failure` (the matched `retry_on`
cause or `null`), and `served_models` (sorted distinct models that providers
reported in usage).

## Steps

Each step ships with tests and leaves the suite, Ruff, and mypy green. Step
2.1 is done. Steps 2.2 and 2.3 build the lock, and 2.4 needs
both. Steps 2.5 and 2.6 need 2.4. Step 2.7 needs only 2.2 and can proceed in
parallel with 2.5 and 2.6. Step 2.8 closes the milestone.

### 2.1 Milestone 1 follow-ups

Done from the milestone 1 review: narrower secret redaction, propagated
cancellation, deterministic `timed_out`, visible preflight reasons, grader
failures that never rewrite the execution outcome, capture accounting, partial
`limits:` blocks keeping their defaults, the stale help test, and
documentation drift. The design doc lists them with acceptance criteria;
`tests/test_evaluation_regressions.py` covers each one.

### 2.2 Plan identity

New `lingcore/evaluation/identity.py`:

- `canonical_json(value) -> bytes` and `digest(data) -> str`.
- `case_identity(case) -> CaseIdentity` with `input_digest`, `graders_digest`,
  grader versions, and evidence needs from 2.7. Tags are excluded from both
  digests.
- `candidate_identity(profile, suite) -> CandidateIdentity`. Offline assembly
  calls `Agent.from_profile` with a client for every model role that raises
  if called, a throwaway state directory, and a throwaway workspace. It then
  composes once and reads the ceiling's schemas, and closes the agent in
  `finally`. Add two small public read-only accessors on `Agent` rather than
  reaching into private members: `initial_system_prompt()` and
  `ceiling_tool_schemas()`.
- `runtime_identity() -> RuntimeIdentity`. The git probe runs
  `git rev-parse HEAD` and `git status --porcelain --untracked-files=no -- lingcore`
  with a short timeout and no shell, and returns `null` on any failure.
- `assert_secret_free(data, secrets)`, using the narrowed secret set from 2.1
  so tracing and locks agree on what a secret is.

Grader classes gain `kind` and `version` attributes, collected into a
registry the lock and regrade read.

Tests (`tests/test_eval_identity.py`): two resolutions of unchanged inputs are
byte-identical apart from `created_at`. A table-driven test changes one input
at a time (turn text, fixture, revision, grader spec, profile YAML value,
`world.md`, an included file, a tool description, `LINGCORE_MODEL`, the
endpoint, a limit, a package file) and asserts exactly one section digest
moves. A changed comment moves nothing. A fixture API key in the environment
never appears in the output, and a secret expanded into `tool_options` is
refused. A client that raises proves no model call is made.

### 2.3 Lock file, `eval resolve`, `eval validate`

New `lingcore/evaluation/lock.py`:

- Pydantic models for `lingcore.eval-lock.v1` with `extra="forbid"` and a
  `source` union keyed by `kind`. An unknown major schema is rejected; an
  unknown additive field in a newer minor revision is rejected too, because a
  lock is an enforcement document.
- `resolve(suite_path, lock_path) -> Lock`. Preflight checks are the
  milestone 1 ones, the evidence checks from 2.7 once it lands, and capture minimums from
  `source.lock.capture` (an unknown name is an error; native resolution
  always captures every section).
- `write_lock(lock, path, replace=False)`: exclusive create by default;
  `--replace` uses atomic replace.
- `load_lock(path) -> Lock`: bounded read, schema check, and self-digest
  verification.
- `validate(lock, lock_path) -> list[Drift]`. Each `Drift` is a dotted field
  name and a kind (`changed`, `missing`, `added`, `unreadable`) and never
  carries a value. Case drift names the case ID.

CLI: `eval resolve <suite> --lock <path> [--replace]` prints the trial count,
cases with repetition counts, the model ID and endpoint, limits, and
limitations. `eval validate --lock <path>` prints `ok` or the drift list.

Tests (`tests/test_eval_lock.py`): resolve and validate the shipped example
against environment fixtures. Every drift class from 2.2 appears by name with
exit 2. A truncated lock, a hand-edited value with a stale digest, an
unknown major version, and an existing lock path without `--replace` are
rejected. Relative paths survive moving the suite and lock together.

### 2.4 Lock-checked run

`eval run --lock <path>`:

1. Load and validate the lock. Any drift exits 2 before creating the job
   directory, so zero executions exist.
2. Create the job, copy the lock bytes to `lock.json`, and write `job.json`
   with `mode: locked_native`, both digests, and the absolute lock location.
   The location lives only in the job, never in the lock.
3. Acquire `.lock`, then execute trials in lock order using the lock's
   trial IDs.

`run_suite` is split into plan construction (identity plus trial list, from
either the suite or the lock) and execution, so both modes share one executor.
`TrialResult.locked` and `Comparison.locked` become `bool`.
`benchmark_compatible` stays `false` on the development track.

`eval run <suite>` remains unlocked and refuses `lock.required: true`. Its
`plan.json` now carries the full identity.

Tests: a drifted lock leaves the output directory empty. Two jobs from one lock
share `plan_digest`. Unlocked results stay `locked: false` and still carry
identity. A second `run` on a held job lock is refused.

### 2.5 Job ledger and resume

New `lingcore/evaluation/jobs.py`:

- `scan(job) -> JobState`: reads every execution directory and groups records
  by trial, ordered by `attempt`. An execution is `finished` (has
  `result.json`), `interrupted` (has `interrupted.json`), or `orphaned` (only
  `trial.json`).
- `select(trial_executions, policy) -> Selection` from 2.6.
- `pending(lock, state, policy) -> list[TrialSpec]`: trials whose selection is
  unresolved and which may receive another execution, in lock order.

The runner's trace sink writes straight into `executions/<id>/trace/` through
an injected recorder, instead of the scratch state directory followed by a
copy. A crash then leaves a partial journal that resume can inspect. The job
directory still never contains the trial workspace.

`eval resume <job> [--lock <path>]`:

1. Acquire `.lock`; refuse if held.
2. Load `lock.json` and resolve its relative paths from the recorded lock
   location. If the files moved, `--lock` names the lock's new location, and
   that file must have the job's digest. Validate against current inputs, and
   refuse on drift.
3. For each orphaned execution, write `interrupted.json` with the detection
   time and the partial journal's request and tool counts from
   `inspect_trace`. Then remove its recorded scratch directory if it still
   exists, has the `lingcore-eval-` prefix, and belongs to the current user.
   Removal failure is reported, not fatal.
4. Run `pending` trials, then rewrite `report.json`.

An unlocked job cannot be resumed; the command says to resolve a lock.

Tests (`tests/test_eval_jobs.py`): a trial killed mid-run leaves a readable
partial journal in its execution directory. Run a two-case job, delete one
`result.json`, and resume. Exactly one new execution runs, with `attempt: 2`,
and the old one gains `interrupted.json`. Resuming a complete job runs nothing
and exits with the job's code. Drift, a held lock, and an unlocked job are
refused. Resume never modifies an existing record (compare bytes before and
after).

### 2.6 Infrastructure retries and attempt selection

Suite schema change, backward compatible with `infrastructure_retries: 0`:

```yaml
execution:
  infrastructure_retries: 1        # extra executions per trial, 0-5
  retry_on: [runner, environment]  # default; add provider_transient to opt in
```

Provider error categories. Add `category` to `LLMStreamError` with the values
`transient`, `request`, `auth`, and `interrupted`. `llm.py` and
`llm_anthropic.py` classify open failures from the SDK exception type:

| Category | SDK failures |
| --- | --- |
| `transient` | Rate limit, server error, connection, timeout after the SDK retry budget |
| `request` | Bad request, not found, unprocessable, context too large |
| `auth` | Authentication and permission errors |
| `interrupted` | Mid-stream failures and truncation (the current `retryable=True`) |

A foreign exception from a duck-typed client stays `unclassified`. The loop
copies the category into `ExecutionOutcome.provider_error_category` (keeping
invariant 1: the category is LingCore's own enum, not an SDK type).

Classification of a finished execution into a `retry_on` cause:

| Execution | Cause | Retryable when listed |
| --- | --- | --- |
| LingCore runner exception before or during the trial | `runner` | yes |
| Scratch or state preparation failure | `environment` | yes |
| Provider `transient` or `interrupted` after the loop's own retries | `provider_transient` | yes |
| Provider `request` | none | never; agent failure |
| Provider `auth`, configuration error | none | never; stops scheduling, job incomplete |
| Agent error, `budget_exceeded`, `timed_out`, completed with any grade | none | never |
| Cleanup failure | none | never; trial invalid |
| Interrupted (resume) | none | replaced without using the allowance |

Selection: walk a trial's executions in attempt order and skip interrupted
ones and retryable infrastructure failures. The first remaining execution is
selected, whatever its grade. If none remains and retryable failures exceed
`infrastructure_retries`, the trial is invalid with `retries_exhausted`. The
runner schedules a retry immediately after a retryable failure while the
allowance lasts.

Tests: a table over the classification matrix. A factory that fails the first
execution with a runner error then succeeds selects attempt 2. A failing grade
on attempt 1 is never retried. With scripted attempts (pass after infra
failure, fail, pass), the selection is never the later pass. Exhaustion gives
`retries_exhausted`. `auth` stops the remaining trials, and the report says
why. Category tests for both backends use the SDKs' own exception classes.

### 2.7 Evidence declarations and regrade

Graders declare the evidence they need:

| Grader | Needs |
| --- | --- |
| `artifact_text` | The artifact path collected |
| `tool_policy` | Complete trace metadata; captured content when any `require` has `arguments` |

Optional case field `evidence.artifacts: [path, …]` collects extra paths for
graders that do not exist yet. Collection covers the union of declared and
grader paths, bounded per file by the grader `max_bytes` (default 1 MiB).

Preflight, in `resolve`, `validate`, and the unlocked run, refuses a
`tool_policy` with argument predicates under `trace.content: metadata`, and an
artifact path outside the workspace.

`eval regrade <job> [--lock <path>]`:

1. Acquire `.lock`. Load the current case files from the job's lock (or
   `plan.json` for an unlocked job).
2. Refuse, by case ID, any case whose `input_digest` differs from the job's.
   Grader changes are the point and are recorded.
3. For each finished execution, check the graders' needs against
   `evidence.json`. Refuse, by case and grader ID, any need that was never
   preserved, such as a new artifact path. An execution whose trace was
   incomplete still regrades: the result is `not_evaluable`, as at run time.
4. Write `grades/<grade-set>.json` per execution, `grade-sets/<grade-set>.json`
   (graders, versions, spec digests, time), and `reports/<grade-set>.json`.
5. Never call `Agent`, a model client, or a tool. A test injects a factory
   that raises to prove it.

Artifacts move from `artifacts/<index>.json` (hex) to content-addressed
`artifacts/<digest>.bin` indexed by `evidence.json`. Files are only ever read
back as bounded bytes, never executed or rendered.

Tests (`tests/test_eval_regrade.py`): correcting `equals` in the example's
grader produces a new grade set while `result.json` stays byte-identical.
A changed turn is refused. A new artifact path is refused. A missing file at a
preserved path regrades as a valid failure. Argument predicates under
metadata capture fail preflight.

### 2.8 Reports and exit codes

Report schema `lingcore.eval-report.v2`:

- Provenance: `locked`, `lock_digest`, `plan_digest`, `grade_set`.
- Coverage adds `interrupted`, `replaced`, `retried`, and `retries_exhausted`.
- Per trial: `attempts: [{execution_id, attempt, status, cause, selected}]`.
- Usage: `selected` and `all` totals, because retried executions were billed
  too.
- `served_models`, with `served_model_differs: true` when a provider reported
  a model other than the lock's.
- Scores use selected executions only, and the milestone 1 headline rules are
  unchanged.

`eval report <job> [--grades original|latest|<id>]` reads v1 and v2 reports.
Resume and regrade refuse v1 jobs because they carry no identity.

Exit codes:

| Command | 0 | 1 | 2 | 130 |
| --- | --- | --- | --- | --- |
| `resolve` | Lock written | — | Preflight failure, existing lock | Ctrl-C, no lock written |
| `validate` | No drift | — | Drift, unreadable or corrupt lock | Ctrl-C |
| `run`, `resume` | Complete, all valid trials pass | Complete, some valid failures | Drift, incomplete, invalid, held lock | Ctrl-C, records kept |
| `regrade` | Complete, all pass | Complete, some failures | Refused, incomplete | Ctrl-C, partial grade set marked incomplete |
| `report` | As the report | As the report | Unreadable | — |

Every command prints the job path, when one exists, before exiting on every
path, including Ctrl-C.

Tests: a CLI matrix over each command and exit path, with
`asyncio.run` interruption simulated for the 130 cases.

## Documentation updates

- `docs/tracing-and-evaluation.md`: milestone status, and links to this plan.
- `docs/examples/evaluation/README.md`: the locked workflow, and committing
  lock files beside suites.
- `docs/examples/evaluation/local.yaml`: unchanged `lock.required: false`, so
  both workflows stay runnable.
- `CLAUDE.md`: architecture entries for `identity.py`, `lock.py`, and
  `jobs.py`, the `LLMStreamError.category` contract, and the rule that
  evaluation never retries an agent-influenced failure.
- `README.md` roadmap.

## Acceptance for the milestone

- The local example resolves, validates, runs locked, resumes after a deleted
  result, and regrades a corrected grader, all against fake clients in CI.
- Any change to an input the lock describes is refused by `validate` and
  `run`, by name and without printing values.
- No lock, job record, report, or journal contains a fixture secret.
- Attempt selection never depends on a grade.
- No existing execution record is modified after it is written.
- Full test suite, `ruff check`, `ruff format --check`, and `mypy` pass.
