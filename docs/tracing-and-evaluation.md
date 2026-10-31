# Tracing and evaluation

Harbor runs in a separate Python environment, with locked local tasks, a
supervised terminal relay, reward reports, resume, separate-verifier regrade,
and ATIF export.
Synthetic Docker tests use fake model clients; public Terminal-Bench runs
remain a manual step. Paired job comparison and the native regression suite are
implemented. See the [Harbor integration guide](evaluation-terminal-bench.md).

The current native subset supports sequential local cases with fresh workspace
fixtures (optionally seeded by `fixture.files`), scripted text turns,
`artifact_text`, and `tool_policy`. It supports locked plans, resume,
infrastructure retries, and regrading saved evidence. External native-case
sources, OpenTelemetry export, and concurrent execution are not supported.
Unknown costs remain null.
Output-allowance capping, trace deletion, and arbitrary fixture copying are not
exposed by this subset.

```sh
lingcore eval run docs/examples/evaluation/local.yaml --output /tmp/lingcore-evals
lingcore eval resolve docs/examples/evaluation/local.yaml --lock local.lock.json
lingcore eval validate --lock local.lock.json
lingcore eval run --lock local.lock.json --output /tmp/lingcore-evals
lingcore eval resume /tmp/lingcore-evals/<job-id>
lingcore eval regrade /tmp/lingcore-evals/<job-id>
lingcore eval report /tmp/lingcore-evals/<job-id>
lingcore eval report /tmp/lingcore-evals/<job-id> --grades latest
lingcore eval compare /tmp/lingcore-evals/<baseline> /tmp/lingcore-evals/<candidate>
lingcore trace show /tmp/lingcore-evals/<job-id>/executions/<execution-id>/trace
```

The model environment names are listed in
[the example setup](examples/evaluation/README.md). CI runs the same example
against a scripted client without paid services. Job directories preserve
per-execution results, bounded output artifacts, traces, and `report.json`;
temporary model workspaces and profile state are cleaned after collection.

`eval compare` reads v2 job identities and rebuilds selected results from the
preserved execution ledger. It requires identical case inputs/revisions,
repetition order, execution/trace policy, and selected grader identities; the
candidate profile, model, and runtime may differ. It does not load live suite
files, start an Agent, or call a provider. `--baseline-grades` and
`--candidate-grades` select `original`, `latest`, or a saved grade-set ID.

Results include paired per-case means, regressions and improvements, selected
and all-attempt usage, and paired latency differences. Deltas are candidate
minus baseline. Missing/invalid pairs retain descriptive differences, but
withhold the overall `mean_delta` and confidence interval. A deterministic
95% percentile bootstrap resamples whole cases with their repetitions together
(2,000 draws, seed 0 by default). One case has no interval; repeated trials do
not create additional independent tasks. Small-suite intervals are descriptive,
not a guarantee of general performance. Exit codes are 0 for a complete result
without a lower per-case mean, 1 when any case regresses, and 2 for incomplete
or incompatible evidence. `--output` writes an exclusive JSON artifact.

Native defaults apply to each omitted limit: 60 seconds, 20 model requests,
and 40 tool calls. Setting another limit does not remove those bounds; explicit
`null` disables a bound. External cancellation propagates from the library
after finalization, cleanup, and best-effort result/report persistence; only
the CLI translates it to exit 130. Grader failures retain the recorded execution
outcome and any independent grader results, with a separate grading diagnostic.

Known-secret replacement uses explicit credential declarations and credential
environment names, not arbitrary exported values. Short credentials match
whole values or credential fields, never substrings of IDs or model names.
Committed content is bounded including item wrappers and separators. Identical
sanitized prompts/tool schemas are captured once per turn; truncation affects
individual values or omitted later items, preserving earlier argument evidence.

The public integration example is
[Terminal-Bench through Harbor](evaluation-terminal-bench.md). A small
[local evaluation example](examples/evaluation/local.yaml) demonstrates the
same result and grading model without Harbor.

## Traces and conversation state

Tracing is opt-in through the profile's `tracing` settings. Metadata capture is
separate from redacted conversation/tool content. The append-only journal
records request usage and lifecycle spans; ATIF is an optional derived export
for Harbor jobs. Missing usage remains unknown rather than estimated.

A failed or cancelled request may have billed usage without a committed
assistant message. Request accounting therefore remains independent of session
history. Stop and failed-turn rollback remove partial conversation state while
retaining the trace of attempted work. Provider reasoning, signed native blocks,
encrypted Responses state, credentials, and environment values are excluded
from trace content.

## Locks, results, and regrading

A lock pins ordered cases, agent/profile/runtime identity, execution policy,
limits, and grader identities. Resolve measures the effective prompt and tool
schemas offline; validate refuses drift before execution. Changing a candidate
requires a fresh lock. Case inputs and graders have separate digests, allowing
regrading against corrected graders while refusing changed agent inputs.

Each job has immutable `job.json`, `plan.json`, and, when locked, `lock.json`.
Its `executions/` ledger keeps each attempt's start, result, evidence, and trace.
Resume replaces interrupted attempts from fresh state, never overwrites them,
and holds a job lock. Infrastructure retries are independent of task scores;
failed grades and exhausted agent budgets never trigger an automatic retry.
New grade sets and their reports live alongside the original results.

Native regrading reads saved evidence without tools or model calls. Harbor
regrading runs a separate verifier in Docker against the preserved artifacts.
Artifacts exceeding the collection limit are marked `oversized`; unsafe paths
still invalidate collection. Missing evidence cannot be reconstructed from a
completed conversation.

For the supported YAML fields, lifecycle commands, exit codes, and comparison
options, use the [evaluation examples](examples/evaluation/README.md). The
[native regression suite](examples/evaluation/regression.yaml) covers retrieval,
calendar recurrence, and multi-turn file work with deterministic graders.
