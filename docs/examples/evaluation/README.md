# Evaluation examples

These files illustrate the
[tracing and evaluation design](../../tracing-and-evaluation.md).
`lingcore eval run local.yaml` is implemented for the native suite. The Harbor
suite and its `terminal_exec` / `terminal_read_output` bindings remain proposed;
they are not available in LingCore v0.5.1.

| File | Purpose |
| --- | --- |
| [terminal-bench.yaml](terminal-bench.yaml) | Public Terminal-Bench 4.0 development subset through Harbor |
| [terminal-agent.yaml](terminal-agent.yaml) | Dedicated environment-only agent profile for that adapter |
| [local.yaml](local.yaml) | Native evaluation suite using a small file-writing task |
| [local-agent.yaml](local-agent.yaml) | Minimal profile for the native example |
| [cases/write-greeting.yaml](cases/write-greeting.yaml) | Agent input, required tools, and evaluator-only assertions |
| [.env.example](.env.example) | Secret-free list of the environment names both profiles need |

Suite and case YAML use the `lingcore.eval.*.v1` schemas. Agent YAML
uses the current profile shape, but terminal tools require the proposed
adapter binding. Keep suite/case files outside the model-visible workspace.

All paths resolve relative to the YAML file that declares them. Agent profile
environment expansion follows the current profile-local rules; set
`LINGCORE_MODEL`, `LINGCORE_BASE_URL`, and `LINGCORE_API_KEY` in the process
environment, or a private `.env` beside a copied profile
(see [.env.example](.env.example)). No credentials belong in a suite, lock,
journal, or benchmark environment.

The native suite runs under milestone 1 as an unlocked
development plan, so it sets `lock.required: false`. The Terminal-Bench suite
requires a lock and therefore needs milestone 2 before the adapter
(milestone 3) can run it.

After configuring the model environment, run:

```sh
lingcore eval run docs/examples/evaluation/local.yaml --output /tmp/lingcore-evals
lingcore eval report /tmp/lingcore-evals/<job-id>
lingcore trace show /tmp/lingcore-evals/<job-id>/executions/<execution-id>/trace
```

Exit codes are 0 for passing trials, 1 for valid task failures, 2 for invalid or
incomplete evaluations/configuration, and 130 for interruption. Each result
records `locked: false` and `benchmark_compatible: false`. The current native
subset supports empty workspace fixtures, sequential trials, and the two
graders in this example. Its deterministic integration test uses fake clients.

For the same example with fake clients and no provider calls:

```sh
uv run pytest tests/test_evaluation.py::test_supplied_local_example_runs_with_fake_client_and_fresh_state -q
```

Omitted native limits default independently to 60 seconds, 20 model requests,
and 40 tool calls, including when the suite sets other limit keys. An explicit
`null` disables a limit. Preflight refusals include a safe reason in the CLI and
`report.json`. Library callers receive external cancellation after reports and
cleanup; the CLI translates interruption to exit 130.

The runner creates a fresh private profile/workspace for every execution;
these example directories never become writable task state. The terminal
profile's host workspace contains no task data and is inaccessible to tools.
Harbor owns the actual task filesystem.

The source version is pinned, but no fabricated task or harness lock is
included. The proposed resolver must capture the exact installed Harbor
version, task selection, images, profile fingerprint, and verifier identities
before a run. Lock validation refuses changes.

The Terminal-Bench subset uses reduced budgets and alphabetical selection only
to test the integration. It cannot produce a full-benchmark score. No tasks
have been downloaded or run as part of this design.
