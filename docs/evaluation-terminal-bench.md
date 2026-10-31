# Terminal-Bench through Harbor

LingCore runs locked terminal tasks through an isolated Harbor 0.25.0 bridge.
The integration supports CPU-only Linux tasks with one task container per
environment and no task-provided tools. GPU, multi-container, and multi-step
workflows are not supported. Use the
[CPU development subset](examples/evaluation/terminal-bench-cpu-10/README.md)
for a concrete setup, or the [evaluation examples](examples/evaluation/README.md)
for suite configuration and lifecycle commands.

## Installation and workflow

The bridge requires Python 3.12 or newer and its own environment. Never install
Harbor into LingCore's environment or loosen either package's dependency pins.
Docker runs require Compose and Buildx, including when task images are prebuilt.

```sh
uv tool install --python 3.12 ./adapters/harbor-bridge
uv run python docs/examples/evaluation/terminal-bench-cpu-10/download.py
lingcore eval resolve docs/examples/evaluation/terminal-bench-cpu-10/smoke.yaml \
  --lock docs/examples/evaluation/terminal-bench-cpu-10/smoke.lock.json
lingcore eval validate \
  --lock docs/examples/evaluation/terminal-bench-cpu-10/smoke.lock.json
```

Downloading exports the pinned tasks locally. Resolve and validate use that
export without containers, network requests, or model calls. Set
`LINGCORE_HARBOR_BRIDGE` or pass `--bridge` if the bridge is not on PATH.
Dataset paths must be absolute without symlink components or confined beneath
the suite directory; parent-relative paths containing `..` are refused.

After configuring the model endpoint and credentials, run the locked suite:

```sh
lingcore eval run \
  --lock docs/examples/evaluation/terminal-bench-cpu-10/smoke.lock.json \
  --output docs/examples/evaluation/terminal-bench-cpu-10/jobs
lingcore eval resume docs/examples/evaluation/terminal-bench-cpu-10/jobs/<job-id>
lingcore eval report docs/examples/evaluation/terminal-bench-cpu-10/jobs/<job-id>
lingcore eval regrade docs/examples/evaluation/terminal-bench-cpu-10/jobs/<job-id>
```

Run and resume make model calls. Regrade starts only a separate verifier against
saved artifacts. Model/provider changes require a fresh lock. For native OpenAI
Responses, select `llm.backend: openai-responses` in a copied agent profile;
existing `openai` profiles use Chat Completions. Credentials stay in the profile
or process environment and never enter suites, locks, or task containers.

## Runtime ownership

LingCore owns the model loop, profile, credentials, budgets, trace, usage,
immutable attempt ledger, and retry policy. The bridge never imports LingCore
or receives model credentials. It runs one Harbor trial per execution and
relays terminal operations through a private Unix socket. Harbor provisions the
environments, collects task artifacts, and runs the verifier.

The `harbor-terminal-v1` binding exposes only `terminal_exec` and
`terminal_read_output` inside the task environment. It provides no host shell
or filesystem fallback. Command output and process lifetimes are bounded;
the bridge kills agent-started processes before artifact collection and
verification. Failure to establish cleanup invalidates grading.

Harbor retries are disabled. LingCore may retry declared infrastructure failures
from fresh state, independent of rewards. Task failures, agent budget exhaustion,
and cleanup failures do not cause automatic retries. Auth/configuration errors
stop the invocation. After fixing credentials, resume can continue from the
interrupted attempt without rewriting earlier records.

## Evidence, rewards, and comparison

Each execution retains Harbor config/results, verifier logs, collected artifacts,
and any ATIF trajectory alongside LingCore's trace and normalized result.
Oversized artifacts are marked `oversized` without invalidating a valid verifier
result; unsafe collection paths still fail. Regrading requires the necessary
artifacts to have been preserved and refuses changed agent inputs or shared
verifiers. New grades are stored separately from original results.

Harbor supplies named rewards and its error policy. LingCore combines them with
execution and cleanup outcomes to determine validity. `mean_reward` averages
valid trials within each task, then across tasks. `harbor_accuracy` uses the
planned denominator and counts missing/invalid trials as zero. Reports expose
coverage, phase timings, selected/all-attempt usage, and unknown costs as null.

ATIF v1.7 export requires captured redacted conversation/tool content. Metadata
traces alone cannot produce a complete trajectory. Truncation, missing usage,
and incomplete content remain explicit; reasoning and missing observations are
never synthesized. Offline `eval compare` requires compatible task inputs,
repetitions, execution/trace policy, and selected grader identities. See the
[comparison workflow](examples/evaluation/README.md#compare-preserved-jobs).

## Release provenance and benchmark eligibility

Terminal-Bench 4.0.0 packages are checked offline against the
[reviewed publisher catalog](../lingcore/evaluation/harbor/releases/README.md).
Locks record the Hub version, package hashes, release source commit, and catalog
digest. A partial export is `verified_subset`. Modified tasks require a distinct
local dataset name; other unverified datasets remain usable for development.

The full benchmark policy requires all 66 tasks, five repetitions, the published
reward policy, original grading, task-defined budgets, pinned images, required
traces, and `training_use: prohibited`. Evaluator wall/request/tool/token caps
are explicitly disabled, while Harbor enforces task timeouts. Capability checks
still apply, so the [full-plan example](examples/evaluation/terminal-bench-benchmark.yaml)
refuses unsupported tasks rather than dropping them.

`benchmark_score` is supplied only for a complete eligible original run with
valid selected grades and traces, successful cleanup, and no recorded
reproducibility limitations. Development subsets and regrades cannot claim
benchmark compatibility. Local eligibility is not upstream submission approval.

Reports distinguish locked agent/verifier resource requests from backend
enforcement. Docker enforces CPU and memory limits, but not storage quotas.
These fields describe requests, not measured utilization.

## Automated checks

The nightly Harbor CI job and opt-in Docker tests use synthetic tasks and fake
model clients. Public model evaluations remain manual. Run bridge checks in
its own environment as described in the [bridge README](../adapters/harbor-bridge/README.md).

```sh
LINGCORE_HARBOR_E2E=1 LINGCORE_HARBOR_BRIDGE=/path/to/lingcore-harbor-bridge \
  uv run pytest tests/test_harbor_e2e.py -q
```
