# Terminal-Bench 4 CPU development subset

Ten explicit tasks from `terminal-bench/terminal-bench@4.0.0`, with one repetition
and one trial running at a time. This is a development sample, not the full
66-task, five-repetition benchmark. Selection covers several task types within
the current adapter's CPU-only capabilities; it is not a random sample or a
prediction that all ten will pass.

| Task | Area | CPUs | RAM (MiB) |
| --- | --- | ---: | ---: |
| `batched-eval-parity` | ML evaluation repair | 2 | 4096 |
| `bun-sourcemap-leak` | Build and release repair | 2 | 4096 |
| `embedding-drift-monitor` | ML monitoring repair | 2 | 4096 |
| `fin-saccr-rwa` | Financial risk calculation | 2 | 4096 |
| `foodstuff-beta-activity` | Scientific data analysis | 2 | 4096 |
| `html-js-filter` | HTML sanitization | 2 | 8192 |
| `mp-checkpoint-consolidation` | Model checkpoint conversion | 4 | 4096 |
| `production-planning` | Operations and database planning | 2 | 4096 |
| `session-window-debug` | Stream processing repair | 2 | 4096 |
| `wal-recovery-ordering` | Storage recovery repair | 2 | 4096 |

CPU/RAM figures are the larger of each task's declared agent and verifier
requirements, per environment. All ten require no GPUs, Linux, a single task
container per environment, no task tools, and no multi-step workflow. The agent
and separate verifier may overlap; their combined requests reach up to 8 CPUs
or 16 GiB RAM, depending on the task. Leave room for the host as well. The current
Linux x86_64, 22-CPU, 32-GiB machine meets those requests. Docker images, build
caches, and retained job artifacts need additional disk space. Each environment
requests at most 10 GiB storage, which Docker does not enforce as a disk quota.
Harbor can also start its own network helper containers.

## Prepare

From the LingCore repository root, install Docker with Compose and Buildx, then:

```sh
uv sync
uv tool install --python 3.12 ./adapters/harbor-bridge
uv run python docs/examples/evaluation/terminal-bench-cpu-10/download.py
```

The downloader uses isolated `harbor@0.25.0` and the bundled release catalogue's
exact package hashes. It downloads only these ten tasks into the adjacent
`dataset/` directory, without starting Docker or making model calls. Existing
task directories are left in place; resolve verifies their contents. Use
`--dry-run` to print commands or `--harbor /path/to/harbor` to reuse an installed
Harbor 0.25.0. Keep `dataset/` beside the suite to avoid parent-relative path
limitations. Downloads, local locks, and `jobs/` are ignored by Git.

Set `LINGCORE_BASE_URL` to your OpenAI-compatible endpoint and `LINGCORE_API_KEY`
to its credential in your shell or an untracked `.env` beside `agent.yaml`.
The model defaults to `gpt-6.1-sol`; `LINGCORE_MODEL` overrides it. The profile
uses `reasoning_effort: high`, so adjust that setting if changing to a model
that does not support it. No key is needed for resolve or validate.

```sh
uv run lingcore eval resolve docs/examples/evaluation/terminal-bench-cpu-10/suite.yaml \
  --lock docs/examples/evaluation/terminal-bench-cpu-10/cpu-10.lock.json
uv run lingcore eval validate \
  --lock docs/examples/evaluation/terminal-bench-cpu-10/cpu-10.lock.json
```

If the bridge is not on PATH, set `LINGCORE_HARBOR_BRIDGE` to its executable or
pass `--bridge /path/to/lingcore-harbor-bridge` to each evaluation command.
Resolve verifies publisher digests and freezes the task inputs, candidate, and
settings. The partial export is recorded as `verified_subset`. Keep the lock
beside the suite. Changing the model, endpoint, profile, or suite requires a new
lock (or `resolve --replace` before starting a new job).

## Run

This command makes paid model calls:

```sh
uv run lingcore eval run \
  --lock docs/examples/evaluation/terminal-bench-cpu-10/cpu-10.lock.json \
  --output docs/examples/evaluation/terminal-bench-cpu-10/jobs
```

The CLI prints the job directory. To continue an interrupted job or read results:

```sh
uv run lingcore eval resume docs/examples/evaluation/terminal-bench-cpu-10/jobs/<job-id>
uv run lingcore eval report docs/examples/evaluation/terminal-bench-cpu-10/jobs/<job-id>
```

The suite schedules ten trials with no automatic infrastructure retries. Resume
may rerun an interrupted attempt. SDK/stream retries remain enabled (three each).
Each task retains its eight-hour agent allowance, with LingCore stopping after
28,650 seconds to reserve 150 seconds for bridge cleanup. Verifier, setup, and
build deadlines remain task-defined. There is no request, tool-call, total-token,
or client response-token cap. The loop's required integer ceiling is 1,000,000;
memory compacts within 128,000 tokens and 2,000 messages. Terminal commands default
to 15 minutes and may request up to eight hours, bounded by the trial deadline.
Traces allow 256 MiB, with 64 MiB per trace artifact.

These are the relaxed settings used for the successful Sol `batched-eval-parity`
run. At the observed approximately ¥0.2 for that one trial, ten similar trials
would be approximately ¥2, but task length, retries, and cache hits vary; this
suite has no monetary spending cap. Preparation and lock validation do not
establish that every container will start or that the model will pass every task.

Offline preparation was checked on 2026-10-31 using Harbor 0.25.0: all ten
downloaded packages matched the publisher catalogue, both agent and verifier
resource requests met the CPU-only bounds above, and resolving and validating
the ten-trial plan reported zero drift. No model calls were made for this check.
