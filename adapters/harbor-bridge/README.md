# LingCore Harbor bridge

An independent Python ≥3.12 distribution pinned to Harbor 0.25.0. It never
imports LingCore and must be installed in its own environment.

```sh
uv tool install --python 3.12 ./adapters/harbor-bridge
lingcore-harbor-bridge describe --dataset /path/to/exported/dataset
```

`describe` provides one bounded JSON
document with package provenance, Harbor's error policy, supported ATIF
versions, and task metadata measured through Harbor's models. It starts no
containers, subprocesses, or model clients. Invalid task metadata is reported
per task without printing parser errors or input values. Dataset paths are
relative, and task instructions are represented by digests.

`trial` runs exactly one Harbor trial with retries disabled, relaying commands
through a private Unix socket. The relay supervises output and timeouts inside
the container and sweeps agent processes before verification. `summarize`
normalizes Harbor's own result and error policy; `regrade` uses a separate
verifier against preserved artifacts. LingCore owns the model loop, credentials,
usage, lock, and job ledger. See [the implementation plan](../../docs/evaluation-harbor-plan.md).

Run this package's checks in its own environment:

```sh
cd adapters/harbor-bridge
uv sync --locked --python 3.12
uv run pytest -q
uv run ruff check
uv run ruff format --check
uv run mypy
```

LingCore's base environment and dependency lock remain independent. The
optional Docker compatibility checks require `LINGCORE_HARBOR_E2E=1` and use
synthetic tasks; they do not call a model or run a public benchmark.
