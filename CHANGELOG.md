# Changelog

Notable user-facing changes to LingCore are documented here. The project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-20

### Added

- Knowledge 1.0: offline grep plus opt-in indexed and hybrid retrieval, stable
  source chunks and citations, incremental content-hash reuse, stale/deleted
  source handling, and provider-neutral embedding and reranking seams.
- Durable session runtime events for compaction snapshots and dynamic skill
  state, with bounded snapshot retention and validation against the canonical
  transcript on restore.
- Stable message/event cursors, stop-safe truncation, editable user-message
  rewind, and atomic session-prefix forks with parent/root provenance.
- Explicit turn cancellation through `cancel_turn()`,
  `finalize_cancelled_turn()`, and the `TurnCancelled` frontend event.
- Multimodal attachment ingest with workspace copies, native image/PDF support,
  text and binary fallbacks, and optional PDF/image-to-text conversion.
- Profile-scoped `.env` loading, offline `lingcore doctor` diagnostics,
  layered prompt composition, persistent memory tooling, dynamic skills, and
  bundled coding, daily, teaching, Canvas, and Ollama examples.

### Changed

- Context windows now use block-aware, prefix-stable eviction and optional
  summarize-then-evict compaction. Persisted snapshots allow bounded resume
  hydration when a valid compaction is available.
- Restored dynamic skills are intersected with the current profile ceiling and
  their recorded high-risk approvals, preventing permission drift from
  widening consent.
- Stored user messages retain the accepted user-authored text separately from
  model-facing attachment notes so frontends can edit and regenerate cleanly.
- Model streaming has typed retry classification, mid-stream recovery events,
  bounded retries, and prompt-cache routing support.

### Fixed

- Unexpected guardrail, persistence, compaction, skill-state, and other turn
  failures now roll back partial state, emit an `Error`, and release the turn
  lease instead of permanently wedging the agent. `CancelledError` retains the
  deliberate two-step cancellation contract, while `aclose()` repairs an
  abandoned generator.
- Hardened workspace, attachment, knowledge-index, Canvas, shell, and web-fetch
  paths against traversal, symlink races, unbounded output, DNS rebinding, and
  unsafe authorization fallbacks.

### Compatibility

- Requires Python 3.11 or newer.
- Existing schema-v1 session databases migrate additively to schema v2 when
  opened. Canonical message history remains intact.
- Direct `Agent(system_prompt=...)` construction and a positional static prompt
  remain supported; new integrations should prefer a `PromptComposer`.
- Custom `ShortTermMemory` implementations must provide the complete v0.2
  protocol: `messages`, `replace()`, and `maybe_compact()` in addition to
  `add()` and `render()`. Transactional cancellation snapshots `messages` and
  restores through `replace()`.
- Wheels contain the runtime but not the source repository's writable example
  profiles. An installed CLI therefore requires an explicit external
  `--profile`; source checkouts retain the four examples under `profiles/`.
- LingChat 0.1 declares `lingcore<0.2.0` and cannot be installed alongside this
  release. Its dependency bound and cancellation integration need a coordinated
  companion release before pairing it with LingCore 0.2.
- PDF extraction remains optional through `lingcore[pdf]` because PyMuPDF is
  not part of the Apache-2.0 base dependency set.

## [0.1.0] - 2026-06-08

- Initial tagged preview of the config-driven async agent runtime.

[0.2.0]: https://github.com/lllluolingyu/LingCore/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lllluolingyu/LingCore/releases/tag/v0.1.0
