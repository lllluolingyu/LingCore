# Changelog

Notable user-facing changes to LingCore are documented here. The project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Provider-reported token usage: `lingcore/usage.py` (`TokenUsage`/`UsageMeter`),
  `stream_options.include_usage` on every streamed request (opt out with
  `llm.stream_usage: false`), and a `UsageReported` agent event covering the
  main model, the compaction/memory summarizer, and the vision fallback.
  `Agent.drain_usage()` returns usage billed before a Stop landed.

- A structured, read-only `git` builtin for coding profiles, covering status,
  diff, log, show, and branch inspection without shell approval while keeping
  repository-changing and networked operations behind `run_shell`.
- `lingcore profile list/init`, an explicit clean wheel-template manifest, and
  atomic initialization into writable per-user application state.
- Profile-selected guardrails through dotted Python targets or the
  `lingcore.guardrails` entry-point group, including constructor options.
- Ruff lint/format and package-wide mypy gates, plus a LingChat-main
  compatibility job on every LingCore change.
- First-party Telegram Bot API channel in the main package, installable with
  `lingcore[telegram]`, with polling and single-process webhook runners.
- Allowlisted private-chat routing, per-user workspaces/memory/session stores,
  durable active-session selection, streaming reconciliation, bounded
  attachment downloads, rate-limited delivery, confirmation callbacks, Stop,
  and graceful shutdown.
- PTB-free Telegram configuration diagnostics through
  `lingcore doctor --telegram-config`.
- Bundled `codex` and `claude-code` collaboration skills: named, resumable
  conversations with an external Codex or Claude Code CLI, read-only
  consultation by default, confirmation-gated implementation mode, bounded
  supervised execution, hardened non-interactive sandbox/configuration
  boundaries, and `lingcore doctor` validation of their options and executables.
- `skill_gated_tools:` — the exclusion form of `initial_tools:` for hiding a
  few ceiling tools until a skill grants them.
- `@tool(high_risk=True)` lets a tool (builtin, skill-shipped, or third-party)
  declare that skill activation must confirm before granting it, alongside the
  name-based builtin baseline.

### Changed

- Rebuilt the `search` builtin with scoped literal/regex matching, recursive
  filename lookup, configurable pruning and hard scan/time/result budgets,
  worker-thread execution, confined no-follow regular-file reads, context
  rendering, deterministic coverage reports, and oversized-output offloading.
  Search no longer follows symlinked files or directories, including links
  whose targets remain inside the workspace; search the target's real path.
- `WindowMemory` reuses tiktoken encodings already loaded in the process and
  otherwise uses a deterministic UTF-8 byte-ratio estimate. Rendering never
  downloads vocabulary data, including when the disk cache is missing.
- Release wheels include immutable profile templates and a `py.typed` marker;
  writable workspaces, memory, and sessions are created only in initialized
  external/user-state profiles.
- The CLI dispatches Telegram synchronously before `asyncio.run`, allowing PTB
  to own its event loop and signal handlers.
- `Agent.turn_pending_finalization` exposes the cancellation readiness check as
  a public read-only lifecycle predicate.

### Fixed

- Regex searches interrupt expensive individual matches at the scan deadline,
  retain earlier matches, and report partial coverage. Matching uses the timed
  `regex` VERSION0 engine; Unicode case folding can differ from stdlib `re`.
- Concurrent knowledge index updates are serialized across tasks and processes
  so partial updates cannot silently overwrite each other.
- Skill modules missing a declared tool roll back all their registrations and
  their module-cache entry, allowing a corrected module to be loaded again.
- Token counting treats special-token-like text as ordinary text instead of
  raising an error while rendering retained history.
- Telegram delivery failures no longer abort and roll back an otherwise valid
  Agent turn; terminal replies use a best-effort plain-message fallback.
- Invalid active-session selections self-heal to a fresh session instead of
  permanently blocking messages and `/new`.

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
