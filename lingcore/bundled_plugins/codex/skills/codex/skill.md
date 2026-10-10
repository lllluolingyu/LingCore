---
name: codex
description: Communicate with an external Codex CLI agent when the user asks to consult,
  cross-check with, or delegate coding work to Codex.
requested_tools:
- codex_agent
---

Use `codex_agent` to start or continue a conversation with an external Codex
CLI agent that shares the current workspace.

- Use `mode: consult` for analysis, review, planning, debugging advice, or a
  second opinion. It is read-only.
- Use `mode: implement` only when the user explicitly asks the external agent
  to make changes. It requires fresh confirmation and grants Codex write access
  only to the workspace.
- Every invocation pins Codex's human approval reviewer so inherited
  auto-review settings cannot approve a sandbox escape, including on resumed
  threads.
- Reuse the same `conversation` name for follow-up questions. Names are scoped
  to the current LingCore session and workspace, and survive resuming that
  LingCore session (a run without a persisted session shares one workspace-wide
  namespace instead). Use `restart: true` to replace a name with a fresh Codex
  thread; the old Codex transcript is left intact.
- On the first turn, give Codex a self-contained brief: objective, relevant
  scope, constraints, known evidence, and the exact result you want back. On
  later turns, include only the new information or question needed to continue.
  Do not include secrets unless the user explicitly authorizes sharing them and
  they are necessary.
- Do not ask Codex to invoke LingCore or another external agent; avoid recursive
  delegation.
- Treat the reply as collaborator input. Inspect any claimed edits and run
  appropriate validation before relying on them or reporting completion.
- If the CLI is missing or unauthenticated, state that setup problem plainly;
  do not substitute a different agent without the user's direction.
