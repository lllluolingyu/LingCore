---
name: claude-code
description: Communicate with an external Claude Code CLI agent when the user asks
  to consult, cross-check with, or delegate coding work to Claude Code.
requested_tools:
- claude_code_agent
---

Use `claude_code_agent` to start or continue a conversation with an external
Claude Code agent that shares the current workspace.

- Use `mode: consult` for analysis, review, planning, debugging advice, or a
  second opinion. It runs in Claude Code's isolated, restricted planning mode
  and cannot modify the workspace.
- Use `mode: implement` only when the user explicitly asks the external agent
  to make changes. It requires fresh confirmation and allows Claude Code's
  built-in file tools inside the workspace. Shell/code tools, WebFetch, local
  hooks, plugins, skills, MCP servers, and user/project permission overrides are
  disabled rather than inherited. Claude Code may therefore report partial
  work when a task needs a command; relay that honestly.
- Reuse the same `conversation` name for follow-up questions. Names are scoped
  to the current LingCore session and workspace, and survive resuming that
  LingCore session (a run without a persisted session shares one workspace-wide
  namespace instead). Use `restart: true` to replace a name with a fresh Claude
  Code session; the old Claude transcript is left intact.
- On the first turn, give Claude Code a self-contained brief: objective,
  relevant scope, constraints, known evidence, and the exact result you want
  back. On later turns, include only the new information or question needed to
  continue. Do not include secrets unless the user explicitly authorizes
  sharing them and they are necessary.
- Do not ask Claude Code to invoke LingCore or another external agent; avoid
  recursive delegation.
- Treat the reply as collaborator input. Inspect any claimed edits and run
  appropriate validation before relying on them or reporting completion.
- If the CLI is missing or unauthenticated, state that setup problem plainly;
  do not substitute a different agent without the user's direction.
