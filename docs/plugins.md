# Plugins (API v1)

A plugin packages skills, tools, lifecycle hooks, prompt templates and static
prompt context behind one manifest. Discovery reads data; enabling a plugin is
explicit consent to execute its Python. Plugins run in the Agent process and
are trusted Python code, not an OS sandbox. The profile's `tools:` list remains
the explicit permission ceiling for every contributed tool.

## Quick start

```bash
uv run lingcore plugin new hello -p profiles/coding
uv run lingcore plugin list -p profiles/coding
uv run lingcore plugin info hello -p profiles/coding
```

Merge the printed entries into the profile:

```yaml
plugins: [hello]
tools: [read_file, activate_skill, hello_echo]
```

Then run `lingcore doctor -p profiles/coding` and start the profile. Try
`/hello:hello world`, or `/hello world` when that bare name is unambiguous.
CLI help and completion, Telegram's command menu and both LingChat interfaces
expose the enabled command catalog. `--no-hooks` and `--no-commands` omit those
scaffold components. Creation refuses an existing directory or a destination
inside LingCore's installed package tree and publishes the complete skeleton
by a temporary-directory rename.

## Layout and manifest

```text
audit/
  plugin.yaml
  plugin.py
  skills/review/skill.md
  commands/review.md
  prompt.md
  .env.example
```

```yaml
name: audit
version: 0.1.0
api: 1
min_lingcore: "0.4.0"
description: Review changes with an audit policy.
module: plugin.py
provides: [audit_log]
hooks: AuditHooks
skills: skills
commands: commands
prompt: prompt.md
options_key: audit
on_hook_error: block
hook_timeout: 10
environment:
  - {name: AUDIT_URL, required: false}
  - {option: token_env, default: AUDIT_TOKEN, required: true}
requires:
  executables: [{name: gh, option: executable}]
  options: [{key: base_url, hint: "set AUDIT_URL in the profile .env"}]
  modules: [{name: httpx, hint: "pip install httpx"}]
```

Unknown manifest fields are errors. Names match
`[a-z0-9][a-z0-9-]{0,39}`; the tool prefix replaces `-` with `_`. Every tool
registered by the plugin's module **or any of its skill modules** must be
named exactly the prefix or begin with `<prefix>_`. `provides` must describe
the module's actual registrations. Modules cannot overwrite existing tools or
register undeclared ones; a failed module load rolls back its registry changes
and synthetic import entry.

Versions and `min_lingcore` use numeric `major.minor.patch` comparisons. Only
plugin API major 1 is supported. Component paths are relative to the plugin
and may not escape it. `skills` and `commands` default to their namesake
directories when present; `module`, `hooks` and `prompt` are optional.
`options_key` defaults to the prefix. Prompt files are capped at 16,000
characters. Hook timeouts are positive seconds, default 10, maximum 60.

Environment declarations name variables; their values stay in the profile's
`.env` or exported environment. An `option` entry reads the variable **name**
from `tool_options[options_key]`, falling back to `default`. Doctor checks
required variables and `.env.example` coverage without printing values,
importing plugin code or launching a plugin executable. Executable declarations
use `PATH` or a configured executable path; required options must be nonempty.
Module declarations name top-level Python packages; doctor locates them with
`importlib.util.find_spec` without importing them. Doctor also warns when a
plugin with hooks is engaged only through its tools or skills, since its hooks
(and any tool that depends on its per-Agent instance) need `plugins:` consent.

## Discovery and consent

Discovery uses this precedence, with later sources shadowing earlier ones:

1. `lingcore/bundled_plugins/*/plugin.yaml`.
2. Installed `lingcore.plugins` entry points.
3. `<profile>/plugins/*/plugin.yaml`.

`plugin list`, `plugin info` and doctor report shadowing. A malformed or
duplicate installed entry point, or an invalid profile-local plugin, is skipped
and reported as a doctor warning and a `plugin list` row; it becomes an error
only for a profile that enables or engages that name, so one bad install cannot
break unrelated profiles. Installing a package does not enable it:
third-party code loads only for a plugin named in `plugins:`. Plugin names in
that list must be valid and unique. Hooks, commands and prompt layers always
require this explicit entry, even for bundled plugins.

Canvas, Codex, Claude Code and Browser are bundled first-party plugins. Their skills
remain in the catalog, and their code also loads when the profile statically
names one of their skills or authorizes a provided tool. Existing teaching and
coding profiles therefore keep working without edits. Skill catalog order is
the instruction-only core skills, plugin skills, then the profile's
`activate_skill.skills_dir` (default `skills`). Two visible plugins cannot
provide the same skill name, and a plugin skill cannot reuse a core skill name.
A profile-local skill can shadow a plugin skill: the shadowed skill, and any
tool the local skill provides, no longer auto-engage the bundled plugin's code,
so the local module can register those tool names itself.

Enabling a plugin registers its tools but does not authorize them. List every
reachable tool in `tools:`; wildcards are unavailable. `initial_tools` and
`skill_gated_tools` retain their usual behavior. `@tool(high_risk=True)` stays
a floor for the skill activation confirmation gate. Hooks can add policy or
context and cannot override authorization, validation or a tool's own approval.

## Stable Python hook API

Import the following types from `lingcore.plugins`:
`PluginHooks`, `PluginContext`, `UserMessageEvent`, `UserMessageDecision`,
`ToolCallEvent`, `ToolDecision`, `ToolResultView`, `ToolResultPatch`,
`TurnEndEvent` and `AttachmentView`.

```python
from __future__ import annotations

from lingcore.plugins import (
    PluginHooks,
    ToolCallEvent,
    ToolDecision,
    ToolResultPatch,
    ToolResultView,
    UserMessageDecision,
    UserMessageEvent,
)


class AuditHooks(PluginHooks):
    async def user_message(self, event: UserMessageEvent) -> UserMessageDecision | None:
        return UserMessageDecision.add_context("Apply the team's audit policy.")

    async def before_tool(self, event: ToolCallEvent) -> ToolDecision | None:
        if event.name == "run_shell":
            return ToolDecision.ask("Approve this shell call under the audit policy?")
        return None

    async def after_tool(
        self, event: ToolCallEvent, result: ToolResultView
    ) -> ToolResultPatch | None:
        return ToolResultPatch.append_note("Reviewed by the audit policy.")
```

Override only what you need. The runner calls overridden methods in `plugins:`
order and applies the manifest timeout separately to each hook call.

| Method | Boundary and permitted effect |
| --- | --- |
| `start()` | Lazy, before the first turn's guardrail. Failure refuses the turn without committing input; unsuccessful starts retry next turn. Successful starts are remembered. |
| `user_message(event)` | After the guardrail and attachment ingest, before the user message is stored. Return `block(reason)` or `add_context(text)`. Added context is bounded to 16,000 characters across the chain and excluded from `input_text`. |
| `before_tool(event)` | After authorization and argument validation. Return `allow()`, `deny(reason)` or `ask(prompt)`. Denial stops the before-hook chain; ask uses the frontend's confirmation handler and denies without one or on decline. The hook timeout bounds the hook call, not the human's answer, which the frontend's own confirmation timeout governs. Allow continues the chain and preserves the tool's own confirmation. |
| `after_tool(event, result)` | For valid, authorized calls, on success and error, including plugin denials. Return `ToolResultPatch(content=..., note=...)`, `replace(content)` or `append_note(note)`. Replacement content and each note are capped at 16,000 characters before commit; a note is appended after the tool's full output and never truncates it. Identity, success, name and attachments cannot change. |
| `turn_end(event)` | Observe only, after the outcome is committed and before Final or a model/iteration error. It does not run when Stop lands earlier. Stop during `turn_end` rolls nothing back: `finalize_cancelled_turn()` returns the committed `Final` or `Error` instead of `TurnCancelled`. |
| `aclose()` | Once per instance, in reverse order, including failed starts. Ordinary errors are reduced to exception types and swallowed; cancellation propagates and another close can finish remaining instances. |

`PluginContext` is frozen: `name`, `root`, `workspace`, `profile_dir`,
`session_id`, a recursively read-only copy of the plugin's options, and scoped
`getenv(name, default=None)`. The Agent's `ToolContext.getenv` supplies environment
values, so profile `.env` values (including explicit empty ones) take precedence.
Avoid resolving secrets from global `os.environ`.

Input events and result views are frozen snapshots. `UserMessageEvent` contains
`text`, `input_text`, and an immutable attachment tuple. `ToolCallEvent` contains
`call_id`, `name`, and recursively frozen `arguments`. `ToolResultView` contains
`call_id`, `name`, `content`, `ok`, and immutable attachments. `TurnEndEvent`
contains `content`, optional `error`, and `turn_index`.

One hook instance belongs to one Agent. Plugin tools can access it through
`ctx.plugins["audit"]` to share, for example, an HTTP client. Parallel tool
batches may call `before_tool` and `after_tool` concurrently on that instance;
plugins must protect shared mutable state themselves. Authorization is still
snapshotted once for the whole batch.

For hook exceptions or timeouts, `on_hook_error: block` blocks user input,
denies tool calls, or withholds tool content. A turn-end failure produces only
a notice. `ignore` skips the failing hook with a notice. Startup failure always
refuses the turn, since the instance is not ready. Failed confirmations deny
even in ignore mode. Cancellation is never treated as a hook error.
`PluginNotice(plugin, hook, action, message)` is transient and rendered by every
frontend; actions are `denied`, `blocked`, `asked`, `modified` and `failed`.

Embedding applications should `await agent.aclose()` or use `async with agent`.
Stop an active turn using cancel → await → `finalize_cancelled_turn()` before
closing. Before finalization, capture `agent.drain_plugin_notices()` and render
those notices with drained usage before the returned cancellation event.
Rollback discards undrained notices so an abandoned stream cannot announce them
on a later turn. CLI switches/exits, Telegram runtime replacement/shutdown and LingChat
disconnect/rebuild perform this lifecycle automatically.

## Bundled browser plugin

`browser` drives a headless Chromium through Playwright, for pages that need
JavaScript, interaction or a visual check (`fetch_url` remains the cheap path
for plain documents). Install the `lingcore[browser]` extra and run
`playwright install chromium` once. Doctor reports a missing Playwright package;
a missing Chromium build surfaces as the first tool call's error.

```yaml
plugins: [browser]
tools: [browser_navigate, browser_snapshot, browser_click, browser_type,
        browser_select, browser_press, browser_back, browser_screenshot,
        browser_close]
tool_options:
  browser:
    timeout: 15                 # seconds per browser action (1-120)
    headless: true
  fetch_url:                    # the network policy for fetch_url *and* the browser
    allowed_networks: [198.18.0.0/15, "2001:2::/48"]  # e.g. a fake-IP proxy
```

The tools need the `plugins:` entry: the per-Agent hook instance owns the
browser, launches it on the first tool call and closes it in `Agent.aclose`.
Each Agent (CLI session, Telegram user, LingChat tab) gets its own browser
with an ephemeral context, so no cookies or logins persist and none are shared.
Pages are returned as Playwright AI-mode accessibility snapshots whose
`[ref=eN]` handles address elements for `browser_click`, `browser_type` and
`browser_select`; each action returns a fresh snapshot, large ones offloaded
like `fetch_url` output (`offload_over_chars`, default 20,000). Screenshots
attach an image. Tool calls on one browser are serialized.

The network policy is `fetch_url`'s (invariant 9) and is configured in one
place: `allow_private_hosts`, `confirm_private_hosts` and `allowed_networks`
under `tool_options.fetch_url` govern both tools, even in a profile that does
not authorize `fetch_url` itself. Putting them under `tool_options.browser` is
an error naming the shared location. Every request URL is validated
like `fetch_url`'s (scheme, embedded credentials, port) before any approval
applies. Chromium connects only through a per-session loopback SOCKS5 proxy,
which resolves each host itself, refuses a local host or any non-public
answer, and pins the connection to the vetted address. Because every
connection passes the proxy, the check covers subresources, WebSockets and
each redirect hop, which Playwright's routing never reports. Only
`browser_navigate` can ask the user about such a host, and approval covers that
hostname for the rest of the browser session (`browser_close` forgets it).
Blocked requests are listed in the next result. Non-http(s) schemes, downloads
and service workers are refused, and JavaScript dialogs are dismissed and
reported.

Because every connection must pass that proxy, the browser does not use an
upstream proxy: it ignores `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` and the
system or PAC proxy settings, whereas `fetch_url` honours the standard proxy
environment variables. Where outbound traffic is only allowed through an HTTP
proxy, the browser therefore cannot reach the web even though `fetch_url` can.
A TUN-mode proxy (Clash/mihomo, Surge) still works, because it captures
connections at the network layer; add its fake-IP ranges to
`tool_options.fetch_url.allowed_networks` as for `fetch_url`.
Chromium's own sandbox is off by default as in Playwright
(`chromium_sandbox: true` enables it where the host supports it), and
`executable_path` selects another Chromium build.

Browser actions can submit forms on the public web. Gate the interaction tools
behind the `browser` skill with `skill_gated_tools`. To make activation ask
first, add them to `tool_options.activate_skill.high_risk_tools`; that list
replaces the builtin baseline, so keep `run_shell`, `write_file`, `patch_file`
and `edit_file` in it.

## Prompt-template commands

```markdown
---
description: Review a path using the audit policy.
argument_hint: "[path]"
---
Review $ARGUMENTS and report the audit findings.
```

This `commands/review.md` expands `/audit:review src/` into the body with
`$ARGUMENTS` replaced literally. `/review` works only when its combined bare and
Telegram alias candidates are unambiguous and the frontend has not reserved it.
Telegram can use `/audit_review` (hyphens become underscores); legal,
unambiguous aliases appear in its command menu. Frontend commands such as
`/new`, `/help` and `/stop` retain precedence. The qualified plugin form remains
available for a command whose bare name is reserved. Command names are
lowercase, and matching is case-insensitive (`/Review` resolves `review`).
Frontmatter accepts LF or CRLF line endings.

Profile-level `<profile>/commands/*.md` provides unqualified commands without
Python code. The expanded `UserInput.text` passes through guardrails and hooks.
`display_text` holds the original input and passes through the guardrail's
`pre_input` separately; that screened command is the stored `Message.input_text`
that resume and Edit show, and regeneration resolves it again.
Attachments are preserved. Commands never enter the system prompt, keeping
its cached prefix stable.

## Distributing a pip plugin

Put the manifest and components inside one top-level Python package, for
example `lingcore_audit/`, and include them as wheel data. Declare an entry point:

```toml
[project.entry-points."lingcore.plugins"]
audit = "lingcore_audit"

[tool.hatch.build.targets.wheel]
packages = ["lingcore_audit"]
```

The entry-point name must equal the manifest name. Its value is a single
top-level package name, with no dots or `:attribute`. Discovery uses
`find_spec(...).submodule_search_locations` without importing the package;
the manifest's Python module is loaded only after engagement. Keep package
initializers free of registration side effects. Ship local relative imports
only if your module layout supports the file loader; absolute imports from
your package work after the plugin is enabled.

MCP components are deferred to v0.4.x. Subprocess hooks, before-tool argument
rewriting, plugin versions in session metadata, hot reload and a plugin
marketplace are also outside v0.4.0.
