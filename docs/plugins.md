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

Canvas, Codex, Claude Code, Browser, Process, Search, Clock, and Calendar are
bundled first-party plugins. Their declared skills remain in the catalog, and
their code also loads
when the profile statically names one of their skills or authorizes a provided
tool. Existing teaching and
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
| `turn_rolled_back(event)` | **Synchronous**, observe only. Called after a turn's conversation state has been voided: Stop (`reason="cancelled"`), a turn that raised (`"failed"`) or an event stream closed unfinished (`"abandoned"`), even when the durable truncate failed and is pending. Not called for a committed turn, Stop during `turn_end`, or a blocked user message. `event.started_at` is the turn's `time.monotonic()` start, so a plugin can undo side effects it recorded at or after it. Never block: schedule async cleanup as a task the plugin owns and awaits (e.g. in `aclose`). Errors are logged by type and never change finalization; no notice is raised. |
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
contains `content`, optional `error`, and `turn_index`. `TurnRolledBackEvent`
contains `reason` and `started_at`.

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
        browser_resize, browser_close]
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
attach an image. Tool calls on one browser are serialized. `browser_snapshot`
accepts an optional `ref` (only that element's subtree, e.g. one list or form)
and `depth` (only that many levels below the root), which keeps long pages
cheap to read; refs in a partial snapshot stay valid for the action tools.
`browser_resize` changes the active tab's viewport during a session (within the
same 320-3840 x 240-2160 bounds as `viewport_width`/`viewport_height`, which set
the startup size), so one session can check mobile and desktop breakpoints. It
changes only the size, not touch or user-agent emulation, and new tabs open at
the configured size. An attached browser refuses it, since its tabs belong to
the owner.

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

### Attaching to a running browser

`tool_options.browser.cdp_url` makes the session attach to a Chromium that is
already running with a DevTools port (`--remote-debugging-port`) instead of
launching one — for example a browser a test harness prepared and logged in,
or your own Chrome:

```yaml
tool_options:
  browser:
    cdp_url: http://127.0.0.1:9222   # loopback http(s)/ws(s) endpoint only
```

The session adopts the browser's default context and its open tabs (the newest
is active), preserving its cookies, logins, init scripts, media preferences and
download settings. Closing the session (`browser_close`, `Agent.aclose`) only
disconnects: the browser, its context
and its tabs stay open. Tabs created by the session and popups opened by its
tool actions are closed; independently opened owner tabs stay open. The
endpoint must be on loopback and free of credentials, and errors never echo
it, since a `ws://` endpoint's path grants full control of the browser.
`headless`, `executable_path`, `chromium_sandbox` and the viewport size
configure a browser LingCore launches, so combining them with `cdp_url` is an
error.

The network policy still applies, but with weaker guarantees, because the
vetting proxy can only be set when Chromium is launched. An attached session
enforces HTTP requests in Playwright routing and guards new page WebSocket
connections in both loaded and future documents, including frames. Each URL is
validated and its host resolved and refused when non-public (approvals work as
above). Disconnecting restores the original WebSocket constructor, and sockets
remain usable without reloading. It cannot pin the vetted address, so DNS can
change between the check and the
connection; it does not see redirect hops; and it cannot block service workers
or downloads in a context it did not create. Existing connections, cached
WebSocket constructors and worker-created sockets are outside these guards.
Attach only to a browser you trust, and set `allow_private_hosts` or
`allowed_networks` under
`tool_options.fetch_url` when the target app runs locally.

Browser actions can submit forms on the public web. Gate the interaction tools
behind the `browser` skill with `skill_gated_tools`. To make activation ask
first, add them to `tool_options.activate_skill.high_risk_tools`; that list
replaces the builtin baseline, so keep `run_shell`, `write_file`, `patch_file`
and `edit_file` in it.

## Bundled search plugin

The `search` plugin exposes `search_web` for everyday research, including local
services, travel planning, opening hours, and current information. It uses the
existing HTTP dependency and needs no additional SDK or browser. The provider
is selected by the profile; the model cannot switch providers or endpoints.

```yaml
plugins: [search]
tools: [search_web, fetch_url]
tool_options:
  search_web:
    provider: searxng              # searxng (default), exa, or deepseek
    # searxng_url: https://your-instance.example  # or SEARXNG_URL in .env
    # api_key_env: MY_SEARCH_KEY   # optional override for an API provider
    max_bytes: 1048576            # response cap; 1024–2097152 bytes
    timeout: 20                   # per-request network timeout; 1–60 seconds
```

`search_web(query, max_results=5)` accepts a nonblank query of up to 128 Unicode
characters and returns up to 10 deduplicated results with titles, direct HTTP(S)
URLs, snippets, and publication dates when supplied. Include relevant places,
dates, and search operators in the query. Results are untrusted source data;
use `fetch_url` to verify important details and cite their source URLs. A search
never fetches the returned URLs itself.

| Provider | Configuration | Request |
| --- | --- | --- |
| SearXNG | `SEARXNG_URL` or `searxng_url`; no API key | `GET <instance>/search?format=json&q=…` |
| Exa | `EXA_API_KEY` (or `api_key_env`) | `POST https://api.exa.ai/search`, with highlights |
| DeepSeek | `DEEPSEEK_API_KEY` (or `api_key_env`) | `POST https://api.deepseek.com/v1/tools/search`, with summaries and no full page content |

SearXNG requires an instance with `json` enabled in `search.formats`; many public
instances disable it and return HTTP 403. The plugin does not pick an arbitrary
public instance. Configure one you operate or trust. See the official
[SearXNG Search API](https://docs.searxng.org/dev/search_api.html),
[Exa Search API](https://exa.ai/docs/reference/search), and
[DeepSeek Search API](https://api-docs.deepseek.com/guides/search/) contracts.

The daily profile enables this plugin and defaults to SearXNG. Its 64k-token,
200-message working set uses automatic compaction and allows up to 50 tool
iterations per turn for search and source verification. Copy its
`.env.example` to `.env` and set `SEARXNG_URL`. To use an API provider, set
`LINGCORE_SEARCH_PROVIDER=exa` with `EXA_API_KEY`, or
`LINGCORE_SEARCH_PROVIDER=deepseek` to reuse `DEEPSEEK_API_KEY`. Keys resolve
through the selected profile's environment and never appear in tool output.
The manifest declares all three environment variables as optional because only
the selected provider needs its settings; missing settings are reported when
search runs. There is no automatic provider fallback.

Requests share `tool_options.fetch_url`'s network policy, including vetted-IP
pinning, fake-IP exemptions, and approval before private addresses. A local
SearXNG instance therefore asks before every query unless explicitly allowed by
that policy; an approval covers only that one request. For a trusted instance,
configure a narrow address exemption, for example:

```yaml
tool_options:
  search_web:
    provider: searxng
    searxng_url: http://192.168.1.20:8080
  fetch_url:
    allowed_networks: [192.168.1.20/32]
```

Use the instance's exact address (`/32` for IPv4, `/128` for IPv6). This
exemption also applies to `fetch_url` and the browser at that address. For a
loopback instance, use an IP literal such as `http://127.0.0.1:8080` with
`127.0.0.1/32`; `localhost` remains a named-local-host approval.

Redirects are refused so credentials and POST bodies stay at their
intended endpoint. Network errors, invalid JSON, and oversized responses fail
explicitly; they are not reported as empty search results.

## Bundled clock plugin

Models have no clock, so without help they guess "today" from their training
data. The `clock` plugin adds one line to every user message:

```text
[clock] Now: Monday 2026-10-19 22:32 Asia/Shanghai (UTC+08:00)
```

```yaml
plugins: [clock]                 # a hooks-only plugin: needs this entry
tool_options:
  clock:
    timezone: ${LINGCORE_TIMEZONE:-}   # IANA name; blank uses the host's zone
```

- The line is `user_message` hook context, not a system-prompt layer, so the
  cached system prefix stays byte-stable. It is stored with the message
  (`input_text` keeps only what the user typed), which also lets the model see
  when each earlier message was sent.
- Without a configured zone the host's is detected from `TZ`, the
  `/etc/localtime` link or `/etc/timezone`. A host whose zone has no IANA name
  falls back to its current UTC offset, shown without a name and unable to
  follow daylight-saving changes. On Windows, install the `tzdata` package to
  use IANA names.
- Options are validated when the Agent is built: an unknown zone or option key
  fails at launch. `on_hook_error: ignore` keeps a clock failure from blocking
  a message.

## Bundled calendar plugin

`calendar` keeps the profile's calendar: one standard iCalendar (`.ics`,
RFC 5545) file that every session of the profile shares, so the agent can
check it for upcoming and long-term arrangements before it plans or schedules
anything. It works offline: nothing is sent to an online calendar, and the
user can import the file into any calendar app (Apple Calendar, Google
Calendar, Outlook, Thunderbird). Authorizing its tools engages it; it has no
hooks.

```yaml
tools: [calendar_add, calendar_update, calendar_list, calendar_remove]
tool_options:
  calendar:
    path: calendar.ics           # relative to the profile directory, like memory.md
    # allow_absolute_path: true  # required for an absolute path
    # timezone: Europe/Berlin    # default: tool_options.clock.timezone, then the host's
    default_duration_minutes: 60
    list_days: 90                # calendar_list's default window
```

| Tool | Arguments | Result |
| --- | --- | --- |
| `calendar_add` | `events` (1-20) | each event as saved with its id, saved events it overlaps, the timezone |
| `calendar_update` | `id`, `changes`, optional `scope`, `occurrence` | the updated event or occurrence with the same id, plus overlaps and incomplete-coverage warnings |
| `calendar_list` | optional `start`, `end` (dates, inclusive), `query` | events overlapping the window, sorted, with their ids |
| `calendar_remove` | `ids` (1-20) | the removed events; a repeating event goes as a whole series |

Each event has a `title` and a `start`: `YYYY-MM-DD` for an all-day event, or
a local `YYYY-MM-DDTHH:MM` in the calendar's timezone (an explicit offset is
converted). It may also have
- `end` (for an all-day event, the last day, inclusive) or `duration_minutes`,
- `location` and `description`,
- up to three `reminders` (minutes before the start),
- `repeat`: `frequency` (daily, weekly, monthly or yearly), `interval`, and
  `count` or `until`.

A time skipped by a daylight-saving change, an end before the start, and a mix
of all-day and timed values are refused instead of being guessed.

`calendar_update` takes only the fields to change. For example, moving a
45-minute appointment keeps its duration, notes, location and alarms:

```json
{"id": "3f9a1c2b7d4e@lingcore", "changes": {"start": "2026-10-22T10:00"}}
```

An explicit `end` or `duration_minutes` replaces the length. `location`,
`description` and `repeat` may be set to `null` to clear them, and
`reminders: []` clears alarms; omitted fields retain their saved values.
Imported properties and nested components are kept unless explicitly replaced.
Updates retain `UID`, increment `SEQUENCE` and record `LAST-MODIFIED`.

For a repeating event, `scope` is required: `series` updates the whole series;
`occurrence` also needs that instance's **original** start. An individual move
uses a `RECURRENCE-ID` override; listings show its original start so later edits
can select the same instance without creating duplicates:

```json
{"id": "3f9a1c2b7d4e@lingcore", "scope": "occurrence", "occurrence": "2026-10-27T09:00", "changes": {"start": "2026-10-28T11:00"}}
```

Simple series support individual occurrence updates, retaining the series'
all-day or timed form. For unsupported rules,
range overrides, or whole-series timing/recurrence changes when exceptions
already exist, the tool refuses the change and keeps the file intact; use an
individual update where supported or edit the calendar in another app. Metadata
updates such as title and location can still apply to the whole series.

- **One file, profile-scoped.** The path resolves against the profile
  directory and is confined like the memory file: a relative path may not
  escape it, an absolute path needs `allow_absolute_path: true`, and a path
  inside the installed package is refused. A symlinked calendar file is
  refused. The Telegram channel gives each allowed user their own
  `calendar.ics` beside their memory file.
- **Safe changes.** Each add, update or remove rereads the whole file under a
  persistent advisory lock (`calendar.ics.lock` beside it) and publishes it
  with an atomic replace, so parallel tool calls and concurrent sessions (a CLI
  and a browser tab on one profile) never lose each other's events. A file that
  is not a single UTF-8 `VCALENDAR` is left unchanged and the tool says so; the
  file is capped at 2 MiB.
- **Hand edits survive.** Components the plugin did not write (a `VTODO`, an
  event added in another app and saved back) and extra calendar properties
  such as `X-WR-CALNAME` are kept as they are.
- **Daylight saving.** Timed events normally carry their IANA `TZID`; an
  explicitly chosen second occurrence of a repeated clock time is saved in
  UTC to preserve its instant. For a recurring event, `EXDATE`/`RDATE` replace
  that first occurrence while the rule keeps its local time. Comparisons and
  timed event lengths use elapsed time, while imported `DURATION` values keep
  their nominal calendar days or weeks. Displayed ambiguous times include
  their UTC offsets. The file's
  `VTIMEZONE` for that zone is regenerated on every change to list the zone's
  offset changes over all of its events (each series at most ten years ahead),
  so a weekly 09:00 meeting stays at 09:00 after the clocks change, even in
  clients that do not know the zone name. `UNTIL` is written in UTC as the RFC
  requires. Without an IANA zone, times are saved as UTC instants and the
  result says so.
- **Listing.** `calendar_list` defaults to `list_days` from today; with a
  `query` (matched case-insensitively against titles, places and notes) and no
  `end` it searches ten years ahead. Repeats using only `FREQ`, `INTERVAL`,
  `COUNT` and `UNTIL` are expanded, including date or date-time `RDATE` additions
  and `EXDATE` exclusions. Invalid dates and nonexistent local times are skipped
  without consuming `COUNT`, as RFC 5545 requires. Individual overrides replace
  their original occurrences without hiding the rest of a simple series.
  Other rules, period-valued `RDATE`s and series with range overrides are listed
  separately, even when their original start is outside the requested range. The result warns
  that coverage is incomplete and unlisted times cannot be assumed free;
  `calendar_add` and `calendar_update` give the same warning when checking for
  overlaps. Cancelled events are omitted. At most 100 entries are shown. Titles and places are
  untrusted file content, and the result says so.
- **Ids.** Events the plugin adds get a short `UID` such as
  `3f9a1c2b7d4e@lingcore`; listings show every event's `UID` as its id, which is
  what `calendar_remove` takes.

The daily profile enables both plugins. Its `/event` command turns a
plain-language description into a `calendar_add` call, whose result reports any
overlap, and `/plan` checks `calendar_list` before planning. `/today` explicitly
reads the clock's local date as a one-day range and combines appointments with
relevant saved preferences and commitments. It distinguishes elapsed from
upcoming appointments, flags conflicts or incomplete coverage, and keeps any
suggested priorities separate from scheduled events. Producing an agenda does
not create events or enable background reminder delivery.

## Bundled process plugin

`process` runs long-lived commands (dev servers such as `npm run dev`, `vite`
or `next dev`, and watchers such as `tsc --watch`) in the background, in the
`run_shell` sandbox, and lets the agent's browser reach a declared server port.
`run_shell` cannot do this by construction: it kills and reaps the whole
process tree when a command returns, and with `network: false` a server inside
the sandbox listens in a private network namespace the host-side browser
cannot enter. The tools take the plugin's name as their prefix, so it is
enabled as `process`:

```yaml
plugins: [process]
skills: [process]               # always-on guidance for the tools
tools: [process_start, process_logs, process_stop, process_list]
tool_options:
  run_shell:
    sandbox: {backend: bubblewrap}   # required: Bubblewrap or OCI
  process:
    max_processes: 3
    max_lifetime: 3600          # seconds before a process is stopped
    allow_patterns: []          # auto-approved starts; separate from run_shell's
```

### Tools

| Tool | Arguments | Result |
| --- | --- | --- |
| `process_start` | `command`, optional `name`, `port`, `ready_pattern` (regex), `ready_timeout` (s) | id, runner, state (`ready`/`running`/`exited`), the browser URL for `port`, the first log lines |
| `process_logs` | `id`, optional `since` (line cursor), `tail`, `grep` | log lines plus the next cursor; large output is offloaded like `run_shell`'s |
| `process_stop` | `id` | exit status and the last log lines; removes the entry |
| `process_list` | none | id, name, command, state, uptime, time left, exit code, port |

- `process_start` is `@tool(high_risk=True)`. It asks for confirmation unless
  its own `allow_patterns` match (same exact-or-prefix rules as `run_shell`).
  The list is separate on purpose: a pattern approving `npm run build` for
  `run_shell` never approves a long-lived server.
- `ready_pattern` makes the call wait until a log line matches (for example
  `Local:\s+http`) or `ready_timeout` passes, so the model does not poll.
  Without it the call still waits briefly to catch an immediate failure. A
  process that exits early is a normal result with its exit code and output.
- Ids carry a random suffix (`p-3f9a1c`) and are never reused, so an id from a
  resumed or forked transcript cannot reach a new process; an unknown id is an
  error explaining that processes do not survive an Agent restart.
- An exited process stays listed with its log until `process_stop` removes it.
  At most `max_exited` are kept; the oldest are dropped with a note.

Options live under `tool_options.process` and are validated when a tool runs,
like the browser's, so a typo is reported in the first tool result:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_processes` | 3 | concurrently running processes (1-16) |
| `max_exited` | 5 | exited entries kept for a final read |
| `max_log_bytes` | 1 MiB | ring buffer per process |
| `max_line_chars` | 2000 | longest stored line |
| `ready_timeout` | 60 s | default wait, clamped to `max_ready_timeout` (300 s) |
| `max_lifetime` | 3600 s | a process is stopped when it reaches this age |
| `require_confirmation` | `true` | ask before every start not matched by `allow_patterns` |
| `allow_patterns` | `[]` | auto-approved `process_start` commands |
| `offload_over_chars` | 8000 | larger `process_logs` output is staged to a workspace file |
| `sandbox` | `run_shell`'s | a separate sandbox configuration for processes |

`max_lifetime` is on by default because Bubblewrap has no CPU, memory or PID
budget: `run_shell`'s timeout was the only bound on a command, and a
background process would otherwise have none.

### Logs and environment

Combined stdout and stderr are drained continuously (a chatty server never
blocks on a full pipe) into a byte-bounded ring buffer with a monotonic line
counter; a read from before the retained range says how many lines were
dropped. Lines are normalized before they are stored or matched: ANSI escape
sequences are stripped, a carriage return starts a new line (progress bars
redraw one line many times), and a line longer than `max_line_chars` is cut
with a marker, so `ready_pattern` and `grep` see what a person would read.
Processes start with `NO_COLOR=1` and `FORCE_COLOR=0` to reduce escapes at the
source, and `PYTHONUNBUFFERED=1` so a Python server's ready line is not held
in a pipe buffer. Like every runner, they get `/dev/null` as stdin; an
inherited stdin is the CLI's terminal, which a dev server's keyboard shortcuts
would otherwise switch to raw mode.

### Lifecycle

A configured Bubblewrap or OCI sandbox is required; the legacy host runner is
refused because a host process group can outlive a crashed LingCore. Bubblewrap
runs the command in its own PID namespace with `--die-with-parent`, so a crash
kills the whole tree, including processes that detach themselves; OCI's
watchdog removes the container.

| Event | Behavior |
| --- | --- |
| `process_stop` | kill and reap the tree, return the last log lines, remove the entry |
| Process exits by itself | state becomes `exited`; the log stays until `process_stop` |
| `max_lifetime` reached | killed like `process_stop`; the entry stays as `exited` with a note |
| Stop of the turn that started it | killed, because the `process_start` result is voided from the conversation and the model would lose track of a live process |
| Stop of a later turn | untouched: earlier starts are committed history |
| `Agent.aclose` | every process is killed in reverse start order by an owned task that resumes after cancellation |
| LingCore crash | Bubblewrap's `--die-with-parent` or the OCI watchdog |

The turn-scoped kill uses the synchronous `turn_rolled_back` hook: the plugin
kills exactly the processes it started at or after the event's `started_at`.
Only one turn runs per Agent and processes start only inside its tool calls,
so that set is the voided turn's starts, whichever plugin hooks did or did not
run before the rollback. The doomed processes leave the registry at once and
the next `process_*` call waits for their reaping, so a killed process never
appears as running. A `process_stop` in a turn that is then stopped has the
mirror problem (the process is gone, the conversation forgets the stop);
re-launching is not safe, so `process_list` simply shows the truth.

Every Agent replacement closes the old Agent and so stops all its processes:
CLI `/new`, `/resume` and exit, LingChat session switches, tab close and
**Edit** (which rebuilds the Agent, so it also stops processes from turns the
edit keeps), and Telegram runtime eviction. Handing a live registry to a
rebuilt Agent would need a cross-Agent ownership model. Processes are not
persisted: logs reach `sessions.db` only through tool results the model read,
and a resumed session's old ids fail with an explanation.

### Port bridge

With `port`, the sandbox's network must be off (`network: false`): either
Bubblewrap, or a Linux OCI container (Docker or Podman) on a Linux host. That
private network namespace guarantees `127.0.0.1:<port>` inside it is the
sandbox's own server and nothing else. `port` is refused under `network: true`,
where the bridge could expose whatever host service holds that loopback port and
bypass the browser's private-host approval. It is also refused for Windows
containers and on non-Linux hosts, where the runtime runs containers in a VM
that cannot receive a host socket. Without `port` the process uses the
sandbox's configured network; under `network: true` its server may be on host
loopback, and `browser_navigate` asks before opening it, as for any local host.

LingCore never connects to a path the sandbox controls:

1. The command runs under `relay.py`, a stdlib-only script shipped with the
   plugin and mounted read-only. The relay is the command's parent and exits
   with its status, so killing the sandbox kills both. It never lets the
   command inherit its control connection. Its interpreter is
   - Bubblewrap: the sandbox's `/usr/bin/python3`, which is the host's,
     through the read-only `/usr` mount;
   - OCI: `python3` from the image's `PATH`. Use an image that has one
     (`node:22` does, `node:22-slim` does not); without it the process exits
     with code 127 and the result says why.

   Either way it needs Python 3.9 or later.
2. The relay gets a control connection back to LingCore:
   - Bubblewrap: LingCore creates a Unix socketpair and passes one end into
     the sandbox as an inherited descriptor.
   - OCI: a container cannot inherit a descriptor. LingCore listens on a Unix
     socket in a fresh `0700` host directory and bind-mounts **only that
     socket file**, read-only, at `/run/lingcore/bridge.sock`. The relay
     connects to it *before* it starts the command (and exits without running
     the command if it cannot). LingCore accepts that first connection, then
     closes the listener and removes the directory, so the command can never
     connect in the relay's place.
3. For each browser connection the plugin sends the relay one end of a fresh
   socketpair over the control connection (`SCM_RIGHTS`). The relay connects it
   to `127.0.0.1:<port>`, then `::1` (Node binds `localhost` to either),
   answers one status byte before any data so a server that is still starting
   gives a clean "connection refused", and copies bytes both ways.

The obvious alternative is LingCore *connecting* to a socket the relay creates
in a mounted directory. That directory would be writable from inside the
sandbox, which could replace the socket with a symlink to a host socket
(Docker's, an ssh-agent's) that the host's `connect` would follow. In the OCI
design the host only listens and accepts, and the container sees one socket
file, not a directory it could write to. A compromised server can kill or
replace the relay, which runs as the same user in the same sandbox, but gains
nothing: the relay only ever carries the server's own traffic to the browser.

The OCI bridge was verified with Docker 29.9.1 (`python:3.13-slim`, run as the
host user). Podman uses the same code path but has not been run end to end.

The browser plugin of the same Agent routes the origin the server prints,
`http://localhost:<port>/` (also `127.0.0.1` and `[::1]`), through the bridge
instead of TCP. HMR WebSockets, absolute redirects, CORS settings, OAuth
callbacks and dev-server host checks are all configured for that origin. It
needs no private-host approval, because the user confirmed the command serving
it and the bridge reaches nothing else; every other local host the page
requests is vetted as before. The agent's browser is per-Agent and ephemeral,
so this does not affect the user's own browser. A listed process keeps
shadowing its port after it exits (connections are refused), so a stopped dev
server never silently falls through to a host service on the same port. If
the log announces a different `localhost` port than the declared one (Vite
moves to the next free port unless given `--strictPort`), `process_start` says
so. An attached (`cdp_url`) browser has no proxy and never uses a bridge. This
was verified with Vite 6.3.5 in the frontend profile's sandbox, including an
HMR reload over the bridged WebSocket.

### Frontends and limits

- The CLI, LingChat and agentgui frame a `process_start` confirmation as a
  command ("start background process?") but never offer "always allow", which
  would write `run_shell`'s allowlist.
- Telegram refuses a profile that lists `process_start`: it runs unattended
  per-user runtimes in one process, and evicting one would silently kill a
  user's server.
- Not provided: supervision or restart-on-crash, surviving an Agent
  replacement or a LingCore restart, and networking between two background
  processes. Each has its own private namespace, so a frontend and its API
  server must be one command (for example a `dev` script using
  `concurrently`).

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

MCP components are postponed with no target release; wrap an external service
in a plugin with declared tools instead. Subprocess hooks, before-tool argument
rewriting, plugin versions in session metadata, hot reload and a plugin
marketplace are also outside v0.4.0.
