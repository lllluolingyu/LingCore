# Workflows and superagents

The coding profile includes `delegate`, `workflow_run`, `workflow_status`,
`workflow_result`, `workflow_stop`, and `workflow_resume`. All workers use ordinary
Agents with inherited guardrails, sandbox settings, profile environment, enabled
hooks, and shared ancestor budgets. Role definitions only narrow authority.

## Run and inspect work

```sh
lingcore -p profiles/coding --workspace /path/to/project
# In the interactive frontend: /workflows lists definitions and recent runs.
lingcore run start review-changes -p profiles/coding --args '{"focus":"concurrency"}'
lingcore run list -p profiles/coding
lingcore run show r-... -p profiles/coding
lingcore run resume r-... -p profiles/coding
lingcore run continue w-... -p profiles/coding
lingcore run continue w-... -p profiles/coding --prompt 'Inspect the remaining case'
lingcore run prune -p profiles/coding
lingcore --list-sessions --all -p profiles/coding
```

For direct CLI workflow runs, set the profile's workspace or `LINGCORE_WORKSPACE`
first. `run start` accepts `--session` to attach an existing parent. Resume and
continuation attach the recorded parent and respect its session lease.

The bundled `review-changes` workflow admits at most 40 workers: three reviewers,
up to four findings each, three independent skeptics per finding, and one writer.
Reviews and verification form a pipeline. The writer receives only findings that
survive the vote. Failed and dropped results are counted in the run result.

Roles live in `roles/*.yaml`, with an optional matching Markdown persona. The
coding profile ships `reviewer`, `skeptic`, and `writer`; `general` is always
available. Roles select tools, workspace, model alias, static skills, context
layers and tighter limits. Model aliases are complete LLM configurations under
`delegation.models`; roles cannot specify endpoints or remove inherited policy.
The effective tool set is the intersection of the role, the delegating batch's
current authorization, and any requested narrowing. Skills cannot enlarge it.

`delegate` accepts `prompt`, optional `role`, `label`, `output` (a bounded JSON
schema or configured schema name), `tools`, `workspace`, and `background`.
`delegate(continue_worker="w-...", prompt="...")` resumes the retained worker session under the intersection of
its recorded and current authority. Workers finish through `submit_result`;
blocked results require one concrete question. Workers can list relative artifact
paths in `submit_result.artifacts`; the host retains and hashes bounded copies.

## Isolation, limits and recovery

- `shared`: parent workspace; write-capable workers serialize.
- `frozen`: one read-only input capture per run. Network and write tools are refused.
- `snapshot`: independent Git clone with no hardlinks or alternates, including
  bounded parent dirty/untracked state. The retained patch contains only worker
  edits. Host-owned metadata outside the worker directory builds the patch; worker
  Git configuration is never executed by the host. The builtin `git` tool also
  uses a bounded metadata copy without repository configuration or helpers.
  Tracked symlinks are skipped and reported at capture. Symlinks created by a worker
  explicitly fail patch finalization. Unborn Git repositories are supported.
  Confined snapshot Git operations require Linux procfs; unsupported platforms
  fail closed. Non-Git workspaces use a confined
  copy. Patches are never auto-applied.
- `scratch`: an empty retained worker directory.
- `ticket`: host-only persistent superagent workspace.

Defaults are depth 1, eight direct delegates per turn, 50 workers per run,
four concurrent workers, 2,000 model requests, 3,600 seconds, and a stopping
threshold of one million reported tokens. Provider usage arrives after requests,
so in-flight requests can overshoot the token threshold. `usage_missing: stop`
stops further admission when usage is unavailable. Own-agent context and worker
spend are displayed separately. Larger or high-risk runs ask for confirmation.
The risk floor includes core mutation tools, scoped tool metadata, and configured
skill risk declarations, even for a single worker.
Worker confirmations are labelled and serialized; they never offer a new
session-wide allow rule. Background confirmation requires a frontend that opts in.
Lark, Telegram and LingChat support idle confirmations and run completion delivery;
the CLI declines worker confirmations while a run is in the background.

Session schema 4 stores runs, node attempts, worker lineage and usage. Stop/Edit
cancels affected work and detaches its run without deleting it. Forks do not copy
runs; an inherited run reference is readable but cannot be resumed from the fork.
Worker sessions are hidden from the ordinary session list. Startup marks orphaned
work interrupted. Retention defaults to 30 days and 500 runs; live and interrupted
runs are protected. Shared continuation artifacts remain until their referencing
runs can be removed.
Pruning is scoped to the recorded workspace. Legacy records without a recorded
workspace are retained rather than deleting journals whose artifacts cannot be located.

Workflow resume requires confirmation. Node keys include local semantics,
rendered prompts, dependency **values**, role/model/schema/context and frozen
input revision. Pure/frozen results with intact artifacts can be reused; live
readers rerun. A started effecting node never executes again in the same run:
unchanged completed results are recorded, while changed/incomplete nodes block
their consumers. Independent branches can proceed. Prior usage and admission
counts remain charged. Altered frozen input requires a new run.
Declining resume leaves the prior status intact. The confirmation shows previous
live readers, cache candidates, and a conservative upper bound on live work;
dependency changes are resolved during execution. Results include
`parent_workspace_changed` (`null` when the comparison is unavailable). Frozen
input verification runs off the event loop and is shared by all workers in a run.
Cache context follows worker persona, configured memory, static skills and plugin
prompts. Plugins without hooks do not globally disable reuse; hooks with undeclared
dependencies remain conservatively non-reusable. Artifact retention and verification
stream bounded chunks off the event loop.

Declarative workflows support bounded `agent`, `map`, `parallel`, `loop`, `keep`,
`transform`, `log`, and one level of named child workflows. `${path}` performs
lookup only. Static validation checks worker bounds, forward references, nesting,
result schemas and model aliases before launching anything. See the complete
[review example](../profiles/coding/workflows/review-changes.yaml) and
[grammar/design](design/workflows-and-superagent.md).

Plugins may declare a `workflows` directory containing YAML or trusted Python
workflows. Python modules execute only for explicitly enabled plugins. Import
`workflow` and `WorkflowAPI` from `lingcore.workflows`; use `agent`, `parallel`,
`pipeline`, `phase`, `log`, and `workflow`. Runtime admission still enforces the
caps. Python resume restarts orchestration with prior spend charged and is refused
once any effecting worker has started.

## Run a team service

```sh
lingcore profile init team --destination ./my-team
# Fill my-team/.env, requester/approver allowlists and channel settings.
lingcore doctor -p ./my-team
lingcore superagent start -p ./my-team
lingcore superagent status -p ./my-team
```

The team profile is an operator template: configure an actual model key, Lark
credentials, requester/approver IDs, and the ticket workspace setup before use.
It requires a fail-closed Bubblewrap or OCI shell sandbox. Host shell execution
is refused at startup and by doctor. The optional `executor.prepare` command runs
inside that sandbox before the executor; an interrupted/failed preparation is
retained for inspection and is not blindly repeated.

The pure domain engine is the only writer of tickets, immutable mandates,
assignments, approvals and effect decisions. SQLite commits decisions and their
outbox effects together. Delivery retries use backoff and preserve per-ticket
order for each recipient. Unreachable approvers do not block delivery to other
approvers. Idle checks do not rewrite state; changed commands update only changed
projection rows. Status exposes repeated failures. A service lock prevents two routers from
owning the same store. Restart ends interrupted assignments, waits for worker
session leases, and resumes their retained conversation with a recovery note.
Router turn rollback cannot cancel ticket runs. Follow-ups reach the executor at
its next turn boundary, or queue a follow-up assignment if it has just finished.
Progress reports never replace a final result. The default router has no file tools.
The router session ID survives restarts, including recovery of older ticket-run
records. Terminal or superseded launch effects settle without retries; intentional
stops remain distinct from host interruption. Final reports produce one delivery.

Service retention defaults to 30 days, 1,000 closed tickets, and 10,000 history
entries (`retention_days`, `max_closed_tickets`, and `max_history` under
`superagent`). Active work, pending effects and unanswered questions remain
protected. Rejected message bodies are not stored; event IDs use a bounded
deduplication window. Completed effects, evaluations and unreferenced triggers
are pruned along with their database projections.

Sources normalize transport-authenticated actor IDs and external event IDs.
Allowlisted actors, bot exclusion, and `off`/`observe`/`auto` policy are checked
before routing. `observe` admits addressed requests. `auto` can admit unaddressed
messages only through configured rules, using the observer role or a matching
rule's role. Rule changes invalidate inferred mandates.

The executor's requested tools must fit the profile. Tools outside the router's
dispatch snapshot require an exact per-ticket approval from a configured human
approver. A chat message saying “approved” cannot grant them. Approval cards bind
a nonce, source conversation, tool set, and mandate digest. Pending approvals hold
no executor slot; denial and expiry end blocked. Three ordinary slots and one
human-follow-up slot are the defaults. Background processes retain their slot for
up to 1,800 seconds, then hand back a concrete question and are reaped.
Cards are sent directly to `policy.approvers`, with a distinct nonce for each
recipient. `policy.approval_source` can select an enabled Lark or Telegram source
for approvals from another source. Telegram approvers must have opened a chat with
the bot. Shell policy approval covers the sandboxed tool's own confirmation;
plugin requests for additional consent remain denied in unattended execution.
Host approval APIs can narrow a requested extra-tool set; subsequent execution stays
within that selection and current role/policy limits. A successfully delivered card
without a conversation receipt is retained for panel action and never resent merely
to obtain the receipt. Its unbound transport buttons cannot grant authority.

## Additional sources and the panel

Configure sources under `superagent.sources`. Each has `enabled`, `allowlist`,
`bots`, optional `config`, `interval_seconds`, and source-specific `options`.

- **Lark:** reuses `lark.yaml` and the existing SDK transport. The actor is an
  app-scoped open ID. Group admission still follows the channel's chat allowlist;
  the service applies its own addressed/automatic routing gate.
- **Telegram:** reuses `telegram.yaml` credentials with polling transport. Actor
  IDs are numeric strings. Inline-button approvals use the authenticated callback
  sender. The ordinary per-user Telegram bridge remains a separate mode.
- **Schedule:** interval ticks are checkpointed. Options include `name`, `prompt`,
  `requester`, `channel`, and `chat_id`. The configured requester must be allowlisted.
- **GitHub:** explicitly enable `plugins: [github]`. Set `options.repository` to
  `owner/repo`, `bot_login`, and optionally `token_env` (default `GITHUB_TOKEN`).
  The first poll establishes a cursor without routing historical comments.
  Issue/PR comments are then polled with durable, paginated recovery;
  oversized backlogs fail without advancing the cursor. Replies use effect markers
  for retry deduplication. Configure `policy.approval_source` to deliver requests
  to approvers through Lark or Telegram; the panel also supports approvals. The adapter uses
  the documented [GitHub issue-comment API](https://docs.github.com/en/rest/issues/comments).

Run LingChat against the same team profile to see “Needs a person”. Its existing
per-launch authentication token protects the panel. `superagent.panel_actor`
binds that authenticated operator to a configured approver; request bodies cannot
choose an actor. The panel shows pending tools and original questions, supports
exact approval/denial and replies, and displays repeated delivery failures.
A panel transaction validates against the service's latest published authority,
including environment binding, rather than the ticket's old authority snapshot.
Launch always revalidates current authority again.
A server started with authentication disabled cannot use these endpoints.

Tests use scripted models, SDK event doubles, and mocked HTTP responses. Real
provider credentials, channel credentials and a functioning sandbox are needed
for deployment; the test suite does not send messages to live services.
