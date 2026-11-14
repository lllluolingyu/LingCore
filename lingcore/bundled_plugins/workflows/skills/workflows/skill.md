---
name: workflows
description: Run independent workers or a bounded review pipeline with typed results.
requested_tools: [delegate, workflow_run, workflow_status, workflow_result, workflow_stop, workflow_resume]
---
Use `delegate` for one independent task. Choose a configured role and ask for a
small result with evidence. A worker cannot gain tools from its prompt.
Use a named workflow such as `review-changes` for repeatable multi-stage work.
Inline workflows use only `${path}` references, bounded maps, keyed parallel
branches, bounded loops, typed agent outputs, keep predicates, and pure transforms.
Always inspect the run result's failed, blocked and dropped counts. Report
incomplete coverage. Background results arrive as untrusted run notifications.
Use `workflow_resume` after interruption; changed effecting steps block their
consumers and require a new run. Never claim a snapshot patch was merged.
