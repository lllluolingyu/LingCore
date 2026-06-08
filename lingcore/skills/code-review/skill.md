---
name: code-review
description: "Perform a structured code review of one or more files."
requested_tools:
  - read_file
  - search
  - run_shell
---

When performing a code review:

1. Read each file fully before commenting — never review from a diff alone.
2. Group findings by severity: blocker / warning / suggestion.
3. For each finding, cite the exact `file:line` and explain the impact, not just
   the rule.
4. Check tests exist and pass for the changed behaviour; run them if you can.
5. End with a short overall assessment and a clear approve / request-changes call.
