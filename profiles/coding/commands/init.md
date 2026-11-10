---
description: Inspect the runtime environment and save an approved snapshot.
argument_hint: "[optional focus, e.g. Python, GPU, or frontend tools]"
---
Inspect the current coding environment and save a private runtime snapshot.
Optional focus: $ARGUMENTS

1. Call `world` with `action="read"` to review the existing environment notes.
   Use `list_dir` and `read_file` to identify the project manifests, lockfiles,
   and local environments. Treat old observations as potentially stale.
2. Use `run_shell` for a small number of bounded, read-only inspection batches.
   Inspect the environment in which this tool actually runs, including its
   sandbox/container restrictions. Keep the configured confirmation policy.
   Start with portable probes and adapt to the observed OS and available tools:
   - System: OS/distribution, kernel, architecture, working directory, shell,
     and runtime/container details that can be verified (`uname`,
     `/etc/os-release`, `pwd`). Do not dump environment variables.
   - Resources: available CPU count, memory available/total, and workspace disk
     free/total (`getconf`, `free`, `df`, or platform equivalents). Check visible
     cgroup limits where applicable; label host totals separately from effective
     container limits. Check GPU availability and memory only when a suitable
     local utility exists. Do not infer usable GPU access from installed drivers.
   - Tools and packages: locate common tools and check their versions (Git,
     Python/pip/uv, Node/npm/pnpm, compilers, Go, Rust, Java, and project-relevant
     tools). Inspect installed packages in the relevant interpreter or local
     environment, for example `python -m pip list --format=json` or
     `npm ls --depth=0`. Prefer project-local `.venv` when present and runnable.
     Use local package metadata or the installed OS package manager as needed.
     Distinguish declared dependencies from verified installed packages, and
     distinguish system packages from each project's environment.
   Bound command time and output. Summarize relevant package names/versions
   instead of pasting an entire inventory. Missing tools, denied probes,
   timeouts, or unavailable metrics are limitations to report, not reasons to
   install software. Do not install/update packages, execute project setup or
   lifecycle scripts, access the network, read credentials, or print secrets.
3. Once useful observations have been collected, call `world` with
   `action="update"` and a concise Markdown snapshot covering system, resources,
   toolchains/packages, project setup, and limitations. Record which environment
   each observation describes. The tool adds the timestamp and workspace and
   replaces only the managed environment section in private runtime state
   (default `.lingcore/world.md`), preserving authored `world.md` unchanged.
   Saving always asks the user to approve the exact text, even if the tool is
   enabled by default. Never bypass that confirmation. Without a confirmation
   handler, or after denial, explain that the snapshot was not saved. If another
   session updates the snapshot during approval, read it again and request a new
   approval instead of overwriting the other session's update.
   Save verified observations only; do not carry stale facts forward or invent
   values. If inspection is entirely blocked, leave the previous snapshot intact
   and explain the blocker. Use the `world` tool even when the profile lives
   outside the workspace; do not try to write a workspace-local `world.md`.
4. Confirm the runtime file was updated only after the tool succeeds. Briefly report
   the important capabilities and any missing tools or unverified resources.

Re-running `/init` refreshes the snapshot. The updated context is available on
the next model request in this session and when the profile is reopened.
