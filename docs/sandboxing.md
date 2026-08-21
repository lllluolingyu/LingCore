# Sandboxed shell runner

`run_shell` has three runners. A profile chooses exactly one:

- no `sandbox` block: the legacy host shell, retained for compatibility;
- `backend: bubblewrap`: a Linux user-namespace sandbox;
- `backend: oci`: a fresh Docker or Podman container per command.

The selection is fail-closed. Once a sandbox is configured, a missing binary,
unsupported host, unavailable namespace/runtime, missing mount or environment
variable, image failure, or container setup failure returns a tool error. It
never retries the command on the host. Every result includes `runner: ...`.

Confirmation remains independent. Sandboxing limits what approved code can do;
confirmation controls whether that code may run at all.

## Bubblewrap (Linux)

The bundled `coding` and `coding_ollama` profiles use this configuration:

```yaml
tool_options:
  run_shell:
    timeout: 60
    require_confirmation: true
    allow_patterns: []
    sandbox:
      backend: bubblewrap
      executable: /usr/bin/bwrap
      network: false
      tmpfs_mb: 512
      path: /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
      pass_env: []
      read_only_mounts: []
```

The runner starts from an empty mount namespace. It exposes `/usr` read-only
and mirrors the host's actual `/bin`, `/sbin`, `/lib`, and `/lib64` directory or
symlink layout instead of assuming a particular usrmerge scheme. It also
provides a minimal set of account/name-resolution files, private `/proc` and
`/dev`, a bounded `/tmp`, and the workspace at `/workspace` read-write. It
creates user, PID, IPC, UTS, and cgroup namespaces, drops all capabilities,
disables nested user namespaces, and removes networking when `network: false`.
The sandbox inherits neither the host home directory nor the host environment.

Tools installed outside `/usr` must be explicitly mounted. The mount target is
visible read-only:

```yaml
read_only_mounts:
  - source: /opt/toolchains/rust
    target: /opt/toolchains/rust
```

Bubblewrap is Linux-only. On macOS or Windows, select an OCI backend instead.

## Docker or Podman

An OCI profile must explicitly select its runtime, executable, image, pull
policy, container OS, non-root identity, and resource budget:

```yaml
tool_options:
  run_shell:
    timeout: 60
    require_confirmation: true
    allow_patterns: []
    sandbox:
      backend: oci
      runtime: docker                 # docker or podman
      executable: /usr/bin/docker
      image: python:3.13-slim
      pull: missing                   # missing, always, or never
      pull_timeout: 300
      container_os: linux
      command_prefix: [/bin/sh, -c]
      workspace_target: /workspace
      user: "1000:1000"
      network: false
      pass_env: []
      read_only_mounts: []
      resources:
        cpus: 2
        memory_mb: 2048
        pids: 256
        ephemeral_storage_mb: 512
```

The runner checks the image OS and rejects images that declare Dockerfile
`VOLUME` paths, pulling according to policy, then performs
`create -> attached start -> exit-code inspect -> forced remove with volumes`.
Containers receive a unique name and LingCore label. A separate watchdog starts
before `create`, removes partially created or running containers after
cancellation or process death, and also cleans up staged environment data.

Linux containers have a read-only root filesystem, a size-limited `/tmp`, the
workspace as their only writable host bind, no added capabilities,
`no-new-privileges`, the runtime's default seccomp policy, an explicit non-root
user, and CPU/memory/PID/tmpfs limits. Numeric spellings of UID zero and the
name `root` are rejected; when a group is supplied, root and numeric GID zero
are rejected too. `network: false` selects the runtime's `none` network.

Docker and Podman run Linux containers on Linux. Docker Desktop or a Podman
machine supplies the trusted Linux VM on macOS and Windows.

### Native Windows containers

Native Windows images use a distinct contract:

```yaml
sandbox:
  backend: oci
  runtime: docker
  executable: C:\Program Files\Docker\Docker\resources\bin\docker.exe
  image: mcr.microsoft.com/windows/nanoserver:ltsc2025
  pull: missing
  container_os: windows
  command_prefix: [cmd.exe, /S, /C]
  workspace_target: C:\workspace
  user: ContainerUser
  network: false
  pass_env: []
  read_only_mounts: []
  resources:
    cpus: 2
    memory_mb: 2048
    ephemeral_storage_mb: 512
```

Only Docker is accepted for this mode. LingCore forces Hyper-V isolation and a
non-administrator `ContainerUser`, plus CPU, memory, and writable-layer storage
limits. A numeric PID limit is intentionally rejected: Hyper-V gives the
container its own kernel, so guest PIDs do not consume the host PID namespace.
The Windows image must be compatible with the host and Docker engine mode.

## Environment and mounts

Sandboxes do not inherit arbitrary credentials. Each name in `pass_env` is
resolved from the selected profile's `.env` first and the process environment
second. A missing name stops the command before backend launch. Values are not
stored in profile YAML:

```yaml
pass_env: [CARGO_REGISTRIES_PRIVATE_TOKEN]
```

Passed values are injected only into the guest command; they are never added to
the environment used to start Bubblewrap or invoke the OCI runtime. Bubblewrap
carries values under randomized inert names and restores the requested names
inside the sandbox. OCI uses a mode-`0600` temporary env file, starts its cleanup
watchdog before container creation, and deletes the file immediately after the
create attempt. OCI env-file values may not contain NUL, carriage-return, or
newline characters.

Read-only mount sources and backend executables must be absolute host paths.
Mount targets must be absolute paths for the guest OS. The workspace bind is
always read-write because builds need to produce artifacts; additional
read-write host mounts are not supported.

## Diagnostics

Run this before enabling a profile in a frontend:

```bash
lingcore doctor --profile profiles/coding
```

Doctor validates the typed configuration, executable and mount paths, required
environment names, and local backend capabilities. Bubblewrap gets a bounded
namespace probe. OCI diagnostics contact the local engine and inspect image
presence, OS metadata, and declared volumes but never pull an image. Doctor
refuses to execute a backend binary resolved inside the writable workspace. A
missing image is an error only under `pull: never`; otherwise doctor reports
that execution will pull it.

## Threat model and limits

The target is untrusted build scripts and dependencies. LingCore trusts the
host kernel, Bubblewrap or the selected container runtime, and (for OCI) the
configured image and VM boundary. This is process/container isolation, not a
VM-grade defense against kernel or runtime exploits.

The workspace is intentionally writable and is not quota-limited; keep it under
version control and do not place secrets there. Temporary storage is bounded,
but workspace output can consume host disk. The runner does not currently offer
fine-grained egress allowlists, GPU/device passthrough, an interactive TTY, or
per-mount writable quotas. Add explicit read-only mounts and passed environment
variables sparingly: each expands what untrusted code can observe.
