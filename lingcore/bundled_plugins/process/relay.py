"""Port-bridge relay that runs *inside* the sandbox (stdlib only).

Started by ``process_start`` for a declared ``port`` as

    python3 -I -S relay.py <control> <port> <command>

It runs ``/bin/sh -c <command>`` as its child and exits with the child's
status, so the process's lifetime is the command's. The control connection is
a Unix stream socket whose other end LingCore holds; it is never inherited by
the command. ``<control>`` is either

- a descriptor number (Bubblewrap): one end of a socketpair LingCore passed
  into the sandbox, or
- an absolute path (OCI): a socket LingCore listens on, bind-mounted into the
  container as a single file. The relay connects *before* starting the
  command, and LingCore accepts exactly one connection and then closes the
  listener, so the command can never take the relay's place. Failing to
  connect exits without running the command.

For every browser connection LingCore sends one end of a fresh socketpair over
the control connection (SCM_RIGHTS). The relay connects that socket to
``127.0.0.1:<port>`` (then ``::1``, since Node binds ``localhost`` to either)
in the sandbox's private network namespace, replies with one status byte
(0 connected, 1 refused) and then copies bytes both ways.

LingCore never connects to a path inside the sandbox, so nothing there can
redirect it to another socket. The relay can only ever carry the command's own
server to the browser. It needs Python 3.9 or later (``socket.recv_fds``).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading

_CONNECT_TIMEOUT = 2.0
_CHUNK = 64 * 1024


def _pump(source: socket.socket, sink: socket.socket) -> None:
    try:
        while True:
            data = source.recv(_CHUNK)
            if not data:
                break
            sink.sendall(data)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _connect(port: int) -> socket.socket | None:
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(_CONNECT_TIMEOUT)
        try:
            sock.connect((host, port))
        except OSError:
            sock.close()
            continue
        sock.settimeout(None)
        return sock
    return None


def _serve(client: socket.socket, port: int) -> None:
    upstream = _connect(port)
    try:
        if upstream is None:
            client.sendall(b"\x01")
            return
        client.sendall(b"\x00")
        back = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
        back.start()
        _pump(client, upstream)
        back.join()
    except OSError:
        pass
    finally:
        client.close()
        if upstream is not None:
            upstream.close()


def _accept(control: socket.socket, port: int) -> None:
    while True:
        try:
            _, fds, _, _ = socket.recv_fds(control, 1, 1)
        except OSError:
            return
        if not fds:
            return  # LingCore closed the control socket
        for fd in fds[1:]:
            os.close(fd)
        client = socket.socket(fileno=fds[0])
        threading.Thread(target=_serve, args=(client, port), daemon=True).start()


def _control(spec: str) -> socket.socket:
    if spec.isdigit():
        control = socket.socket(fileno=int(spec))
    else:
        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control.settimeout(_CONNECT_TIMEOUT)
        control.connect(spec)
        control.settimeout(None)
    control.set_inheritable(False)
    return control


def main() -> int:
    spec, port, command = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    try:
        control = _control(spec)
    except OSError as exc:
        print(
            f"lingcore relay: cannot reach LingCore's port bridge ({exc}); "
            "the command was not started",
            file=sys.stderr,
            flush=True,
        )
        return 1
    child = subprocess.Popen(["/bin/sh", "-c", command], close_fds=True)
    threading.Thread(target=_accept, args=(control, port), daemon=True).start()
    code = child.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
