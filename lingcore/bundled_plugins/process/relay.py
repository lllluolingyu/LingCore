"""Port-bridge relay that runs *inside* a Bubblewrap sandbox (stdlib only).

Started by ``process_start`` for a declared ``port`` as

    python3 -I -S relay.py <control-fd> <port> <command>

It runs ``/bin/sh -c <command>`` as its child and exits with the child's
status, so the process's lifetime is the command's. The control descriptor is
one end of a Unix socketpair whose other end LingCore holds; it is never
inherited by the command. For every browser connection LingCore sends one end
of a fresh socketpair over it (SCM_RIGHTS). The relay connects that socket to
``127.0.0.1:<port>`` (then ``::1``, since Node binds ``localhost`` to either)
in the sandbox's private network namespace, replies with one status byte
(0 connected, 1 refused) and then copies bytes both ways.

No filesystem path is involved, so nothing inside the sandbox can redirect
LingCore to another socket. The relay can only ever carry the command's own
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


def main() -> int:
    control_fd, port, command = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    control = socket.socket(fileno=control_fd)
    control.set_inheritable(False)
    child = subprocess.Popen(["/bin/sh", "-c", command], close_fds=True)
    threading.Thread(target=_accept, args=(control, port), daemon=True).start()
    code = child.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
