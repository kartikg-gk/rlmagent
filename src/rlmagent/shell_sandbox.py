"""Build a shell with a private filesystem and an explicit workspace mount."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from rlmness.environment import execution_environment

_SUPERVISE = """
import json, os, signal, subprocess, sys, threading
child = subprocess.Popen(json.loads(sys.argv[1]), stdin=subprocess.DEVNULL, start_new_session=True)
def stop():
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
def watch():
    while os.read(0, 4096): pass
    stop()
threading.Thread(target=watch, daemon=True).start()
try: code = child.wait()
finally: stop()
raise SystemExit(code if code >= 0 else 128-code)
"""


class SandboxUnavailable(OSError):
    pass


@lru_cache(maxsize=1)
def _runner():
    if os.name == "nt":
        wsl = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/wsl.exe"
        prefix = [str(wsl), "--exec"]
        probe = prefix + ["/usr/bin/bwrap", "--version"]
    elif sys.platform.startswith("linux"):
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise SandboxUnavailable("Install Bubblewrap for workspace-isolated shell execution")
        prefix, probe = [], [bwrap, "--version"]
    else:
        raise SandboxUnavailable("Workspace shell isolation requires Linux/Bubblewrap or WSL")
    try:
        answer = subprocess.run(probe, capture_output=True, timeout=30,
                                env=execution_environment())
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SandboxUnavailable("Workspace shell isolation is unavailable; configure Bubblewrap "
                                 "in Linux/WSL or explicitly grant --shell-access host") from error
    if answer.returncode:
        raise SandboxUnavailable("Bubblewrap is unavailable in the selected Linux/WSL environment; "
                                 "install it or explicitly grant --shell-access host")
    return prefix, probe[len(prefix)]


def plan(cwd, command, *, argv=None, env=None, network=False):
    """Return a supervised argv. Keep its stdin pipe open for its entire lifetime."""
    prefix, bwrap = _runner()
    workspace = str(Path(cwd).resolve())
    if prefix:
        mapped = subprocess.run(prefix + ["/usr/bin/wslpath", "-a", "-u", workspace],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
            env=execution_environment())
        if mapped.returncode or not mapped.stdout.strip().startswith("/"):
            raise SandboxUnavailable("WSL could not resolve the workspace mount")
        workspace = mapped.stdout.strip()
    options = [bwrap, "--die-with-parent", "--new-session", "--unshare-user",
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--cap-drop", "ALL",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/home",
        "--bind", workspace, "/workspace", "--chdir", "/workspace", "--clearenv"]
    if not network:
        options += ["--unshare-net"]
    else:
        for name in ("resolv.conf", "hosts", "nsswitch.conf"):
            options += ["--ro-bind-try", "/etc/" + name, "/etc/" + name]
        options += ["--ro-bind-try", "/etc/ssl/certs", "/etc/ssl/certs"]
    variables = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/home",
        "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **(env or {})}
    for key, value in variables.items():
        if value is not None:
            options += ["--setenv", str(key), str(value)]
    options += ["--", *(argv or ["/bin/bash", "--noprofile", "--norc", "-c", command])]
    return [*prefix, "/usr/bin/python3", "-I", "-c", _SUPERVISE, json.dumps(options)]
