"""OS-backed single-writer ownership for a saved session."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


class SessionBusy(RuntimeError):
    pass


def _record_holder(path: Path) -> None:
    """Note who holds the lease, beside the lock file, so a refusal can name them."""
    command = " ".join(os.path.basename(part) if i == 0 else part
                       for i, part in enumerate(sys.argv[:3]))
    try:
        path.write_text(json.dumps({"pid": os.getpid(), "since": time.time(),
                                    "command": command[:200]}), encoding="utf-8")
    except OSError:
        pass


def describe_holder(path: Path) -> str:
    """`` It is held by process 1234 (rlm-agent worker run), since 14:02.`` or empty."""
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        pid = int(info["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    since = time.strftime("%H:%M", time.localtime(float(info.get("since") or 0)))
    command = str(info.get("command") or "").strip()
    return (f" It is held by process {pid}" + (f" ({command})" if command else "")
            + f", since {since}. Close that window or stop that worker to open it here.")


def acquire(directory: Path, session_id: str):
    if Path(session_id).name != session_id or any(c in session_id for c in "/\\:"):
        raise ValueError("invalid session ID")
    locks = directory / ".leases"
    locks.mkdir(parents=True, exist_ok=True)
    handle = (locks / f"{session_id}.lock").open("a+b")
    try:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SessionBusy(
                f"Session {session_id} already has an active writer."
                + describe_holder(locks / f"{session_id}.holder")
            ) from exc
        _record_holder(locks / f"{session_id}.holder")
        return handle
    except BaseException:
        handle.close()
        raise
