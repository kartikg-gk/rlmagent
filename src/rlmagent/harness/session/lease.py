"""OS-backed single-writer ownership for a saved session."""

from __future__ import annotations

import os
from pathlib import Path


class SessionBusy(RuntimeError):
    pass


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
            raise SessionBusy(f"Session {session_id} already has an active writer.") from exc
        return handle
    except BaseException:
        handle.close()
        raise
