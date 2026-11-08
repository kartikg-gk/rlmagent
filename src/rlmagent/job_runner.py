"""Detached shell controller. Descriptors arrive on stdin, never through argv."""
from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def main():
    spec = json.loads(sys.stdin.readline())
    meta, log = Path(spec["metadata"]), Path(spec["log"])
    stop = meta.with_suffix(".stop")
    import psutil
    state = dict(spec["state"], controller=os.getpid(),
                 controller_started=psutil.Process().create_time(), heartbeat=time.time(),
                 pid=None, ended=None,
                 exit=None, timed_out=False, stopped=False, interrupted=False,
                 total_bytes=0, truncated=False)
    lock = threading.Lock()

    def save():
        with lock:
            state["heartbeat"] = time.time()
            stage = meta.with_suffix(".tmp")
            stage.write_text(json.dumps(state), encoding="utf-8")
            stage.replace(meta)

    proc = None
    try:
        proc = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=spec["env"],
            stdin=subprocess.PIPE if spec.get("keep_stdin_open") else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            **({} if os.name == "nt" else {"start_new_session": True}))
        state["pid"] = proc.pid
        save()

        def drain():
            stored = 0
            capped = False
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            with log.open("wb") as out:
                while chunk := proc.stdout.read1(8192):
                    remaining = 0 if capped else max(0, spec["max_bytes"] - stored)
                    shown = chunk[:remaining]
                    decoder.decode(shown, final=False)
                    out.write(shown)
                    stored += len(shown)
                    if remaining and len(chunk) >= remaining:
                        # The disk prefix must end at a complete Unicode character.
                        pending = len(decoder.getstate()[0])
                        if pending:
                            stored -= pending
                            out.truncate(stored)
                        capped = True
                    out.flush()
                    with lock:
                        state["total_bytes"] += len(chunk)
                        state["truncated"] = state["total_bytes"] > stored

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        deadline = time.monotonic() + spec["timeout"] if spec["timeout"] else None
        while proc.poll() is None:
            expired = deadline is not None and time.monotonic() >= deadline
            owner_gone = False
            if not state["persistent"]:
                try:
                    owner = psutil.Process(spec["owner"]["pid"])
                    owner_gone = abs(owner.create_time() - spec["owner"]["started"]) > 0.01
                except psutil.NoSuchProcess:
                    owner_gone = True
            if stop.exists() or expired or owner_gone:
                state["stopped"] = stop.exists() or owner_gone
                state["interrupted"] = owner_gone
                state["timed_out"] = expired
                if os.name == "nt":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    import signal
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                proc.wait()
                break
            save()
            time.sleep(0.1)
        reader.join(timeout=2)
        state["exit"] = proc.returncode
    except BaseException as exc:
        state["error"] = str(exc)
        state["exit"] = -1
    finally:
        state["ended"] = time.time()
        save()
        stop.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
