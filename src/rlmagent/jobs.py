"""Shell jobs and services the harness owns, and watches that wait for something to change.

A job runs in the background with its output kept in a file, so a cell can start it,
carry on, and read what is new from a cursor. Jobs belong to the session, not to a
kernel: a sandbox that restarts or a lost variable does not lose them. A service is a
job expected to keep running (a dev server); starting one can wait for a line that says
it is ready.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rlmness.environment import execution_environment

from rlmagent.subscriptions import Subscriptions

MAX_READ = 64 * 1024
MAX_JOBS = 32
MAX_LOG = 8 * 1024 * 1024
POLL = 0.2


@dataclass
class Job:
    id: str
    command: str
    log: Path
    service: bool = False
    name: str = ""
    started: float = field(default_factory=time.time)
    ended: float | None = None
    exit: int | None = None
    timed_out: bool = False
    stopped: bool = False
    proc: subprocess.Popen | None = None
    persistent: bool = False
    total_bytes: int = 0
    truncated: bool = False
    interrupted: bool = False

    @property
    def metadata(self):
        return self.log.with_suffix(".json")

    def refresh(self):
        if not self.metadata.exists():
            return
        try:
            state = json.loads(self.metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # caught mid-update; the next refresh reads the new state
        if state.get("ended") is None and time.time() - state.get("heartbeat", 0) > 3:
            import psutil
            try:
                controller = psutil.Process(state["controller"])
                alive = abs(controller.create_time() - state["controller_started"]) < 0.01
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                alive = False
            if not alive:
                state.update(ended=time.time(), exit=-1, interrupted=True)
        for key in ("ended", "exit", "timed_out", "stopped", "total_bytes", "truncated",
                    "interrupted"):
            if key in state:
                setattr(self, key, state[key])
        if self.proc is not None:
            self.proc.poll()

    @property
    def done(self) -> bool:
        return self.ended is not None

    def summary(self) -> dict:
        self.refresh()
        return {"id": self.id, "name": self.name, "command": self.command,
                "service": self.service, "done": self.done, "exit": self.exit,
                "timed_out": self.timed_out, "stopped": self.stopped,
                "seconds": round((self.ended or time.time()) - self.started, 1),
                "bytes": self.log.stat().st_size if self.log.exists() else 0,
                "total_bytes": self.total_bytes, "truncated": self.truncated,
                "interrupted": self.interrupted,
                "persistent": self.persistent}


def _bash() -> str:
    found = shutil.which("bash")
    if found is None and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if candidate.is_file():
            found = str(candidate)
    if found is None:
        raise OSError("Bash is required for shell jobs")
    return found


class Jobs:
    """The session's background jobs. `approve(command)` raises when a command is refused."""

    def __init__(self, cwd: str, folder: Path, approve=None, check=None, *, env=None,
                 max_log_bytes: int = MAX_LOG, shell_access="host", shell_network=False) -> None:
        self.cwd = cwd
        self.folder = folder
        self.approve = approve  # async (command) -> None, raises when refused
        self.check = check  # (command) -> reason or None, for policy refusals
        self.jobs: dict[str, Job] = {}
        self.subscriptions = Subscriptions(folder / ".watches", self._watch_snapshot)
        self.child_snapshot = None
        self.env = dict(env or {})
        self.shell_access, self.shell_network = shell_access, shell_network
        self.max_log_bytes = max(4, int(max_log_bytes))
        for path in self.folder.glob("*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                key = state["id"]
                if not re.fullmatch(r"[a-f0-9]{8}", key):
                    continue
                job = Job(key, state["command"], self.folder / f"{key}.log",
                          state["service"], state["name"], started=state["started"],
                          persistent=state.get("persistent", False))
                job.refresh()
                self.jobs[key] = job
            except (OSError, ValueError, KeyError):
                continue

    def _get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id) or next(
            (j for j in self.jobs.values() if j.name and j.name == job_id), None)
        if job is None:
            raise ValueError(f"no job {job_id!r}; jobs() lists them")
        job.refresh()
        return job

    async def _launch(self, command: str, timeout: float | None, service: bool, name: str,
                      persistent: bool = False, argv=None, env=None) -> Job:
        if self.check is not None and (reason := self.check(command)):
            raise ValueError(reason)
        if self.approve is not None:
            await self.approve(command)
        if sum(not j.summary()["done"] for j in self.jobs.values()) >= MAX_JOBS:
            raise ValueError(f"{MAX_JOBS} jobs are already running; stop some first")
        job_id = uuid.uuid4().hex[:8]
        self.folder.mkdir(parents=True, exist_ok=True)
        job = Job(job_id, command, self.folder / f"{job_id}.log", service, name,
                  persistent=persistent)
        job.log.touch()
        isolated = self.shell_access == "workspace"
        target_argv = argv
        if isolated:
            from rlmagent.shell_sandbox import plan
            target_argv = await asyncio.to_thread(plan, self.cwd, command, argv=argv,
                env={**self.env, **(env or {})}, network=self.shell_network)
        else:
            target_argv = argv or [_bash(), "--noprofile", "--norc", "-c", command]
        import psutil
        spec = {"metadata": str(job.metadata.resolve()), "log": str(job.log.resolve()),
                "owner": {"pid": os.getpid(), "started": psutil.Process().create_time()},
                "cwd": self.cwd, "argv": target_argv, "keep_stdin_open": isolated,
                "env": execution_environment({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                                              **self.env, **(env or {})}),
                "timeout": timeout, "max_bytes": self.max_log_bytes,
                "state": {"id": job.id, "command": command, "service": service,
                          "name": name, "started": job.started, "persistent": persistent}}
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS}
                   if os.name == "nt" else {"start_new_session": True})
        runner = str(Path(__file__).with_name("job_runner.py"))
        job.proc = subprocess.Popen([sys.executable, "-I", runner],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=execution_environment(), **options)
        job.proc.stdin.write((json.dumps(spec) + "\n").encode("utf-8"))
        job.proc.stdin.close()
        self.jobs[job_id] = job
        while not job.metadata.exists():
            if job.proc.poll() is not None:
                raise RuntimeError("job controller failed before accepting the command")
            await asyncio.sleep(0.02)
        return job

    @staticmethod
    def _kill(job: Job) -> None:
        job.refresh()
        if not job.done:
            job.metadata.with_suffix(".stop").touch()

    # -- functions agents call ---------------------------------------------

    async def job_start(self, command: str = "", timeout: float | None = None, name: str = "",
                        *, argv=None, env=None) -> dict:
        """Run a shell command in the background; returns {"id", ...}. Read its output with
        job_read(id, cursor) and wait for it with job_wait(id)."""
        if argv is not None:
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                raise ValueError("argv must be a nonempty list of strings")
            command = subprocess.list2cmdline(argv)
        job = await self._launch(str(command), float(timeout) if timeout else None, False, name,
                                 argv=argv, env=env)
        return job.summary()

    async def job_read(self, job_id: str, cursor: int = 0) -> dict:
        """Output written since cursor (a byte offset): {"text", "cursor", "done", "exit"}.
        Pass the returned cursor next time to get only what is new."""
        job = self._get(job_id)
        with open(job.log, "rb") as f:
            offset = max(0, min(int(cursor), job.log.stat().st_size))
            f.seek(offset)
            data = f.read(MAX_READ)
        # Keep an incomplete trailing UTF-8 sequence for the next read. Invalid bytes
        # within output remain visible as replacement characters.
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        text = decoder.decode(data, final=job.done and offset + len(data) == job.log.stat().st_size)
        pending = decoder.getstate()[0]
        used = len(data) - len(pending)
        return {"text": text, "cursor": offset + used,
                "more": offset + used < job.log.stat().st_size, "done": job.done, "exit": job.exit,
                "timed_out": job.timed_out}

    async def job_wait(self, job_id: str, timeout: float = 300.0) -> dict:
        """Wait until a job ends, at most timeout seconds; returns its summary."""
        job = self._get(job_id)
        deadline = time.monotonic() + max(0, float(timeout))
        while not job.done and time.monotonic() < deadline:
            await asyncio.sleep(POLL)
            job.refresh()
        return job.summary()

    async def job_stop(self, job_id: str) -> dict:
        """Stop a job or service and everything it started."""
        job = self._get(job_id)
        job.stopped = True
        self._kill(job)
        return await self.job_wait(job.id, 30)

    async def jobs_list(self) -> list[dict]:
        """Every job and service of this session, running or finished."""
        return [job.summary() for job in self.jobs.values()]

    async def service_start(self, name: str, command: str, ready: str | None = None,
                            wait: float = 60.0, *, persist: bool = False, env=None) -> dict:
        """Start a long-running process such as a dev server, under a name. With ready, a
        regular expression, wait up to wait seconds for a line that matches it."""
        for job in self.jobs.values():
            job.refresh()
            if job.name == name and not job.done:
                raise ValueError(f"a service named {name!r} is up; stop it with job_stop first")
        job = await self._launch(str(command), None, True, str(name), persist, env=env)
        summary = job.summary()
        if ready:
            pattern = re.compile(ready)
            deadline = time.monotonic() + float(wait)
            while time.monotonic() < deadline and not job.done:
                job.refresh()
                if pattern.search(job.log.read_text(encoding="utf-8", errors="replace")):
                    summary = job.summary()
                    summary["ready"] = True
                    return summary
                await asyncio.sleep(POLL)
            summary = job.summary()
            summary["ready"] = False
        return summary

    async def watch(self, job: str | None = None, cursor: int = 0, path: str | None = None,
                    inbox=None, timeout: float = 300.0) -> dict:
        """Wait until something changes: a job writes past cursor or ends (job=ID), a file or
        folder changes (path=...), or a message arrives (inbox). Returns what happened, or
        {"event": "timeout"}."""
        deadline = time.monotonic() + float(timeout)
        target = None
        before = None
        if path is not None:
            target = (Path(self.cwd) / path).resolve()
            before = _stamp(target)
        tracked = self._get(job) if job is not None else None
        while time.monotonic() < deadline:
            if tracked is not None:
                tracked.refresh()
                size = tracked.log.stat().st_size if tracked.log.exists() else 0
                if size > int(cursor):
                    return {"event": "output", "job": tracked.id, "start": int(cursor), "end": size}
                if tracked.done:
                    return {"event": "ended", "job": tracked.id, "exit": tracked.exit}
            if target is not None:
                now = _stamp(target)
                if now != before:
                    changed = sorted(k for k in set(now) | set(before or {})
                                     if (now or {}).get(k) != (before or {}).get(k))
                    return {"event": "path", "paths": changed[:200],
                            "truncated": len(changed) > 200}
            if inbox is not None and (messages := inbox()):
                return {"event": "message", "messages": messages}
            await asyncio.sleep(POLL)
        return {"event": "timeout"}

    def _watch_snapshot(self, kind, target):
        if kind == "child":
            if self.child_snapshot is None:
                raise ValueError("no child watch source is attached")
            return target, self.child_snapshot(target)
        if kind == "job":
            job = self._get(target)
            return job.id, {key: value for key, value in job.summary().items()
                            if key in {"bytes", "total_bytes", "done", "exit"}}
        root = Path(self.cwd).resolve()
        absolute = (root / target).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError(f"{target} is outside the workspace")
        storage = str(self.subscriptions.storage.resolve())
        return absolute.relative_to(root).as_posix(), {
            Path(name).relative_to(root).as_posix(): stamp
            for name, stamp in _stamp(absolute, follow_files=False).items() if name != storage
        }

    async def close(self) -> None:
        await self.subscriptions.close()
        for job in self.jobs.values():
            if not job.persistent:
                self._kill(job)
        for job in self.jobs.values():
            if not job.persistent:
                await self.job_wait(job.id, 30)

    def functions(self, inbox=None) -> dict:
        async def watch(job=None, cursor=0, path=None, messages=False, timeout=300.0):
            """Wait for a change: job=ID (new output past cursor, or its end), path=PATH (a
            file or folder changing), messages=True (another agent writing to you)."""
            return await self.watch(job, cursor, path, inbox if messages else None, timeout)

        return {"job_start": self.job_start, "job_read": self.job_read,
                "job_wait": self.job_wait, "job_stop": self.job_stop, "jobs": self.jobs_list,
                "service_start": self.service_start, "watch": watch,
                **self.subscriptions.functions()}


def _stamp(target: Path, *, follow_files=True) -> dict:
    """Modification times under a path (one level of files for a folder, skipping noise)."""
    if target.is_file():
        return {str(target): target.stat().st_mtime_ns}
    if not target.is_dir():
        return {}
    stamps = {}
    for folder, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "__pycache__"}]
        for name in files:
            full = Path(folder) / name
            if not follow_files and full.is_symlink():
                continue
            try:
                stamps[str(full)] = full.stat().st_mtime_ns
            except OSError:
                pass
            if len(stamps) > 5000:
                return stamps
    return stamps


__all__ = ["Job", "Jobs"]
