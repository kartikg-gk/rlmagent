"""Shell jobs and services the harness owns, and watches that wait for something to change.

A job runs in the background with its output kept in a file, so a cell can start it,
carry on, and read what is new from a cursor. Jobs belong to the session, not to a
kernel: a sandbox that restarts or a lost variable does not lose them. A service is a
job expected to keep running (a dev server); starting one can wait for a line that says
it is ready.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

MAX_READ = 64 * 1024
MAX_JOBS = 32
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
    proc: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None

    @property
    def done(self) -> bool:
        return self.ended is not None

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "command": self.command,
                "service": self.service, "done": self.done, "exit": self.exit,
                "timed_out": self.timed_out, "stopped": self.stopped,
                "seconds": round((self.ended or time.time()) - self.started, 1),
                "bytes": self.log.stat().st_size if self.log.exists() else 0}


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

    def __init__(self, cwd: str, folder: Path, approve=None, check=None) -> None:
        self.cwd = cwd
        self.folder = folder
        self.approve = approve  # async (command) -> None, raises when refused
        self.check = check  # (command) -> reason or None, for policy refusals
        self.jobs: dict[str, Job] = {}

    def _get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id) or next(
            (j for j in self.jobs.values() if j.name and j.name == job_id), None)
        if job is None:
            raise ValueError(f"no job {job_id!r}; jobs() lists them")
        return job

    async def _launch(self, command: str, timeout: float | None, service: bool, name: str) -> Job:
        from rlmagent.model.external.cli import own_group

        if self.check is not None and (reason := self.check(command)):
            raise ValueError(reason)
        if self.approve is not None:
            await self.approve(command)
        if sum(not j.done for j in self.jobs.values()) >= MAX_JOBS:
            raise ValueError(f"{MAX_JOBS} jobs are already running; stop some first")
        job_id = uuid.uuid4().hex[:8]
        self.folder.mkdir(parents=True, exist_ok=True)
        job = Job(job_id, command, self.folder / f"{job_id}.log", service, name)
        out = open(job.log, "wb")
        job.proc = await asyncio.create_subprocess_exec(
            _bash(), "--noprofile", "--norc", "-c", command, cwd=self.cwd,
            stdin=asyncio.subprocess.DEVNULL, stdout=out, stderr=asyncio.subprocess.STDOUT,
            **own_group())
        out.close()
        job.task = asyncio.create_task(self._reap(job, timeout))
        self.jobs[job_id] = job
        return job

    async def _reap(self, job: Job, timeout: float | None) -> None:
        try:
            await asyncio.wait_for(job.proc.wait(), timeout)
        except asyncio.TimeoutError:
            job.timed_out = True
            self._kill(job)
            await job.proc.wait()
        finally:
            job.exit = job.proc.returncode
            job.ended = time.time()

    @staticmethod
    def _kill(job: Job) -> None:
        from rlmagent.model.external.cli import kill_tree

        if job.proc is not None and job.proc.returncode is None:
            kill_tree(job.proc)

    # -- functions agents call ---------------------------------------------

    async def job_start(self, command: str, timeout: float | None = None, name: str = "") -> dict:
        """Run a shell command in the background; returns {"id", ...}. Read its output with
        job_read(id, cursor) and wait for it with job_wait(id)."""
        job = await self._launch(str(command), float(timeout) if timeout else None, False, name)
        return job.summary()

    async def job_read(self, job_id: str, cursor: int = 0) -> dict:
        """Output written since cursor (a byte offset): {"text", "cursor", "done", "exit"}.
        Pass the returned cursor next time to get only what is new."""
        job = self._get(job_id)
        with open(job.log, "rb") as f:
            f.seek(max(0, int(cursor)))
            data = f.read(MAX_READ)
        return {"text": data.decode("utf-8", errors="replace"), "cursor": int(cursor) + len(data),
                "more": len(data) == MAX_READ, "done": job.done, "exit": job.exit,
                "timed_out": job.timed_out}

    async def job_wait(self, job_id: str, timeout: float = 300.0) -> dict:
        """Wait until a job ends, at most timeout seconds; returns its summary."""
        job = self._get(job_id)
        try:
            await asyncio.wait_for(asyncio.shield(job.task), float(timeout))
        except asyncio.TimeoutError:
            pass
        return job.summary()

    async def job_stop(self, job_id: str) -> dict:
        """Stop a job or service and everything it started."""
        job = self._get(job_id)
        job.stopped = True
        self._kill(job)
        await asyncio.shield(job.task)
        return job.summary()

    async def jobs_list(self) -> list[dict]:
        """Every job and service of this session, running or finished."""
        return [job.summary() for job in self.jobs.values()]

    async def service_start(self, name: str, command: str, ready: str | None = None,
                            wait: float = 60.0) -> dict:
        """Start a long-running process such as a dev server, under a name. With ready, a
        regular expression, wait up to wait seconds for a line that matches it."""
        for job in self.jobs.values():
            if job.name == name and not job.done:
                raise ValueError(f"service {name!r} is already running; job_stop it first")
        job = await self._launch(str(command), None, True, str(name))
        summary = job.summary()
        if ready:
            pattern = re.compile(ready)
            deadline = time.monotonic() + float(wait)
            while time.monotonic() < deadline and not job.done:
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
                    return {"event": "path", "paths": changed[:200], "truncated": len(changed) > 200}
            if inbox is not None and (messages := inbox()):
                return {"event": "message", "messages": messages}
            await asyncio.sleep(POLL)
        return {"event": "timeout"}

    async def close(self) -> None:
        for job in self.jobs.values():
            self._kill(job)
        for job in self.jobs.values():
            if job.task is not None:
                await asyncio.gather(job.task, return_exceptions=True)

    def functions(self, inbox=None) -> dict:
        async def watch(job=None, cursor=0, path=None, messages=False, timeout=300.0):
            """Wait for a change: job=ID (new output past cursor, or its end), path=PATH (a
            file or folder changing), messages=True (another agent writing to you)."""
            return await self.watch(job, cursor, path, inbox if messages else None, timeout)

        return {"job_start": self.job_start, "job_read": self.job_read,
                "job_wait": self.job_wait, "job_stop": self.job_stop, "jobs": self.jobs_list,
                "service_start": self.service_start, "watch": watch}


def _stamp(target: Path) -> dict:
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
            try:
                stamps[str(full)] = full.stat().st_mtime_ns
            except OSError:
                pass
            if len(stamps) > 5000:
                return stamps
    return stamps


__all__ = ["Job", "Jobs"]
