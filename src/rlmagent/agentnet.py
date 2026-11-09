"""Agents on this machine finding and messaging each other, scheduling work, reporting progress.

Every interactive session registers itself under <home>/agents/<id>/ while it runs; workers
are already listed under <home>/workers/. A message to a worker becomes its next task; a
message to a session waits in its inbox and is shown to it before its next turn.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

MAX_MESSAGE = 8_000
MAX_MESSAGES = 50


def _home() -> Path:
    from rlmagent.plugins import rlm_agent_home

    return rlm_agent_home() / "agents"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
    staged.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(staged, path)


def _read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class AgentNet:
    """This agent's place among the others: its name, inbox, and what it can reach."""

    def __init__(self, me: str, cwd: str, worker: str | None = None) -> None:
        self.me = me
        self.cwd = cwd
        self.worker = worker  # set when this session is a background worker
        self.on_progress = None  # called with progress text, to show it to the user

    # -- registration ------------------------------------------------------

    def _folder(self, name: str) -> Path:
        return _home() / name

    def register(self) -> None:
        if self.worker is None:
            _write(self._folder(self.me) / "meta.json", {
                "pid": os.getpid(), "cwd": self.cwd, "started": time.time(),
                "beat": time.time()})

    def beat(self) -> None:
        meta = _read(self._folder(self.me) / "meta.json")
        if meta is not None:
            meta["beat"] = time.time()
            _write(self._folder(self.me) / "meta.json", meta)

    def leave(self) -> None:
        (self._folder(self.me) / "meta.json").unlink(missing_ok=True)

    # -- functions agents call ---------------------------------------------

    async def list_agents(self) -> list[dict]:
        """Other agents on this machine you can message: background workers and open sessions."""
        from rlmagent import worker
        from rlmagent.worker import _pid_alive

        found = []
        base = worker.root()
        for path in sorted(base.iterdir()) if base.is_dir() else []:
            if path.is_dir():
                meta = worker.state_of(path.name)
                found.append({"name": path.name, "kind": "worker",
                              "state": meta.get("state", "?") if meta["alive"] else "not running",
                              "cwd": meta.get("cwd"), "me": path.name == self.worker})
        sessions = _home()
        for path in sorted(sessions.iterdir()) if sessions.is_dir() else []:
            meta = _read(path / "meta.json")
            if meta is None:
                continue
            alive = bool(meta.get("pid")) and _pid_alive(int(meta["pid"]))
            if alive or path.name == self.me:
                found.append({"name": path.name, "kind": "session", "state": "open",
                              "cwd": meta.get("cwd"), "me": path.name == self.me})
        return found

    async def send_message(self, to: str, text: str) -> str:
        """Send text to another agent by name (see list_agents). A worker runs it as a task;
        a session sees it before its next turn."""
        from rlmagent import worker

        text = str(text)[:MAX_MESSAGE]
        sender = self.worker or self.me
        if to in (self.me, self.worker):
            raise ValueError("that is you; pick another agent from list_agents()")
        try:
            is_worker = worker.folder(to).is_dir()
        except ValueError:
            is_worker = False
        if is_worker:
            worker.send(to, f"[message from {sender}] {text}", source=sender)
            return f"sent to worker {to}"
        if (self._folder(to) / "meta.json").exists():
            _write(self._folder(to) / "inbox" / f"{time.time_ns()}-{uuid.uuid4().hex[:6]}.json",
                   {"from": sender, "text": text, "time": time.time()})
            return f"sent to session {to}"
        raise ValueError(f"no agent named {to!r}; list_agents() shows who is there")

    async def read_messages(self) -> list[dict]:
        """Messages other agents sent you since you last read them: [{"from", "text", "time"}]."""
        return self.take()

    def take(self) -> list[dict]:
        taken = []
        for path in sorted((self._folder(self.me) / "inbox").glob("*.json"))[:MAX_MESSAGES]:
            message = _read(path)
            path.unlink(missing_ok=True)
            if message:
                taken.append(message)
        return taken

    async def schedule(self, task: str, at: float | None = None, in_seconds: float | None = None,
                       every: float | None = None, worker: str | None = None, *,
                       cron: str | None = None, timezone: str | None = None) -> str:
        """Run task later on a background worker: at a unix time, or in_seconds from now,
        optionally every N seconds. Alternatively use a five-field cron expression and
        an IANA timezone (defaults to UTC). Defaults to this worker when you are one."""
        from rlmagent import worker as workers

        target = worker or self.worker
        if target is None:
            raise ValueError("this session is not a worker; pass worker=NAME (see list_agents())")
        if sum(value is not None for value in (at, in_seconds, cron)) != 1:
            raise ValueError("give exactly one of at, in_seconds, or cron")
        due = None
        if at is not None:
            due = float(at)
        elif in_seconds is not None:
            due = time.time() + float(in_seconds)
        if every is not None and float(every) < 60:
            raise ValueError("every must be at least 60 seconds")
        job = workers.schedule(target, str(task), due, float(every) if every else None,
                               cron=cron, timezone=timezone)
        when = workers.job_time(job)
        return f"job {job['id']} on {target} at {when}" + (f", every {every}s" if every else "")

    async def heartbeat(self, text: str) -> str:
        """Report progress on long work; the user sees it while you keep going."""
        from rlmagent import worker

        text = str(text)[:500]
        if self.worker is not None:
            worker.log_line(self.worker, "progress", text)
        if self.on_progress is not None:
            self.on_progress(text)
        return "noted"

    def functions(self) -> dict:
        return {"list_agents": self.list_agents, "send_message": self.send_message,
                "read_messages": self.read_messages, "schedule": self.schedule,
                "heartbeat": self.heartbeat}


def inbox_note(messages: list[dict]) -> str:
    """Messages put in front of the model before its next turn."""
    lines = [f"[message from {m.get('from', '?')}] {m.get('text', '')}" for m in messages]
    return "Messages from other agents arrived:\n" + "\n".join(lines)


__all__ = ["AgentNet", "inbox_note"]
