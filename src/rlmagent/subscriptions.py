"""Saved subscriptions for workspace changes and session jobs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from functools import wraps
from math import isfinite
from pathlib import Path
from time import time
from uuid import uuid4

MAX_WATCHES = 32
MAX_EVENTS = 200
POLL_SECONDS = 1.0


@dataclass
class ActivityRequest:
    watch_id: str | None
    previous: set[str]
    timeout: float
    owner: str | None
    deadline: float | None = None


class Subscriptions:
    def __init__(self, storage: Path, snapshot):
        self.storage = storage
        self.snapshot = snapshot
        self.task = None
        try:
            self.watches = json.loads(storage.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.watches = {}

    def _save(self, watches):
        self.storage.parent.mkdir(parents=True, exist_ok=True)
        staged = self.storage.with_suffix(f".{uuid4().hex}.tmp")
        try:
            staged.write_text(json.dumps(watches, ensure_ascii=False), encoding="utf-8")
            staged.replace(self.storage)
        finally:
            staged.unlink(missing_ok=True)
        self.watches = watches

    def _put(self, watch):
        self._save({**self.watches, watch["id"]: watch})

    def _visible(self, owner):
        return {key: watch for key, watch in self.watches.items()
                if watch.get("owner") in (None, owner)}

    def _get(self, identity, owner=None):
        if not isinstance(identity, str) or identity not in (visible := self._visible(owner)):
            raise ValueError("unknown watch ID")
        return visible[identity]

    @staticmethod
    def _summary(watch):
        return {key: watch[key] for key in ("id", "kind", "target", "active", "dropped")} | {
            "unread": sum(event["status"] == "unread" for event in watch["events"]),
        }

    async def watch_start(self, job=None, path=None):
        """Subscribe to one job or workspace path. Returns a persistent watch ID."""
        if (job is None) == (path is None):
            raise ValueError("give exactly one of job or path")
        kind, target = ("job", job) if job is not None else ("path", path)
        return self._subscribe(kind, target)

    def watch_child(self, child_id, owner):
        return self._subscribe("child", child_id, owner)

    def _subscribe(self, kind, target, owner=None):
        if not isinstance(target, str) or not target:
            raise ValueError("watch target must be a nonempty string")
        if len(self.watches) >= MAX_WATCHES:
            raise ValueError("watch limit reached; remove an old watch with watch_cancel")
        target, state = self.snapshot(kind, target)
        watch = {"id": uuid4().hex, "kind": kind, "target": target, "state": state, "owner": owner,
                 "events": [], "active": kind != "job" or not state["done"], "dropped": 0}
        self._put(watch)
        self.start()
        return self._summary(watch)

    async def watch_list(self, *, owner=None):
        """List subscriptions and their unread counts."""
        self.poll()
        return [self._summary(watch) for watch in self._visible(owner).values()]

    async def watch_events(self, watch_id, unread_only=True, *, owner=None):
        """List event metadata without changing read state."""
        self.poll()
        return [{k: v for k, v in event.items() if k != "change"}
                for event in self._get(watch_id, owner)["events"]
                if not unread_only or event["status"] == "unread"]

    async def watch_read(self, watch_id, event_id, *, owner=None):
        """Read one event by ID. Repeated reads return the same event."""
        watch = self._get(watch_id, owner)
        for event in watch["events"]:
            if event["id"] != event_id:
                continue
            if event["status"] == "read":
                return dict(event)
            read = {**event, "status": "read"}
            self._put({**watch, "events": [read if row is event else row for row in watch["events"]]})
            return read
        raise ValueError("unknown watch event ID")

    async def watch_cancel(self, watch_id, *, owner=None):
        """Remove a subscription and its retained events."""
        self._get(watch_id, owner)
        self._save({key: value for key, value in self.watches.items() if key != watch_id})
        return {"id": watch_id, "active": False}

    def poll(self):
        changed = {}
        for key, watch in self.watches.items():
            if not watch["active"]:
                continue
            try:
                _, state = self.snapshot(watch["kind"], watch["target"])
            except (OSError, ValueError):
                # A target may be briefly unavailable during a job update or file swap.
                continue
            prior = watch["state"]
            if prior == state:
                continue
            if watch["kind"] == "path":
                paths = sorted(name for name in prior.keys() | state.keys()
                               if prior.get(name) != state.get(name))
                change = {"paths": paths[:200], "truncated": len(paths) > 200}
            elif watch["kind"] == "job":
                change = {"start": prior["bytes"], "end": state["bytes"],
                          "done": state["done"], "exit": state["exit"]}
            else:
                change = dict(state)
            events = [*watch["events"], {"id": uuid4().hex, "status": "unread",
                                        "time": time(), "change": change}]
            dropped = watch["dropped"] + sum(event["status"] == "unread" for event in events[:-MAX_EVENTS])
            ended = (watch["kind"] == "job" and state["done"] or
                     watch["kind"] == "child" and state["status"] in ("cancelled", "missing"))
            changed[key] = {**watch, "state": state, "events": events[-MAX_EVENTS:], "dropped": dropped,
                            "active": not ended}
        if changed:
            self._save({**self.watches, **changed})

    def prepare_wait(self, watch_id, timeout, *, owner=None):
        timeout = float(timeout)
        if not isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("wait timeout must be between 0 and 300 seconds")
        if watch_id is not None:
            self._get(watch_id, owner)
        self.poll()
        return ActivityRequest(watch_id, self._event_ids(watch_id, owner), timeout, owner)

    def _event_ids(self, watch_id, owner):
        return {event["id"] for key, watch in self._visible(owner).items()
                if watch_id is None or key == watch_id for event in watch["events"]}

    def wait_status(self, request, steering_pending):
        loop = asyncio.get_running_loop()
        if request.deadline is None:
            request.deadline = loop.time() + request.timeout
        watch_id = request.watch_id
        if steering_pending():
            return {"event": "steering"}
        if watch_id is not None and watch_id not in self.watches:
            return {"event": "cancelled", "watch_id": watch_id}
        self.poll()
        if self._event_ids(watch_id, request.owner) - request.previous:
            return {"event": "watch", "watch_id": watch_id}
        remaining = request.deadline - loop.time()
        if remaining <= 0:
            return {"event": "timeout"}
        return {"retry_after": min(POLL_SECONDS, remaining)}

    async def wait_for_change(self, request, steering_pending):
        while True:
            status = self.wait_status(request, steering_pending)
            if "event" in status:
                return status
            await asyncio.sleep(status["retry_after"])

    def start(self):
        if self.task is None and any(watch["active"] for watch in self.watches.values()):
            self.task = asyncio.create_task(self._monitor())

    async def _monitor(self):
        try:
            while any(watch["active"] for watch in self.watches.values()):
                try:
                    self.poll()
                except OSError:
                    # Keep the prior cursor and retry persistence on the next sample.
                    pass
                await asyncio.sleep(POLL_SECONDS)
        finally:
            self.task = None

    async def close(self):
        if self.task is not None:
            task = self.task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.task = None

    def functions(self, *, owner=None):
        def scoped(method):
            @wraps(method)
            async def call(*args, **kwargs):
                return await method(*args, **kwargs, owner=owner)
            return call

        return {"watch_start": self.watch_start, **{method.__name__: scoped(method) for method in (
            self.watch_list, self.watch_events, self.watch_read, self.watch_cancel)}}
