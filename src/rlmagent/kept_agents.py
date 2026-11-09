"""Checkpoint kept recursive agents alongside a saved conversation."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from rlmness.background import Background, Job
from rlmness.session import Session

from rlmagent.agents.child_handles import checked_name


class KeptAgents(Background):
    def __init__(self, max_agents):
        super().__init__(max_agents)
        self.directory: Path | None = None
        self.saved: dict[str, dict] = {}
        self.live_states: dict[str, Session] = {}
        self.names: dict[str, str] = {}
        self.steering = None

    def _expire_steering(self, key: str) -> None:
        if self.steering is not None:
            self.steering.expire_child(key)

    def close(self):
        super().close()
        for state in list(self.live_states.values()):
            state.close_runtime()
        self.live_states.clear()

    def open(self, state_path: Path | None) -> None:
        if state_path is None or self.directory is not None:
            return
        self.directory = state_path.with_suffix(".children")
        for path in sorted(self.directory.glob("*.json")):
            if not re.fullmatch(r"[a-f0-9]{32}", path.stem):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            deliveries = record.get("deliveries", {})
            interrupted = [item for item in deliveries.values() if item["status"] == "running"]
            for item in interrupted:
                item["status"] = "interrupted"
            job = Job(parent=record.get("parent", "root"), keep=True, status="idle")
            job.value = record.get("value")
            job.outbox = record.get("messages", [])
            job.ready.set()
            self.saved[path.stem] = record
            self.jobs[path.stem] = job
            if name := record.get("name"):
                self.names[path.stem] = name
            if interrupted:
                self._expire_steering(path.stem)
                self._write(path.stem)

    def _write(self, key: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        staged = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            staged.write_text(json.dumps(self.saved[key], allow_nan=False), encoding="utf-8")
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)

    def node_records(self):
        with self.lock:
            return {key: dict(record) for key, record in self.saved.items()}

    def node_state(self, key):
        with self.lock:
            state = self.live_states.get(key)
            if state is None:
                state = (Session.load(self.directory / f"{key}.state.json")
                         if self.directory is not None else Session())
                self.live_states[key] = state
            return state

    def child_handle(self, parent, action, key=None, name=None):
        with self.lock:
            if action in ("name", "find"):
                name = checked_name(name)
                match = next((other for other, alias in self.names.items()
                              if alias == name and self.jobs[other].parent == parent
                              and self.jobs[other].status != "cancelled"), None)
                if action == "find":
                    if match is None:
                        raise ValueError(f"no child named {name!r}")
                    return {"id": match}
                if match is not None and match != key:
                    raise ValueError(f"child name {name!r} is already in use")
            job = self.jobs[key]
            if job.parent != parent or job.status == "cancelled":
                raise ValueError("child is not available to this agent")
            if action == "name":
                if key in self.saved:
                    old = self.saved[key].get("name")
                    self.saved[key]["name"] = name
                    try:
                        self._write(key)
                    except OSError:
                        if old is None:
                            self.saved[key].pop("name", None)
                        else:
                            self.saved[key]["name"] = old
                        raise
                self.names[key] = name
            elif action != "info":
                raise ValueError("invalid child handle operation")
            deliveries = self.saved.get(key, {}).get("deliveries", {})
            return {"id": key, "name": self.names.get(key), "status": job.status, "keep": job.keep,
                    "queued": sum(item["status"] == "queued" for item in deliveries.values())}

    def node_record(self, key, record):
        """Mirror an execution-side job and persist its settled mailbox."""
        if not re.fullmatch(r"[a-f0-9]{32}", key):
            raise ValueError("invalid child identity")
        with self.lock:
            job = self.jobs.setdefault(key, Job(parent=record["parent"], keep=record["keep"]))
            prior = self.saved.get(key, {}).get("deliveries", {})
            current = record.get("deliveries", {})
            if record["status"] in ("idle", "done", "failed", "cancelled") or any(
                item["status"] == "running" and current.get(receipt, {}).get("status") != "running"
                for receipt, item in prior.items()
            ):
                self._expire_steering(key)
            job.status = record["status"]
            job.value = record.get("value")
            job.outbox = list(record.get("messages", []))
            if job.status in ("idle", "done", "failed", "cancelled"):
                job.ready.set()
            else:
                job.ready.clear()
            if job.status == "cancelled":
                self.names.pop(key, None)
                job.token.set()
                self.saved.pop(key, None)
                state = self.live_states.pop(key, None)
                if state is not None:
                    state.close_runtime()
                if self.directory is not None:
                    (self.directory / f"{key}.json").unlink(missing_ok=True)
                    (self.directory / f"{key}.state.json").unlink(missing_ok=True)
            elif job.keep and self.directory is not None:
                self.saved[key] = dict(record)
                if name := self.names.get(key):
                    self.saved[key]["name"] = name
                self._write(key)

    def bindings(self, parent, child):
        owner = self.owners.get(parent)
        if owner in self.saved:
            # Each reopened agent gets a fresh run ID. Its mailbox and children use its saved ID.
            parent = owner
            self.owners[parent] = owner
        bindings = super().bindings(parent, child)
        if (parent != "root" and parent not in self.saved) or self.directory is None:
            return bindings

        def enqueue(record, text):
            receipt = uuid.uuid4().hex
            record.setdefault("deliveries", {})[receipt] = {"status": "queued", "text": text}
            return receipt

        def start(key):
            job = self.jobs[key]
            record = self.saved[key]
            job.status = "starting"
            job.error = None
            job.ready.clear()

            def work():
                current = None
                try:
                    with self.lock:
                        state = self.live_states.get(key)
                        if state is None:
                            state = Session.load(self.directory / f"{key}.state.json")
                            self.live_states[key] = state
                    while not job.token.is_set():
                        with self.lock:
                            if job.token.is_set():
                                return
                            current = next((item for item in record.get("deliveries", {}).values()
                                            if item["status"] == "queued"), None)
                            if current is None:
                                job.status = "idle"
                                job.ready.set()
                                return
                            question = current.pop("text")
                            current["status"] = "running"
                            job.status = "running"
                            self._write(key)
                        value = child(record["data"], question, job.token, saved=state,
                                      owner=key, model=record["model"])
                        with self.lock:
                            if job.token.is_set():
                                return
                            try:
                                json.dumps(value, allow_nan=False)
                            except (ValueError, TypeError):
                                record["value"] = str(value)
                            else:
                                record["value"] = value
                            record["messages"] = list(job.outbox)
                            current["status"] = "done"
                            self._expire_steering(key)
                            self._write(key)
                            job.value = value
                except Exception as exc:
                    with self.lock:
                        if not job.token.is_set():
                            job.error = exc
                            job.status = "failed"
                            if current is not None and current["status"] == "running":
                                current["status"] = "failed"
                            job.ready.set()
                            self._expire_steering(key)
                            self._write(key)

            job.future = self.pool.submit(work)

        def spawn(task, data=None, keep=False, role=None, model=None):
            if not keep:
                return bindings["_spawn"](task, data, keep, role, model)
            if role:
                if self.resolve_role is None:
                    raise ValueError("no role resolver configured")
                task = self.resolve_role(role) + "\n\n" + task
            record = {"data": data, "model": model, "parent": parent}
            json.dumps(record, allow_nan=False)
            with self.lock:
                if len(self.jobs) >= self.max_agents:
                    raise RuntimeError("background agent limit reached")
                key = uuid.uuid4().hex
                enqueue(record, task)
                self.saved[key] = record
                self.jobs[key] = Job(parent=parent, keep=True)
                try:
                    self._write(key)
                except OSError:
                    del self.saved[key]
                    del self.jobs[key]
                    raise
                start(key)
            return key

        def send(key, text):
            if key not in self.saved:
                return bindings["_send"](key, text)
            with self.lock:
                job = self.jobs[key]
                if job.parent != parent:
                    raise ValueError("job does not belong to this agent")
                record = self.saved[key]
                receipt = enqueue(record, str(text))
                try:
                    self._write(key)
                except OSError:
                    del record["deliveries"][receipt]
                    raise
                if job.status in ("idle", "failed"):
                    start(key)
                return receipt

        def delivery(key, receipt):
            with self.lock:
                if self.jobs[key].parent != parent:
                    raise ValueError("job does not belong to this agent")
                item = self.saved.get(key, {}).get("deliveries", {}).get(receipt)
                return item["status"] if item is not None else None

        def cancel(key):
            with self.lock:
                bindings["_cancel"](key)
                pending = [key]
                while pending:
                    item = pending.pop()
                    self.names.pop(item, None)
                    self._expire_steering(item)
                    if item not in self.saved:
                        continue
                    del self.saved[item]
                    job = self.jobs[item]
                    job.token.set()
                    live = self.live_states.pop(item, None)
                    if live is not None:
                        live.close_runtime()
                    job.status = "cancelled"
                    job.ready.set()
                    pending.extend(k for k, record in self.saved.items() if record.get("parent") == item)
                    (self.directory / f"{item}.json").unlink(missing_ok=True)
                    state = self.directory / f"{item}.state.json"
                    if job.future is not None:
                        job.future.add_done_callback(lambda _future, path=state: path.unlink(missing_ok=True))
                    else:
                        state.unlink(missing_ok=True)

        def messages(key):
            with self.lock:
                items = bindings["_messages"](key)
                if key in self.saved and self.jobs[key].status == "idle":
                    self.saved[key]["messages"] = []
                    self._write(key)
                return items

        with self.lock:
            for key, record in self.saved.items():
                if (self.jobs[key].parent == parent and self.jobs[key].status == "idle"
                        and any(item["status"] == "queued" for item in record.get("deliveries", {}).values())):
                    start(key)

        return {**bindings, "_spawn": spawn, "_send": send, "_cancel": cancel, "_messages": messages,
                "_delivery": delivery}
