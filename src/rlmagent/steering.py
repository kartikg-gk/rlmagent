"""Accepted instructions and model-boundary delivery receipts."""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from uuid import uuid4

from rlmagent.refinement.store import _lock


class Steering:
    def __init__(self, path: Path | None):
        self.path = path
        self.memory = {}
        self._mutex = threading.RLock()

    def _read(self):
        if self.path is None:
            with self._mutex:
                return copy.deepcopy(self.memory)
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _write(self, records):
        if self.path is None:
            with self._mutex:
                self.memory = records
            return
        temporary = self.path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(records), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _guard(self):
        return self._mutex if self.path is None else _lock(self.path.with_suffix(".lock"))

    def accept(self, text: str, identity: str | None = None, target: str = "root", *,
               once: bool = False, allow_new: bool = True):
        identity = identity or uuid4().hex
        if (
            not isinstance(identity, str)
            or not 1 <= len(identity) <= 128
            or not isinstance(target, str)
            or not 1 <= len(target) <= 256
            or not isinstance(text, str)
            or not text.strip()
            or len(text) > 65536
        ):
            raise ValueError("Invalid steering identity, target or text")

        def update():
            records = self._read()
            if identity in records:
                previous = records[identity]
                if (previous["text"] != text or previous["target"] != target
                        or previous.get("once", False) != once):
                    raise ValueError("messageId was already used with different content or target")
                return self.receipt(identity, previous)
            if not allow_new:
                raise ValueError("child is idle or finished; use send for a follow-up turn")
            if len(records) >= 10000:
                raise ValueError("Steering identity limit reached")
            records[identity] = {
                "text": text,
                "target": target,
                "accepted_at": time.time(),
                "deliveries": {},
            }
            if once:
                records[identity]["once"] = True
            self._write(records)
            return self.receipt(identity, records[identity])

        with self._guard():
            return update()

    @staticmethod
    def receipt(identity, record):
        return {
            "messageId": identity,
            "target": record["target"],
            "status": ("consumed" if record["deliveries"] else
                       "expired" if "expired_at" in record else "accepted"),
            "deliveries": dict(record["deliveries"]),
        }

    def receipts(self, *, target: str | None = None):
        return [self.receipt(identity, record) for identity, record in self._read().items()
                if target is None or record["target"] == target]

    def context(self, agent: str, *, root: bool = False, child: str | None = None):
        targets = {agent}
        if root:
            targets.add("root")
        if child is not None:
            targets.add(child)
        instructions, pending = [], []
        for identity, record in self._read().items():
            target = record["target"]
            if target not in targets or "expired_at" in record:
                continue
            delivered = (child if target == child else agent) in record["deliveries"]
            if not delivered or not record.get("once", False):
                instructions.append(record["text"])
            if not delivered:
                pending.append(identity)
        return "\n\n".join(instructions), pending

    def consumed(self, identities, agent: str, request: str | None, *, child: str | None = None):
        if not identities:
            return

        def update():
            records = self._read()
            for identity in identities:
                record = records[identity]
                record["deliveries"].setdefault(
                    child if record["target"] == child else agent,
                    {
                        "requestId": request,
                        "consumed_at": time.time(),
                    },
                )
            self._write(records)

        with self._guard():
            update()

    def expire_child(self, child: str) -> None:
        """Close undelivered one-shot instructions at the end of a child turn."""
        with self._guard():
            records = self._read()
            pending = [record for record in records.values()
                       if record["target"] == child and record.get("once")
                       and not record["deliveries"] and "expired_at" not in record]
            if pending:
                finished = time.time()
                for record in pending:
                    record["expired_at"] = finished
                self._write(records)
