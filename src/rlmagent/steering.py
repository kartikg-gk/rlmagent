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

    def accept(self, text: str, identity: str | None = None, target: str = "root"):
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
                if previous["text"] != text or previous["target"] != target:
                    raise ValueError("messageId was already used with different content or target")
                return self.receipt(identity, previous)
            if len(records) >= 10000:
                raise ValueError("Steering identity limit reached")
            records[identity] = {
                "text": text,
                "target": target,
                "accepted_at": time.time(),
                "deliveries": {},
            }
            self._write(records)
            return self.receipt(identity, records[identity])

        if self.path is None:
            with self._mutex:
                return update()
        with _lock(self.path.with_suffix(".lock")):
            return update()

    @staticmethod
    def receipt(identity, record):
        return {
            "messageId": identity,
            "target": record["target"],
            "status": "consumed" if record["deliveries"] else "accepted",
            "deliveries": dict(record["deliveries"]),
        }

    def receipts(self):
        return [self.receipt(identity, record) for identity, record in self._read().items()]

    def context(self, agent: str, *, root: bool = False):
        selected = [
            (identity, record)
            for identity, record in self._read().items()
            if record["target"] == agent or (root and record["target"] == "root")
        ]
        # Keep accepted guidance in subsequent requests and through transcript rollback.
        text = "\n\n".join(record["text"] for _, record in selected)
        pending = [identity for identity, record in selected if agent not in record["deliveries"]]
        return text, pending

    def consumed(self, identities, agent: str, request: str | None):
        if not identities:
            return

        def update():
            records = self._read()
            for identity in identities:
                records[identity]["deliveries"].setdefault(
                    agent,
                    {
                        "requestId": request,
                        "consumed_at": time.time(),
                    },
                )
            self._write(records)

        if self.path is None:
            with self._mutex:
                update()
        else:
            with _lock(self.path.with_suffix(".lock")):
                update()
