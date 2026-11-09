"""Branch-local snapshots inherited along the saved conversation ancestry."""

from __future__ import annotations

import json
from uuid import uuid4

from rlmagent.harness.session.store import deserialize_records
from rlmagent.refinement.store import HarnessStore, Snapshot


class ScopedStore(HarnessStore):
    def __init__(self, path, session, inherited: Snapshot):
        super().__init__(path)
        self.session = session
        self.initial = inherited.model_copy(deep=True)
        self._document = {"base": self.initial.model_dump(), "anchors": {}}

    def _data(self):
        if self.path is None:
            return self._document
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._document

    def _save(self, data):
        if self.path is None:
            self._document = data
            return
        temporary = self.path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _lineage(self):
        tip = self.session._tip_id
        vault = self.session._vault
        parents = {}
        if vault is not None and vault.path.exists():
            records = deserialize_records(vault.path.read_text(encoding="utf-8").split("\n"))
            parents = {record.id: record.parent_id for record in records}
        seen = set()
        while tip is not None and tip not in seen:
            seen.add(tip)
            yield tip
            tip = parents.get(tip)
        yield "base"

    def _read(self):
        data = self._data()
        for anchor in self._lineage():
            versions = data["anchors"].get(anchor, [])
            if versions:
                return Snapshot.model_validate(versions[-1])
        return Snapshot.model_validate(data["base"])

    def initialize(self):
        with self.transaction():
            self._save(self._data())

    def commit(self, baseline: Snapshot, candidate: Snapshot):
        candidate = Snapshot.model_validate(candidate.model_dump())
        with self.transaction():
            if self._read() != baseline:
                raise ValueError("Branch guidance changed during review; start a fresh review")
            if candidate.revision != baseline.revision + 1:
                raise ValueError("Candidate must advance the harness revision once")
            data = self._data()
            anchor = self.session._tip_id or "base"
            history = data["anchors"].setdefault(anchor, [])
            if not history:
                history.append(baseline.model_dump())
            history.append(candidate.model_dump())
            self._save(data)
        return candidate.model_copy(deep=True)

    def version(self, revision: int):
        with self.transaction():
            data = self._data()
            for anchor in self._lineage():
                for snapshot in reversed(data["anchors"].get(anchor, [])):
                    if snapshot["revision"] == revision:
                        return Snapshot.model_validate(snapshot)
            baseline = Snapshot.model_validate(data["base"])
            if baseline.revision == revision:
                return baseline
        raise ValueError("No saved snapshot for this revision in the current branch")
