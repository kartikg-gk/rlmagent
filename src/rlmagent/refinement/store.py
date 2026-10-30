"""Validated harness snapshots and compare-and-swap persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CONTENT = 32 * 1024
MAX_ENTRIES = 128
_LOCKS: dict[str, threading.RLock] = {}
_LOCK_GUARD = threading.Lock()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Guidance(Record):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    kind: Literal["prompt", "skill", "subagent"]
    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_content(self):
        if not self.title.strip() or not self.content.strip():
            raise ValueError("Guidance needs a title and nonblank content")
        if len(self.content.encode("utf-8")) > MAX_CONTENT:
            raise ValueError("Guidance exceeds the byte limit")
        if self.kind == "skill":
            skill_parts(self.id, self.content)
        return self


def skill_parts(name: str, content: str) -> tuple[str, str]:
    match = re.fullmatch(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", content, re.S)
    if match is None:
        raise ValueError("A workflow must begin with YAML frontmatter")
    # Reject duplicate metadata keys rather than silently choosing the last value.
    try:
        node = yaml.compose(match[1], Loader=yaml.SafeLoader)
        meta = yaml.safe_load(match[1])
    except yaml.YAMLError as error:
        raise ValueError("Invalid workflow metadata") from error
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("Workflow metadata must be a mapping")
    keys = [key.value for key, _ in node.value]
    if sorted(keys) != ["description", "name"]:
        raise ValueError("Workflow metadata needs exactly name and description")
    if meta["name"] != name or not isinstance(meta["description"], str):
        raise ValueError("Workflow name must match its id; description must be text")
    if not meta["description"].strip() or not match[2].strip():
        raise ValueError("Workflow description and procedure must not be blank")
    return meta["description"], match[2]


class ChangeNote(Record):
    id: str
    summary: str = Field(min_length=1, max_length=2000)
    evidence: str = Field(min_length=1, max_length=8000)
    expected: str = Field(min_length=1, max_length=2000)
    edits: list[dict]


class Snapshot(Record):
    format: Literal["rlmagent.harness/1"] = "rlmagent.harness/1"
    revision: int = Field(default=0, ge=0)
    entries: dict[str, Guidance] = Field(default_factory=dict)
    history: list[ChangeNote] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_ids(self):
        if len(self.entries) > MAX_ENTRIES:
            raise ValueError("Too many guidance entries")
        if any(key != entry.id for key, entry in self.entries.items()):
            raise ValueError("Stored guidance id does not match its key")
        return self


@contextmanager
def _lock(path: Path):
    key = str(path.resolve())
    with _LOCK_GUARD:
        mutex = _LOCKS.setdefault(key, threading.RLock())
    with mutex:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class HarnessStore:
    def __init__(self, path: Path | None):
        self.path = path
        self._memory = Snapshot()
        self._mutex = threading.RLock()

    def _read(self) -> Snapshot:
        if self.path is None:
            return self._memory.model_copy(deep=True)
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Snapshot()
        return Snapshot.model_validate(json.loads(raw))

    @contextmanager
    def transaction(self):
        with self._mutex:
            if self.path is None:
                yield
            else:
                with _lock(self.path.with_suffix(".lock")):
                    yield

    def read(self) -> Snapshot:
        with self.transaction():
            return self._read()

    def commit(self, baseline: Snapshot, candidate: Snapshot) -> Snapshot:
        candidate = Snapshot.model_validate(candidate.model_dump())
        with self.transaction():
            if self._read() != baseline:
                raise ValueError("Harness changed during refinement; start a fresh review")
            if candidate.revision != baseline.revision + 1:
                raise ValueError("Candidate must advance the harness revision once")
            if self.path is None:
                self._memory = candidate.model_copy(deep=True)
            else:
                descriptor, name = tempfile.mkstemp(prefix=".harness-", dir=self.path.parent)
                temporary = Path(name)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                        output.write(candidate.model_dump_json(indent=2) + "\n")
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, self.path)
                finally:
                    temporary.unlink(missing_ok=True)
            return candidate.model_copy(deep=True)
