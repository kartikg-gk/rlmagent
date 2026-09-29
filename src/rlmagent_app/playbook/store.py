"""Playbook entries and their JSON file."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = ("rule", "memory", "skill", "role")


@dataclass
class Entry:
    id: str
    kind: str
    title: str
    content: str
    scope: str
    callable: str | None = None
    version: int = 1
    updated: float = field(default_factory=time.time)


@dataclass
class Playbook:
    entries: dict[str, Entry] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


def load(path: Path | None) -> Playbook:
    if path is None:
        return Playbook()
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = {e["id"]: Entry(**e) for e in doc.get("entries", [])}
        return Playbook(entries=entries, history=list(doc.get("history", [])))
    except (OSError, ValueError, TypeError, KeyError):
        return Playbook()


def save(path: Path | None, pb: Playbook) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"entries": [asdict(e) for e in pb.entries.values()], "history": pb.history}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def merged(local: Playbook, shared: Playbook) -> list[Entry]:
    """Shared entries plus local ones; a local id that clashes is shown as `local:<id>`."""
    out = list(shared.entries.values())
    for entry in local.entries.values():
        if entry.id in shared.entries:
            entry = Entry(**{**asdict(entry), "id": f"local:{entry.id}"})
        out.append(entry)
    return out
