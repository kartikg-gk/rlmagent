"""The short view of the playbook the agent sees in its conversation."""

from __future__ import annotations

import hashlib
import re

from rlmagent_app.playbook.store import KINDS, Entry

PER_KIND = 5
WIDTH = 200

_HEADER = """[Playbook] Entries you (or earlier sessions) kept for later. Follow the rules, \
use the memories. In the python tool: `await playbook("id")` shows an entry in full, \
`fn = await skill("id")` loads a skill as a function, and `await improve("what to \
keep")` asks for the playbook to be updated from this conversation once the turn ends."""


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]{3,}", text.lower())}


def render(entries: list[Entry], query: str) -> str:
    wanted = _terms(query)
    lines = [_HEADER]
    for kind in KINDS:
        chosen = [e for e in entries if e.kind == kind]
        chosen.sort(key=lambda e: (-len(wanted & _terms(f"{e.id} {e.title} {e.content}")), -e.updated))
        for e in chosen[:PER_KIND]:
            text = " ".join(f"{e.title}: {e.content}".split())
            if len(text) > WIDTH:
                text = text[: WIDTH - 1] + "…"
            lines.append(f"- [{kind}] {e.id}: {text}")
    return "\n".join(lines)


def fingerprint(entries: list[Entry]) -> str:
    key = "|".join(sorted(f"{e.id}:{e.version}" for e in entries))
    return hashlib.sha1(key.encode()).hexdigest()[:12]
