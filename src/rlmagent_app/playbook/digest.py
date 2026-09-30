"""The short view of the playbook the agent sees in its conversation."""

from __future__ import annotations

import hashlib
import math
import re

from rlmagent_app.playbook.store import KINDS, Entry

PER_KIND = 5
WIDTH = 200

_HEADER = """[Playbook] Entries you (or earlier sessions) kept for later. Follow the rules, \
use the memories. In the python tool: `await playbook("id")` shows an entry in full, \
`fn = await skill("id")` loads a skill as a function, and `await improve("what to \
keep")` asks for the playbook to be updated from this conversation once the turn ends."""


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", text.lower()))


def _weights(goal: str, recent: list[str]) -> dict[str, float]:
    """Goal words weigh 3; recent messages weigh from 2 (newest) down to 1 (oldest)."""
    weights: dict[str, float] = {}
    for i, text in enumerate(recent):
        w = 1 + (i / max(len(recent) - 1, 1))
        for term in _terms(text):
            weights[term] = max(weights.get(term, 0), w)
    for term in _terms(goal):
        weights[term] = 3.0
    return weights


def _score(entry_terms: set[str], weights: dict[str, float], rarity: dict[str, float]) -> float:
    return sum(weights[t] * rarity.get(t, 1.0) for t in entry_terms if t in weights)


def render(entries: list[Entry], goal: str = "", recent: list[str] | None = None,
           per_kind: int = PER_KIND) -> str:
    weights = _weights(goal, recent or [])
    terms = {e.id: _terms(f"{e.id} {e.title} {e.content}") for e in entries}
    counts: dict[str, int] = {}
    for ts in terms.values():
        for t in ts:
            counts[t] = counts.get(t, 0) + 1
    rarity = {t: math.log(1 + len(entries) / c) for t, c in counts.items()}
    lines = [_HEADER]
    for kind in KINDS:
        chosen = [e for e in entries if e.kind == kind]
        chosen.sort(key=lambda e: (-_score(terms[e.id], weights, rarity), -e.updated))
        for e in chosen[:per_kind]:
            text = " ".join(f"{e.title}: {e.content}".split())
            if len(text) > WIDTH:
                text = text[: WIDTH - 1] + "…"
            lines.append(f"- [{kind}] {e.id}: {text}")
    return "\n".join(lines)


def fingerprint(entries: list[Entry]) -> str:
    key = "|".join(sorted(f"{e.id}:{e.version}" for e in entries))
    return hashlib.sha1(key.encode()).hexdigest()[:12]
