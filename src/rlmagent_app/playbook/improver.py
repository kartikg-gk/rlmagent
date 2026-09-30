"""Runs playbook updates for one session: when asked, between turns, with undo."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from rlmagent_app.playbook.apply import apply_edits, reverse_edits
from rlmagent_app.playbook.digest import fingerprint, render
import time

from rlmagent_app.playbook.plan import (
    AUTO_NOTE,
    CHECK_CHARS,
    CHECK_SYSTEM,
    SYSTEM,
    build_request,
    parse_check,
    parse_proposal,
)
from rlmagent_app.playbook.store import Playbook, load, merged, save


class Improver:
    def __init__(self, session, local_path: Path | None, shared_path: Path | None) -> None:
        self.session = session
        self.paths = {"local": local_path, "global": shared_path}
        self.local = load(local_path)
        self.shared = load(shared_path)
        self._pending: list[tuple[str | None, bool, str]] = []
        self._shown: str | None = None
        self.auto = True
        self.every = 20
        self.cooldown = 900.0
        self._counted = 0
        self._last_check = float("-inf")

    def _store(self, scope: str) -> Playbook:
        if self.paths[scope] is not None:
            fresh = load(self.paths[scope])
            setattr(self, "local" if scope == "local" else "shared", fresh)
        return self.local if scope == "local" else self.shared

    def entries(self) -> list:
        return merged(self.local, self.shared)

    def entry(self, entry_id: str) -> dict:
        for e in self.entries():
            if e.id == entry_id:
                return asdict(e)
        raise KeyError(f"no playbook entry {entry_id!r}")

    def request(self, instructions: str | None, shared: bool = False, trigger: str = "agent") -> None:
        self._pending.append((instructions, shared, trigger))

    async def run_pending(self) -> list[str]:
        notes = []
        while self._pending:
            instructions, shared, trigger = self._pending.pop(0)
            notes.append(await self.improve(instructions, shared=shared, trigger=trigger))
        return notes

    async def after_turn(self) -> None:
        """Every `every` assistant turns (at most once per `cooldown` seconds), check."""
        if not self.auto:
            return
        turns = sum(1 for m in self.session.transcript if getattr(m, "role", "") == "assistant")
        if turns - self._counted < self.every or time.monotonic() - self._last_check < self.cooldown:
            return
        self._counted = turns
        await self._auto("auto:turns")

    async def before_compaction(self) -> None:
        if self.auto:
            await self._auto("auto:compaction")

    async def _auto(self, trigger: str) -> None:
        self._last_check = time.monotonic()
        tail = self._conversation()[-CHECK_CHARS:]
        worth_it, focus = parse_check(await self.session._utility_completion(tail, CHECK_SYSTEM))
        if worth_it:
            instructions = f"{AUTO_NOTE}\nFocus: {focus}" if focus else AUTO_NOTE
            await self.improve(instructions, trigger=trigger)

    async def improve(self, instructions: str | None, *, shared: bool = False,
                      trigger: str = "user") -> str:
        scope = "global" if shared else "local"
        target = self._store(scope)
        baseline = {k: e.version for k, e in target.entries.items()}
        request = build_request(self.entries(), target.history, self._conversation(),
                                scope=scope, instructions=instructions)
        try:
            proposal = parse_proposal(await self.session._utility_completion(request, SYSTEM))
        except ValueError as exc:
            return f"Playbook not changed: {exc}"
        event = self.apply(proposal["edits"], trigger=trigger, scope=scope, baseline=baseline,
                           summary=str(proposal.get("summary", "")),
                           rationale=str(proposal.get("rationale", "")))
        return _describe(event)

    def apply(self, edits: list[dict], *, trigger: str, scope: str = "local",
              baseline: dict | None = None, summary: str = "", rationale: str = "") -> dict:
        target = self._store(scope)
        if baseline is None:
            baseline = {k: e.version for k, e in target.entries.items()}
        event = apply_edits(target, edits, baseline=baseline, scope=scope, trigger=trigger,
                            summary=summary, rationale=rationale,
                            read_only=self.shared if scope == "local" else None)
        save(self.paths[scope], target)
        return event

    def undo(self, event_id: str) -> str:
        for scope in ("local", "global"):
            target = self._store(scope)
            for event in target.history:
                if event["id"] == event_id:
                    undone = self.apply(reverse_edits(event), trigger="undo", scope=scope,
                                        summary=f"undo {event_id}")
                    return f"Change {event_id} undone. " + _describe(undone)
        return f"No playbook change with id {event_id}."

    def show_digest(self) -> None:
        """Put the playbook in front of the model when it has changed since last shown."""
        entries = self.entries()
        fp = fingerprint(entries)
        if not entries or fp == self._shown:
            return
        self._shown = fp
        recent = " ".join(getattr(m, "text", "") for m in self.session.transcript[-4:])
        self.session.inject(render(entries, recent))

    def _conversation(self) -> str:
        return "\n".join(
            f"[{getattr(m, 'role', '?')}] {getattr(m, 'text', '')}" for m in self.session.transcript
        )

    async def command(self, arg: str) -> str:
        words = arg.split()
        if words[:1] == ["list"]:
            return "\n".join(f"{e.id} [{e.kind}, {e.scope}] {e.title}" for e in self.entries()) \
                or "The playbook is empty."
        if words[:1] == ["undo"] and len(words) == 2:
            return self.undo(words[1])
        shared = "--shared" in words
        instructions = " ".join(w for w in words if w != "--shared") or None
        note = await self.improve(instructions, shared=shared, trigger="user")
        self.show_digest()
        return note


def _describe(event: dict) -> str:
    done = [f"{c['action']} {c['id']}" for c in event["changes"] if c["applied"]]
    skipped = [f"{c['id']} ({c['error']})" for c in event["changes"] if not c["applied"]]
    text = f"[{event['id']}] " + (", ".join(done) if done else "no change")
    return text + (f"; skipped: {', '.join(skipped)}" if skipped else "")
