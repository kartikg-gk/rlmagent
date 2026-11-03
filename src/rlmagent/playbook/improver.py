"""Runs playbook updates for one session: when asked, between turns, with undo."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import json
import time

from rlmagent.playbook.apply import apply_edits, reverse_edits
from rlmagent.playbook.digest import fingerprint, render

from rlmagent.playbook.plan import (
    AUTO_NOTE,
    CHECK_CHARS,
    CHECK_SYSTEM,
    SYSTEM,
    build_request,
    parse_check,
    parse_proposal,
)
from rlmagent.playbook.store import Playbook, load, merged, save

#: How long a memory saved without being asked for lasts, in finished tasks. It
#: was written from one moment of the work, which is what makes it go stale.
AUTO_MEMORY_TASKS = 3
#: Failed tool calls in one task that are worth a look for something to keep.
TROUBLE_CALLS = 3


class Improver:
    def __init__(self, session, local_path: Path | None, shared_path: Path | None) -> None:
        self.writable = True
        self.session = session
        self.paths = {"local": local_path, "global": shared_path}
        self.local = load(local_path)
        self.shared = load(shared_path)
        self._pending: list[tuple[str | None, bool, str]] = []
        self._shown: str | None = None
        self.auto = True
        self.every = 5
        self.cooldown = 900.0
        self._counted = 0
        self._finished = 0
        self._last_check = float("-inf")
        self._compacted = False
        # When set, proposals from automatic checks wait in held for /improve accept.
        self.hold_auto = False
        self.held: list[dict] = []

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
        if not self.writable:
            raise RuntimeError("playbook is read-only")
        self._pending.append((instructions, shared, trigger))

    async def run_pending(self) -> list[str]:
        notes = []
        while self._pending:
            instructions, shared, trigger = self._pending.pop(0)
            notes.append(await self.improve(instructions, shared=shared, trigger=trigger))
        return notes

    async def after_turn(self) -> None:
        """Every `every` finished tasks (at most once per `cooldown` seconds), check.

        Tasks rather than model turns: one task can take dozens of turns, and a
        count of those lands in the middle of work that has nothing to teach yet.
        A task that went badly is checked sooner, since that is where the lessons are.
        """
        if not self.auto or time.monotonic() - self._last_check < self.cooldown:
            return
        evidence = getattr(self.session, "evidence", None)
        if evidence is not None and evidence.last_failed >= TROUBLE_CALLS:
            self._counted = self._finished
            await self._auto("auto:trouble")
        elif self._finished - self._counted >= self.every:
            self._counted = self._finished
            await self._auto("auto:tasks")

    def compacted(self) -> None:
        """Note a compaction; the check runs at the next turn boundary, never inside a turn."""
        self._compacted = True

    def drop_pending(self) -> None:
        self._pending.clear()
        self._compacted = False

    async def at_boundary(self) -> None:
        self._finished += 1
        await self._task_finished()
        await self.run_pending()
        if self._compacted:
            self._compacted = False
            if self.auto:
                await self._auto("auto:compaction")
        await self.after_turn()

    async def _task_finished(self) -> None:
        """Count the task in the session's playbook, and drop what has now gone stale."""
        if not self.writable:
            return
        target = self._store("local")
        target.tasks += 1
        save(self.paths["local"], target)
        stale = [e.id for e in target.entries.values()
                 if e.expires_after_tasks is not None
                 and target.tasks - e.made_at_task >= e.expires_after_tasks]
        if stale:
            event = self.apply([{"action": "delete", "id": eid} for eid in stale],
                               trigger="expire", summary="written for work that has moved on")
            await self._record(event)

    async def _auto(self, trigger: str) -> None:
        self._last_check = time.monotonic()
        kept = "\n".join(f"- {e.id}: {e.title}: {e.content}" for e in self.entries()) or "(empty)"
        prompt = f"Current playbook:\n{kept}\n\nRecent conversation:\n{self._conversation()[-CHECK_CHARS:]}"
        worth_it, focus = parse_check(await self.session._utility_completion(prompt, CHECK_SYSTEM))
        if worth_it:
            instructions = f"{AUTO_NOTE}\nFocus: {focus}" if focus else AUTO_NOTE
            await self.improve(instructions, trigger=trigger)

    async def improve(self, instructions: str | None, *, shared: bool = False,
                      trigger: str = "user") -> str:
        if not self.writable:
            return "Playbook is read-only."
        scope = "global" if shared else "local"
        target = self._store(scope)
        baseline = {k: e.version for k, e in target.entries.items()}
        request = build_request(self.entries(), target.history, self._conversation(),
                                scope=scope, instructions=instructions)
        try:
            proposal = parse_proposal(await self.session._utility_completion(request, SYSTEM))
        except ValueError as exc:
            return f"Playbook not changed: {exc}"
        if trigger.startswith("auto:"):
            for edit in proposal["edits"]:
                if isinstance(edit, dict) and edit.get("kind") == "memory":
                    edit.setdefault("expires_after_tasks", AUTO_MEMORY_TASKS)
            if self.hold_auto and proposal["edits"]:
                self.held.append({"edits": proposal["edits"], "trigger": trigger, "scope": scope,
                                  "summary": str(proposal.get("summary", "")),
                                  "rationale": str(proposal.get("rationale", ""))})
                return (f"Held proposal {len(self.held)} for review: "
                        f"{proposal.get('summary', '')} (/improve pending)")
        event = self.apply(proposal["edits"], trigger=trigger, scope=scope, baseline=baseline,
                           summary=str(proposal.get("summary", "")),
                           rationale=str(proposal.get("rationale", "")))
        await self._record(event)
        return _describe(event)

    async def _record(self, event: dict) -> None:
        """Log the change in the session, and in the shared log when it is shared."""
        brief = {k: event[k] for k in ("id", "trigger", "scope", "summary", "time")}
        brief["changes"] = [{k: c[k] for k in ("action", "id", "applied")} for c in event["changes"]]
        note = getattr(self.session, "note_playbook_change", None)
        if note is not None:
            await note(brief)
        shared = self.paths["global"]
        if event["scope"] == "global" and shared is not None:
            log = shared.with_name("playbook-changes.jsonl")
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")

    def apply(self, edits: list[dict], *, trigger: str, scope: str = "local",
              baseline: dict | None = None, summary: str = "", rationale: str = "") -> dict:
        if not self.writable:
            raise RuntimeError("playbook is read-only")
        target = self._store(scope)
        if baseline is None:
            baseline = {k: e.version for k, e in target.entries.items()}
        event = apply_edits(target, edits, baseline=baseline, scope=scope, trigger=trigger,
                            summary=summary, rationale=rationale,
                            read_only=self.shared if scope == "local" else None)
        save(self.paths[scope], target)
        done = sorted({c["id"] for c in event["changes"] if c["applied"]})
        evidence = getattr(self.session, "evidence", None)
        if done and evidence is not None:
            evidence.saved(event["id"], f"playbook:{trigger}", done, summary)
        return event

    async def undo(self, event_id: str) -> str:
        for scope in ("local", "global"):
            target = self._store(scope)
            for event in target.history:
                if event["id"] == event_id:
                    undone = self.apply(reverse_edits(event), trigger="undo", scope=scope,
                                        summary=f"undo {event_id}")
                    await self._record(undone)
                    return f"Change {event_id} undone. " + _describe(undone)
        return f"No playbook change with id {event_id}."

    def show_digest(self) -> None:
        """Put the playbook in front of the model when it has changed since last shown."""
        entries = self.entries()
        fp = fingerprint(entries)
        if fp == self._shown or (not entries and self._shown is None):
            return
        self._shown = fp
        if not entries:
            self.session.inject("[Playbook] Earlier entries no longer apply; it is now empty.")
            return
        transcript = self.session.transcript
        goal = next((getattr(m, "text", "") for m in transcript if getattr(m, "role", "") == "user"), "")
        recent = [getattr(m, "text", "") for m in transcript[-4:]]
        self.session.inject(render(entries, goal=goal, recent=recent))

    def _conversation(self) -> str:
        return "\n".join(
            f"[{getattr(m, 'role', '?')}] {getattr(m, 'text', '')}" for m in self.session.transcript
        )

    def pending_text(self) -> str:
        if not self.held:
            return "No proposals are waiting."
        rows = []
        for n, held in enumerate(self.held, 1):
            rows.append(f"{n}. [{held['trigger']}] {held['summary']}")
            rows += [f"   {e.get('action')} {e.get('kind', '')} {e.get('id', '')}: "
                     f"{str(e.get('content', e.get('title', '')))[:200]}" for e in held["edits"]]
        return "\n".join(rows)

    async def _decide(self, words: list[str]) -> str:
        if len(words) != 2 or not words[1].isdigit() or not 1 <= int(words[1]) <= len(self.held):
            return f"Use /improve {words[0]} N, with N from /improve pending."
        held = self.held.pop(int(words[1]) - 1)
        if words[0] == "reject":
            return f"Dropped: {held['summary']}"
        event = self.apply(held["edits"], trigger=held["trigger"], scope=held["scope"],
                           summary=held["summary"], rationale=held["rationale"])
        await self._record(event)
        self.show_digest()
        return _describe(event)

    async def command(self, arg: str) -> str:
        words = arg.split()
        if words[:1] == ["pending"]:
            return self.pending_text()
        if words[:1] in (["accept"], ["reject"]):
            return await self._decide(words)
        if words[:1] == ["list"]:
            return "\n".join(f"{e.id} [{e.kind}, {e.scope}] {e.title}" for e in self.entries()) \
                or "The playbook is empty."
        if words[:1] == ["undo"] and len(words) == 2:
            return await self.undo(words[1])
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
