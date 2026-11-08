"""Single-job refinement with activation at session boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from rlmagent.refinement.engine import evidence_text, review
from rlmagent.refinement.store import HarnessStore, skill_parts
from rlmagent.skillset import Skill


#: A saved note is sent with every request, so a long one is paid for every time.
DIGEST_CHARS = 400
AUTO_COOLDOWN_TURNS = 3
BUSY_PHASES = ("reviewing", "awaiting review", "ready", "applying")


class Signal:
    def __init__(self, enabled=None):
        self.cancelled = False
        self.enabled = enabled

    def is_cancelled(self):
        return self.cancelled or (self.enabled is not None and not self.enabled())


class Refinement:
    def __init__(self, session, store: HarnessStore):
        self.session, self.store = session, store
        self.active = store.read()
        self.base_system = session._system
        self.base_backend = getattr(getattr(session, "backend", None), "instructions", None)
        self.installed = list(session._skills)
        self.task = None
        self.signal = Signal(self.enabled)
        self.generation = 0
        self.phase = "idle"
        self.detail = "No manual review has been requested."
        self.job_id = None
        self.draft = None
        self.automatic = "off"
        self._automatic_job = False
        self._pending_evidence = None
        self._settled_turns = 0
        self._next_review_turn = 0
        # A proposal that changes guidance waits for /refine apply before it is saved.
        self.review_first = True
        self._apply_active()

    def writable(self):
        if not self.enabled():
            return False
        if self.session.plan_mode:
            return False
        improver = getattr(self.session, "improver", None)
        return getattr(improver, "writable", True)

    def enabled(self) -> bool:
        home = getattr(self, "home", None)
        if home is None:
            return getattr(self, "_enabled", True)
        try:
            value = json.loads((home / "refinement.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return True
        if not isinstance(value, dict) or value.keys() != {"enabled"} or type(value["enabled"]) is not bool:
            raise ValueError("Invalid global refinement preference")
        return value["enabled"]

    def status(self):
        identity = f" {self.job_id}" if self.job_id else ""
        return f"Refinement{identity}: {self.phase}. {self.detail}"

    async def command(self, argument):
        mode = argument.strip()
        if mode == "auto" or mode.startswith("auto "):
            selection = mode.removeprefix("auto").strip()
            if not selection:
                return f"Automatic harness refinement: {self.automatic}."
            if selection not in ("review", "apply", "off"):
                return "Use /refine auto review|apply|off."
            if selection != "off" and not self.writable():
                return "Automatic refinement requires writable harness guidance."
            if self._automatic_job and selection != self.automatic:
                self.cancel()
            self.automatic = selection
            if selection == "off":
                self._pending_evidence = None
            return f"Automatic harness refinement: {selection}."
        if mode.startswith("scope"):
            words = mode.split()
            if len(words) == 1:
                return "Refinement scope: " + getattr(self, "scope", "project")
            if len(words) != 2 or words[1] not in ("session", "project"):
                return "Use /refine scope session|project."
            if self.session.harness.active or not self.writable():
                return "Changing scope requires an idle writable session."
            self.cancel()
            from rlmagent.refinement.scoped import ScopedStore

            if not hasattr(self, "project_store"):
                self.project_store = self.store
            if words[1] == "session":
                folder = (project_folder(self.home, self.session.cwd) / "sessions" /
                          self.session.session_id) if hasattr(self, "home") and self.store.path else None
                path = folder / "state.json" if folder else None
                self.store = ScopedStore(path, self.session, self.active)
                self.store.initialize()
            else:
                self.store = self.project_store
            self.scope = words[1]
            self.active = self.store.read()
            if hasattr(self, "home") and self.project_store.path is not None:
                folder = project_folder(self.home, self.session.cwd) / "sessions" / self.session.session_id
                folder.mkdir(parents=True, exist_ok=True)
                preference = folder / "scope.json"
                temporary = preference.with_suffix(f".{uuid4().hex}.tmp")
                try:
                    temporary.write_text(json.dumps({"scope": self.scope}), encoding="utf-8")
                    temporary.replace(preference)
                finally:
                    temporary.unlink(missing_ok=True)
            self._apply_active()
            return "Refinement scope: " + self.scope
        if mode in ("on", "off"):
            self.cancel()
            self._enabled = mode == "on"
            if home := getattr(self, "home", None):
                from rlmagent.refinement.store import _lock

                home.mkdir(parents=True, exist_ok=True)
                with _lock(home / "refinement.lock"):
                    path = home / "refinement.json"
                    staged = path.with_suffix(f".{uuid4().hex}.tmp")
                    try:
                        staged.write_text(json.dumps({"enabled": self._enabled}), encoding="utf-8")
                        staged.replace(path)
                    finally:
                        staged.unlink(missing_ok=True)
            self._apply_active()
            return "Harness refinements " + ("enabled." if self._enabled else "disabled.")
        if mode.startswith("rollback "):
            if self.session.harness.active or not self.writable():
                return "Rollback requires an idle writable session with refinements enabled."
            from rlmagent.refinement.store import ChangeNote

            try:
                revision = int(mode.split()[1])
                baseline = self.store.read()
                target = self.store.version(revision)
                candidate = baseline.model_copy(deep=True)
                candidate.entries = target.entries
                candidate.revision += 1
                candidate.history.append(ChangeNote(
                    id=uuid4().hex, summary=f"Restore guidance from revision {revision}",
                    evidence="User requested rollback", expected="Restore the selected saved guidance",
                    edits=[{"action": "rollback", "revision": revision}],
                ))
                self.cancel()
                self.active = self.store.commit(baseline, candidate)
                self._apply_active()
                return f"Restored guidance from revision {revision}."
            except (ValueError, IndexError) as exc:
                return str(exc)
        if not self.enabled() and mode not in ("status", "cancel"):
            return "Harness refinements are disabled. Use /refine on to enable them."
        if argument.strip() == "status":
            return self.status()
        if argument.strip() == "cancel":
            self.cancel()
            return self.status()
        if argument.strip() == "show":
            if self.draft is None:
                return "No proposal is waiting. " + self.status()
            return proposal_diff(self.draft.baseline, self.draft.state)
        if argument.strip() == "apply":
            if self.phase != "awaiting review":
                return "No proposal is waiting for review. " + self.status()
            self.phase, self.detail = "ready", "Approved proposal awaits a turn boundary."
            if not self.session.harness.active:
                self.at_boundary()
            return self.status()
        words = argument.split()
        evidence = getattr(self.session, "evidence", None)
        if words[:1] == ["evidence"]:
            return evidence.report() if evidence is not None else "No evidence is kept here."
        if words[:1] == ["rate"]:
            if evidence is None or len(words) != 3:
                return "Use /refine rate <change> better|same|worse."
            return evidence.rate(words[1], words[2])
        if not self.writable():
            return "Refinement is unavailable while guidance is read-only."
        if self.phase in BUSY_PHASES:
            return self.status() + " Finish or cancel this review before starting another."
        return self._start_review(argument.strip() or None, evidence_text(list(self.session.transcript)))

    def compacted(self, transcript) -> None:
        if self.automatic != "off" and self.writable():
            self._pending_evidence = evidence_text(transcript)
            self._maybe_start_review()

    def settled(self) -> None:
        self._settled_turns += 1
        self._maybe_start_review()

    def _maybe_start_review(self) -> None:
        if (
            self._pending_evidence is None or self.session.harness.active
            or self.phase in BUSY_PHASES or self._settled_turns < self._next_review_turn
            or not self.writable()
        ):
            return
        evidence, self._pending_evidence = self._pending_evidence, None
        self._next_review_turn = self._settled_turns + AUTO_COOLDOWN_TURNS
        self._start_review("Review the run that led to compaction.", evidence, automatic=True)

    def _start_review(self, focus, evidence, *, automatic=False):
        baseline = self.store.read()
        provider, model = self.session._provider, self.session.model
        self.generation += 1
        generation = self.generation
        self.signal = Signal(self.enabled)
        self.job_id = uuid4().hex[:12]
        self.phase, self.detail = "reviewing", "Starting a bounded harness review."
        self._automatic_job = automatic
        review_first = self.automatic != "apply" if automatic else self.review_first
        signal = self.signal

        def progress(text):
            if generation == self.generation:
                self._progress(text)

        async def run():
            try:
                candidate = await asyncio.wait_for(
                    review(
                        provider,
                        model,
                        baseline,
                        evidence,
                        focus,
                        signal,
                        progress,
                    ),
                    timeout=300,
                )
                if generation == self.generation and not self.signal.is_cancelled():
                    self.draft = candidate
                    if review_first and candidate.changed:
                        self.phase, self.detail = "awaiting review", (
                            "/refine show lists the proposed changes; /refine apply saves "
                            "them; /refine cancel drops them.")
                        return
                    self.phase, self.detail = "ready", "Validated proposal awaits a turn boundary."
                    if not self.session.harness.active:
                        self.at_boundary()
            except asyncio.CancelledError:
                if generation == self.generation:
                    self.phase, self.detail = "cancelled", "No staged changes were committed."
            except Exception as error:  # provider failures are visible through status
                if generation == self.generation:
                    self.phase, self.detail = "failed", str(error)[:1000]

        self.task = asyncio.create_task(run(), name=f"harness-review-{self.job_id}")
        return self.status()

    def _progress(self, text):
        self.detail = text

    def cancel(self):
        self.generation += 1
        self.signal.cancelled = True
        self.draft = None
        self._pending_evidence = None
        if self.task is not None and not self.task.done():
            self.task.cancel()
        if self.phase in BUSY_PHASES:
            self.phase, self.detail = "cancelled", "Discarded the pending review."

    async def close(self):
        self.cancel()
        if self.task is not None:
            with suppress(asyncio.CancelledError):
                await self.task

    def at_boundary(self):
        if not self.enabled():
            self.cancel()
            self._apply_active()
            return
        if self.session.harness.active:
            return
        if self.phase == "ready":
            if not self.writable():
                self.cancel()
                return
            candidate = self.draft
            self.draft = None
            self.phase = "applying"
            try:
                if candidate.changed:
                    self.active = self.store.commit(candidate.baseline, candidate.state)
                    note = self.active.history[-1]
                    evidence = getattr(self.session, "evidence", None)
                    if evidence is not None:
                        evidence.saved(note.id[:12], "refine",
                                       sorted({edit["id"] for edit in note.edits}), note.summary)
                else:
                    self.active = self.store.read()
                self.phase = "completed"
                self.detail = (
                    f"Active harness revision {self.active.revision}. "
                    "Expected benefits have not been benchmarked."
                )
            except Exception as error:
                self.phase, self.detail = "failed", str(error)[:1000]
        # Reload only at a boundary, including changes committed by other sessions.
        try:
            latest = self.store.read()
            if latest != self.active:
                self.active = latest
            self._apply_active()
        except Exception as error:
            self.phase, self.detail = "failed", str(error)[:1000]

    def render(self, base: str, *, ignore_gate=False) -> str:
        if not ignore_gate and not self.enabled():
            return base
        parts = []
        for entry in self.active.entries.values():
            if entry.kind == "skill":
                description, _ = skill_parts(entry.id, entry.content)
                parts.append(f"Workflow {entry.id}: {description}")
            else:
                parts.append(f"{entry.kind} {entry.id} ({entry.title}):\n{_digest(entry)}")
        if not parts:
            return base
        return base + (
            "\n\nSaved working guidance (supplements the base instructions):\n"
            + "\n\n".join(parts)
            + "\nRead a selected workflow with await guidance(id), or invoke "
            "/skill:<id>. Read its full procedure before using it."
        )

    def filter_system(self, text):
        if self.enabled():
            return text
        supplemental = self.render("", ignore_gate=True)
        return text.replace(supplemental, "") if supplemental else text

    def read_guidance(self, entry_id=None):
        if not self.enabled():
            raise ValueError("Harness refinements are disabled")
        if entry_id is not None:
            return self.active.entries[entry_id].model_dump()
        return [
            {"id": entry.id, "kind": entry.kind, "title": entry.title}
            for entry in self.active.entries.values()
        ]

    def _apply_active(self):
        self.session._system = self.render(self.base_system)
        if self.base_backend is not None:
            self.session.backend.instructions = self.render(self.base_backend)
        self.session._apply_plan_mode()
        skills = {skill.name: skill for skill in self.installed}
        for entry in self.active.entries.values() if self.enabled() else ():
            if entry.kind == "skill":
                description, body = skill_parts(entry.id, entry.content)
                skills[entry.id] = Skill(
                    entry.id, description, body, self.store.path or Path("<ephemeral-harness>")
                )
        self.session._skills = list(skills.values())

    def transfer(self, session):
        """Reattach guidance when the frontend replaces its session object."""
        session._system = self.base_system
        if hasattr(self, "home"):
            return attach(session, self.home, persist=self.store.path is not None)
        store = self.store if session.cwd == self.session.cwd else HarnessStore(None)
        session.refinement = Refinement(session, store)
        return session.refinement


def proposal_diff(before, after) -> str:
    """Each entry a proposal adds, changes or removes, with a unified diff of its text."""
    import difflib

    parts = []
    for entry_id in sorted(set(before.entries) | set(after.entries)):
        old, new = before.entries.get(entry_id), after.entries.get(entry_id)
        if old == new:
            continue
        verb = "add" if old is None else "remove" if new is None else "change"
        shown = new or old
        parts.append(f"{verb} {shown.kind} {entry_id}: {shown.title}")
        lines = difflib.unified_diff(
            (old.content if old else "").splitlines(), (new.content if new else "").splitlines(),
            fromfile=f"a/{entry_id}", tofile=f"b/{entry_id}", lineterm="")
        parts.append("\n".join(lines))
    return "\n".join(parts) or "The proposal changes nothing."


def _digest(entry) -> str:
    """The entry as it goes in every prompt: short ones whole, long ones the start of them."""
    if len(entry.content) <= DIGEST_CHARS:
        return entry.content
    head = entry.content[:DIGEST_CHARS].rstrip()
    return f'{head}... (full text: await guidance("{entry.id}"))'


def project_folder(home: Path, cwd: str) -> Path:
    project = hashlib.sha256(str(Path(cwd).resolve()).encode()).hexdigest()[:24]
    return home / "harness" / project


def attach(session, home: Path, *, persist: bool):
    path = project_folder(home, session.cwd) / "state.json" if persist else None
    controller = Refinement(session, HarnessStore(path))
    controller.home = home
    controller.project_store = controller.store
    controller.scope = "project"
    if persist:
        from rlmagent.refinement.scoped import ScopedStore

        folder = project_folder(home, session.cwd) / "sessions" / session.session_id
        try:
            selected = json.loads((folder / "scope.json").read_text(encoding="utf-8"))["scope"]
        except FileNotFoundError:
            selected = "project"
        if selected == "session":
            controller.store = ScopedStore(folder / "state.json", session, controller.active)
            controller.active = controller.store.read()
            controller.scope = "session"
    controller._apply_active()
    session.refinement = controller
    tree = getattr(session, "agent_tree", None)
    if tree is not None:
        tree.refinement = controller
        if hasattr(tree, "system_for"):
            original = tree.system_for
            tree.system_for = lambda node, tools: controller.render(original(node, tools))
    return controller
