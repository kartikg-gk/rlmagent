"""Single-job refinement with activation at session boundaries."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from rlmagent.refinement.engine import evidence_text, review
from rlmagent.refinement.store import HarnessStore, skill_parts
from rlmagent.skillset import Skill


class Signal:
    def __init__(self):
        self.cancelled = False

    def is_cancelled(self):
        return self.cancelled


class Refinement:
    def __init__(self, session, store: HarnessStore):
        self.session, self.store = session, store
        self.active = store.read()
        self.base_system = session._system
        self.base_backend = getattr(getattr(session, "backend", None), "instructions", None)
        self.installed = list(session._skills)
        self.task = None
        self.signal = Signal()
        self.generation = 0
        self.phase = "idle"
        self.detail = "No manual review has been requested."
        self.job_id = None
        self.draft = None
        self._apply_active()

    def writable(self):
        if self.session.plan_mode:
            return False
        improver = getattr(self.session, "improver", None)
        return getattr(improver, "writable", True)

    def status(self):
        identity = f" {self.job_id}" if self.job_id else ""
        return f"Refinement{identity}: {self.phase}. {self.detail}"

    async def command(self, argument):
        if argument.strip() == "status":
            return self.status()
        if argument.strip() == "cancel":
            self.cancel()
            return self.status()
        if not self.writable():
            return "Refinement is unavailable while guidance is read-only."
        if self.phase in ("reviewing", "ready", "applying"):
            return self.status() + " Finish or cancel this review before starting another."
        baseline = self.store.read()
        evidence = evidence_text(list(self.session.transcript))
        provider, model = self.session._provider, self.session.model
        self.generation += 1
        generation = self.generation
        self.signal = Signal()
        self.job_id = uuid4().hex[:12]
        self.phase, self.detail = "reviewing", "Starting a bounded harness review."
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
                        argument.strip() or None,
                        signal,
                        progress,
                    ),
                    timeout=300,
                )
                if generation == self.generation and not self.signal.is_cancelled():
                    self.draft = candidate
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
        if self.task is not None and not self.task.done():
            self.task.cancel()
        if self.phase in ("reviewing", "ready", "applying"):
            self.phase, self.detail = "cancelled", "Discarded the pending review."

    async def close(self):
        self.cancel()
        if self.task is not None:
            with suppress(asyncio.CancelledError):
                await self.task

    def at_boundary(self):
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

    def render(self, base: str) -> str:
        parts = []
        for entry in self.active.entries.values():
            if entry.kind == "skill":
                description, _ = skill_parts(entry.id, entry.content)
                parts.append(f"Workflow {entry.id}: {description}")
            else:
                parts.append(f"{entry.kind} {entry.id} ({entry.title}):\n{entry.content}")
        if not parts:
            return base
        return base + (
            "\n\nSaved working guidance (supplements the base instructions):\n"
            + "\n\n".join(parts)
            + "\nRead a selected workflow with await guidance(id), or invoke "
            "/skill:<id>. Read its full procedure before using it."
        )

    def read_guidance(self, entry_id=None):
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
        for entry in self.active.entries.values():
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


def attach(session, home: Path, *, persist: bool):
    project = hashlib.sha256(str(Path(session.cwd).resolve()).encode()).hexdigest()[:24]
    path = home / "harness" / project / "state.json" if persist else None
    controller = Refinement(session, HarnessStore(path))
    controller.home = home
    session.refinement = controller
    tree = getattr(session, "agent_tree", None)
    if tree is not None:
        tree.refinement = controller
        if hasattr(tree, "system_for"):
            original = tree.system_for
            tree.system_for = lambda node, tools: controller.render(original(node, tools))
    return controller
