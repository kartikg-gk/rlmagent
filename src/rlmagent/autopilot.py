"""A goal the session keeps working toward, and turns that continue without the user.

/goal sets an objective. It is put in front of the model on every turn until the model
calls goal_done(summary) or the user clears it. /autonomous lets the session start its
own next turn while a goal is open, within turn, token and time limits; a gate command,
when set, must pass before a goal counts as done.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

GATE_SECONDS = 600
GATE_OUTPUT = 4_000


@dataclass
class Goal:
    objective: str = ""
    status: str = "none"  # none, active, paused, done
    summary: str = ""
    token_budget: int | None = None
    tokens_at_start: int = 0
    started: float = 0.0


@dataclass
class Limits:
    max_turns: int = 10
    max_tokens: int | None = 500_000
    max_seconds: float | None = 3600.0
    gate: str | None = None


@dataclass
class Autopilot:
    path: Path | None = None
    goal: Goal = field(default_factory=Goal)
    limits: Limits = field(default_factory=Limits)
    autonomous: bool = False
    turns: int = 0
    tokens_at_start: int = 0
    started: float = 0.0
    stopped_because: str = ""
    cwd: str = "."
    used_tokens: object = None  # () -> int, tokens the session has used so far

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.goal = Goal(**data.get("goal", {}))
            self.limits = Limits(**data.get("limits", {}))
        except (OSError, ValueError, TypeError):
            pass

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"goal": asdict(self.goal),
                                             "limits": asdict(self.limits)}), encoding="utf-8")

    def _tokens(self) -> int:
        return int(self.used_tokens()) if self.used_tokens is not None else 0

    # -- the goal ----------------------------------------------------------

    def set_goal(self, objective: str, token_budget: int | None = None) -> str:
        self.goal = Goal(objective.strip(), "active", "", token_budget, self._tokens(), time.time())
        self.save()
        return self.describe()

    def describe(self) -> str:
        goal = self.goal
        if goal.status == "none":
            return "No goal. Set one with /goal TEXT [--tokens N]."
        used = self._tokens() - goal.tokens_at_start
        budget = f" of {goal.token_budget:,}" if goal.token_budget else ""
        text = f"Goal ({goal.status}): {goal.objective}\nTokens used: {used:,}{budget}"
        if goal.summary:
            text += f"\nSummary: {goal.summary}"
        if self.autonomous:
            text += (f"\nAutonomous: turn {self.turns} of {self.limits.max_turns}"
                     + (f", gate: {self.limits.gate}" if self.limits.gate else ""))
        return text

    def reminder(self) -> str | None:
        """The note sent with each turn while a goal is open."""
        if self.goal.status != "active":
            return None
        return (f"[Goal] Keep working toward: {self.goal.objective}\n"
                "When it is fully achieved, call goal_done(summary) (or tell the user it is "
                "done). Do not call it while anything is left to do.")

    async def goal_status(self) -> dict:
        """The current goal: objective, status, summary and tokens used."""
        return {"objective": self.goal.objective, "status": self.goal.status,
                "summary": self.goal.summary,
                "tokens_used": self._tokens() - self.goal.tokens_at_start,
                "token_budget": self.goal.token_budget}

    async def goal_done(self, summary: str) -> str:
        """Mark the goal achieved, with a short summary of what was done and how it was checked."""
        if self.goal.status != "active":
            return "There is no active goal."
        self.goal.summary = str(summary)[:2000]
        failure = self.run_gate()
        if failure:
            self.goal.summary = ""
            return f"The gate command failed, so the goal is still open:\n{failure}"
        self.goal.status = "done"
        self.save()
        return "Goal marked done."

    def run_gate(self) -> str | None:
        """None when there is no gate or it passes; otherwise what it printed."""
        if not self.limits.gate:
            return None
        try:
            out = subprocess.run(shlex.split(self.limits.gate, posix=True), cwd=self.cwd,
                                 capture_output=True, text=True, timeout=GATE_SECONDS,
                                 stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"{self.limits.gate}: {exc}"
        if out.returncode == 0:
            return None
        text = (out.stdout + out.stderr)[-GATE_OUTPUT:]
        return f"{self.limits.gate} exited {out.returncode}\n{text}"

    # -- autonomous turns --------------------------------------------------

    def start_autonomous(self) -> str:
        if self.goal.status != "active":
            return "Set a goal first with /goal TEXT; autonomous mode works toward it."
        self.autonomous, self.turns, self.stopped_because = True, 0, ""
        self.tokens_at_start, self.started = self._tokens(), time.time()
        return self.describe()

    def stop_autonomous(self, why: str) -> None:
        self.autonomous, self.stopped_because = False, why

    def next_prompt(self) -> str | None:
        """After a turn: the text of the next turn to start on its own, or None to stop."""
        if not self.autonomous:
            return None
        goal, limits = self.goal, self.limits
        used = self._tokens() - self.tokens_at_start
        if goal.status != "active":
            self.stop_autonomous("goal is " + goal.status)
        elif self.turns >= limits.max_turns:
            self.stop_autonomous(f"reached {limits.max_turns} turns")
        elif limits.max_tokens and used >= limits.max_tokens:
            self.stop_autonomous(f"used {used:,} tokens")
        elif goal.token_budget and self._tokens() - goal.tokens_at_start >= goal.token_budget:
            self.stop_autonomous("goal token budget spent")
        elif limits.max_seconds and time.time() - self.started >= limits.max_seconds:
            self.stop_autonomous(f"ran {limits.max_seconds:.0f}s")
        if not self.autonomous:
            return None
        self.turns += 1
        return ("Continue toward the goal. Check what is left, do the next piece, and verify it. "
                "Call goal_done(summary) only when everything is finished.")

    def command_goal(self, arg: str) -> str:
        words = arg.split()
        if not words:
            return self.describe()
        if words[0] in ("pause", "resume", "clear"):
            if self.goal.status == "none":
                return "No goal."
            if words[0] == "clear":
                self.goal = Goal()
                self.stop_autonomous("goal cleared")
            else:
                self.goal.status = "paused" if words[0] == "pause" else "active"
            self.save()
            return self.describe()
        budget = None
        if "--tokens" in words:
            at = words.index("--tokens")
            try:
                budget = int(words[at + 1])
            except (IndexError, ValueError):
                return "Usage: /goal TEXT [--tokens N] | pause | resume | clear"
            words = words[:at] + words[at + 2:]
        return self.set_goal(" ".join(words), budget)

    def command_autonomous(self, arg: str) -> str:
        words = arg.split()
        if not words:
            return self.start_autonomous()
        if words[0] == "off":
            self.stop_autonomous("turned off")
            return "Autonomous mode off."
        if words[0] == "status":
            state = "on" if self.autonomous else f"off{' (' + self.stopped_because + ')' if self.stopped_because else ''}"
            return (f"Autonomous mode {state}. Limits: {self.limits.max_turns} turns, "
                    f"{self.limits.max_tokens or 'no'} tokens, {self.limits.max_seconds or 'no'} "
                    f"seconds, gate: {self.limits.gate or 'none'}.")
        if words[0] in ("turns", "tokens", "seconds") and len(words) == 2:
            try:
                value = float(words[1])
            except ValueError:
                return "Give a number."
            if words[0] == "turns":
                self.limits.max_turns = int(value)
            elif words[0] == "tokens":
                self.limits.max_tokens = int(value) or None
            else:
                self.limits.max_seconds = value or None
            self.save()
            return self.command_autonomous("status")
        if words[0] == "gate":
            self.limits.gate = " ".join(words[1:]) or None
            self.save()
            return self.command_autonomous("status")
        return "Usage: /autonomous [off | status | turns N | tokens N | seconds N | gate CMD]"

    def functions(self) -> dict:
        return {"goal": self.goal_status, "goal_done": self.goal_done}


__all__ = ["Autopilot", "Goal", "Limits"]
