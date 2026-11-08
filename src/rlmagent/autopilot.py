"""A goal the session keeps working toward, and turns that continue without the user.

/goal sets an objective. It is put in front of the model on every turn until the model
calls goal_done(summary) or the user clears it. /autonomous lets the session start its
own next turn while a goal is open, within turn, token and time limits. All
configured verification commands must pass before a goal counts as done.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

GATE_SECONDS = 600
GATE_OUTPUT = 4_000
STALL_TURNS = 3


def _only_reports_result(code: str) -> bool:
    try:
        call = ast.parse(code.strip(), mode="eval").body
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "FINAL" and len(call.args) == 1 and not call.keywords):
            return False
        ast.literal_eval(call.args[0])
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


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
    gates: dict[str, str] = field(default_factory=dict)


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
    _last_turn: bytes | None = field(default=None, init=False, repr=False)
    _unchanged_turns: int = field(default=0, init=False, repr=False)
    _failed_turns: list[str] = field(default_factory=list, init=False, repr=False)
    durable: bool = False
    _in_flight: bool = field(default=False, init=False, repr=False)
    _pending_prompt: str | None = field(default=None, init=False, repr=False)
    gate_results: dict[str, dict] = field(default_factory=dict)
    _verifying: bool = field(default=False, init=False, repr=False)

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.goal = Goal(**data.get("goal", {}))
            self.limits = Limits(**data.get("limits", {}))
            self.gate_results = data.get("gate_results", {})
            for result in self.gate_results.values():
                if result["status"] == "running":
                    result["status"] = "interrupted"
            if self.durable and (state := data.get("continuation")):
                self.turns = state["turns"]
                self.tokens_at_start = state["tokens_at_start"]
                self.started = state["started"]
                self.autonomous = state["active"]
                self._last_turn = bytes.fromhex(state["last_turn"]) if state["last_turn"] else None
                self._unchanged_turns = state["unchanged_turns"]
                self._failed_turns = state.get("failed_turns", [])[-2 * STALL_TURNS:]
                self.stopped_because = state["stopped_because"]
                self._pending_prompt = state.get("pending_prompt")
                if state["in_flight"]:
                    self.stop_autonomous("a worker turn was interrupted; review it before continuing")
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"goal": asdict(self.goal), "limits": asdict(self.limits)}
            data["gate_results"] = self.gate_results
            if self.durable:
                data["continuation"] = {
                    "turns": self.turns, "tokens_at_start": self.tokens_at_start,
                    "started": self.started, "active": self.autonomous,
                    "in_flight": self._in_flight, "stopped_because": self.stopped_because,
                    "last_turn": self._last_turn.hex() if self._last_turn else None,
                    "unchanged_turns": self._unchanged_turns,
                    "failed_turns": self._failed_turns,
                    "pending_prompt": self._pending_prompt,
                }
            staged = self.path.with_suffix(f".{uuid4().hex}.tmp")
            try:
                staged.write_text(json.dumps(data), encoding="utf-8")
                staged.replace(self.path)
            finally:
                staged.unlink(missing_ok=True)

    def _tokens(self) -> int:
        return int(self.used_tokens()) if self.used_tokens is not None else 0

    # -- the goal ----------------------------------------------------------

    def set_goal(self, objective: str, token_budget: int | None = None) -> str:
        self.goal = Goal(objective.strip(), "active", "", token_budget, self._tokens(), time.time())
        self.gate_results.clear()
        self._last_turn, self._unchanged_turns = None, 0
        self._failed_turns.clear()
        self._pending_prompt = None
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
                     + (f", gates: {', '.join(self.gates())}" if self.gates() else ""))
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
                "token_budget": self.goal.token_budget,
                "gates": {name: dict(result) for name, result in self.gate_results.items()}}

    async def goal_done(self, summary: str) -> str:
        """Mark the goal achieved, with a short summary of what was done and how it was checked."""
        if self.goal.status != "active":
            return "There is no active goal."
        if self._verifying:
            return "Goal verification is already running."
        goal, gates = self.goal, self.gates()
        failures = []
        self._verifying = True
        try:
            for name, command in gates.items():
                result = self.gate_results.setdefault(name, {"attempts": 0})
                result.update(status="running", attempts=result["attempts"] + 1)
                self.save()
                try:
                    failure = await self.run_gate(command)
                except asyncio.CancelledError:
                    result["status"] = "interrupted"
                    self.save()
                    raise
                result["status"] = "failed" if failure else "passed"
                self.save()
                if failure:
                    failures.append(f"{name}: {failure}")
                if self.goal is not goal or goal.status != "active" or self.gates() != gates:
                    return "Goal or verification settings changed during verification."
        finally:
            self._verifying = False
        if failures:
            return "The gate command failed, so the goal is still open:\n" + "\n".join(failures)
        goal.summary = str(summary)[:2000]
        goal.status = "done"
        self.save()
        return "Goal marked done."

    def gates(self) -> dict[str, str]:
        return {**({"default": self.limits.gate} if self.limits.gate else {}), **self.limits.gates}

    async def run_gate(self, command: str) -> str | None:
        from rlmagent.agents.output import OutputTail
        from rlmagent.model.external.cli import kill_tree, own_group

        tail = OutputTail()
        options = own_group()
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            argv = shlex.split(command, posix=True)
            if not argv:
                return "empty command"
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=self.cwd, stdin=subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=subprocess.STDOUT, **options,
            )
        except (OSError, ValueError) as exc:
            return str(exc)

        async def drain():
            while chunk := await process.stdout.read(8192):
                tail.feed(chunk)
            await process.wait()

        try:
            await asyncio.wait_for(drain(), GATE_SECONDS)
        except (TimeoutError, asyncio.CancelledError) as exc:
            kill_tree(process)
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return f"timed out after {GATE_SECONDS}s\n{tail.text()[-GATE_OUTPUT:]}"
        return None if process.returncode == 0 else f"exited {process.returncode}\n{tail.text()[-GATE_OUTPUT:]}"

    # -- autonomous turns --------------------------------------------------

    def start_autonomous(self) -> str:
        if self.goal.status != "active":
            return "Set a goal first with /goal TEXT; autonomous mode works toward it."
        self.autonomous, self.turns, self.stopped_because = True, 0, ""
        self.tokens_at_start, self.started = self._tokens(), time.time()
        self._last_turn, self._unchanged_turns = None, 0
        self._failed_turns.clear()
        self._pending_prompt = None
        if self.durable:
            self.save()
        return self.describe()

    def stop_autonomous(self, why: str) -> None:
        self.autonomous, self.stopped_because = False, why
        self._pending_prompt = None
        if self.durable:
            self.save()

    def start_worker_turn(self, *, continuation=False) -> None:
        if continuation:
            self._pending_prompt = None
        self._in_flight = True
        self.save()

    def finish_worker_turn(self) -> None:
        self._in_flight = False
        self.save()

    def observe_turn(self, messages, backend_events=()) -> None:
        """Stop on repeated results or a cycle of the same failed operations."""
        if not self.autonomous or self.goal.status != "active":
            return
        from rlmagent.harness.contracts.transcript import ModelEntry, ToolOutcomeEntry

        reply = next((entry for entry in reversed(messages) if isinstance(entry, ModelEntry)), None)
        if reply is not None and reply.stop_reason in ("error", "aborted"):
            self.stop_autonomous("the last turn failed")
            return
        calls = {call.id: call.arguments for entry in messages if isinstance(entry, ModelEntry)
                 for call in entry.tool_calls}
        operations = [
            (entry.tool_name, calls.get(entry.tool_call_id), entry.is_error, " ".join(entry.text.split()))
            for entry in messages if isinstance(entry, ToolOutcomeEntry)
        ]
        cells = {event.get("step"): event.get("code") for event in backend_events
                 if event.get("kind") == "code_generated" and event.get("run_id") == "root"}
        for event in backend_events:
            if (event.get("kind") != "output_received" or event.get("run_id") != "root"
                    or event.get("step", 0) <= 0 or not (code := cells.get(event.get("step")))):
                continue
            output, failed = " ".join(str(event.get("output", "")).split()), bool(event.get("error"))
            if failed or output or not _only_reports_result(code):
                operations.append(("cell", code, failed, output))
        if operations and all(operation[2] for operation in operations):
            fingerprint = hashlib.sha256(json.dumps(operations, sort_keys=True).encode()).hexdigest()
            self._failed_turns.append(fingerprint)
            self._failed_turns = self._failed_turns[-2 * STALL_TURNS:]
            for period in (1, 2):
                pattern = self._failed_turns[-period:]
                if len(pattern) == period and self._failed_turns[-period * STALL_TURNS:] == pattern * STALL_TURNS:
                    self.stop_autonomous("the same failed operations repeated without progress")
                    return
        else:
            self._failed_turns.clear()
        answer = " ".join(reply.text.split()) if reply is not None else ""
        digest = hashlib.sha256(json.dumps([answer, operations], sort_keys=True).encode()).digest()
        self._unchanged_turns = self._unchanged_turns + 1 if digest == self._last_turn else 1
        self._last_turn = digest
        if self._unchanged_turns >= STALL_TURNS:
            self.stop_autonomous(f"no visible progress across {STALL_TURNS} turns")

    def next_prompt(self) -> str | None:
        """After a turn: the text of the next turn to start on its own, or None to stop."""
        if not self.autonomous:
            return None
        goal, limits = self.goal, self.limits
        used = self._tokens() - self.tokens_at_start
        if goal.status != "active":
            self.stop_autonomous("goal is " + goal.status)
        elif self._pending_prompt is None and self.turns >= limits.max_turns:
            self.stop_autonomous(f"reached {limits.max_turns} turns")
        elif limits.max_tokens and used >= limits.max_tokens:
            self.stop_autonomous(f"used {used:,} tokens")
        elif goal.token_budget and self._tokens() - goal.tokens_at_start >= goal.token_budget:
            self.stop_autonomous("goal token budget spent")
        elif limits.max_seconds and time.time() - self.started >= limits.max_seconds:
            self.stop_autonomous(f"ran {limits.max_seconds:.0f}s")
        if not self.autonomous:
            return None
        if self._pending_prompt is not None:
            return self._pending_prompt
        self.turns += 1
        prompt = ("Continue toward the goal. Check what is left, do the next piece, and verify it. "
                  "Call goal_done(summary) only when everything is finished.")
        if self.durable:
            self._pending_prompt = prompt
            self.save()
        return prompt

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
        if not words or words == ["on"]:
            return self.start_autonomous()
        if words[0] == "off":
            self.stop_autonomous("turned off")
            return "Autonomous mode off."
        if words[0] == "status":
            state = "on" if self.autonomous else f"off{' (' + self.stopped_because + ')' if self.stopped_because else ''}"
            return (f"Autonomous mode {state}. Limits: {self.limits.max_turns} turns, "
                    f"{self.limits.max_tokens or 'no'} tokens, {self.limits.max_seconds or 'no'} "
                    f"seconds, gates: {', '.join(self.gates()) or 'none'}.")
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
            _, *command_parts = arg.strip().split(maxsplit=1)
            command = command_parts[0] if command_parts else ""
            action, *arguments = command.split(maxsplit=1) or [""]
            if action == "list":
                return "\n".join(
                    f"{name}: {cmd} ({self.gate_results.get(name, {}).get('status', 'pending')})"
                    for name, cmd in self.gates().items()
                ) or "No verification gates."
            if self._verifying:
                return "Wait for the active verification to finish."
            if action in ("add", "remove"):
                name, *gate_parts = (arguments[0].split(maxsplit=1) if arguments else [""])
                gate = gate_parts[0] if gate_parts else ""
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", name) or name == "default":
                    return "Use a gate name of 1 to 40 letters, numbers, hyphens, or underscores."
                if action == "add":
                    if not gate.strip():
                        return "Use /autonomous gate add NAME COMMAND."
                    self.limits.gates[name] = gate.strip()
                else:
                    self.limits.gates.pop(name, None)
                self.gate_results.pop(name, None)
            else:
                self.limits.gate = command or None
                self.gate_results.pop("default", None)
            self.save()
            return self.command_autonomous("status")
        return ("Usage: /autonomous [on | off | status | turns N | tokens N | seconds N | "
                "gate CMD | gate list | gate add NAME CMD | gate remove NAME]")

    def functions(self) -> dict:
        return {"goal": self.goal_status, "goal_done": self.goal_done}


__all__ = ["Autopilot", "Goal", "Limits"]
