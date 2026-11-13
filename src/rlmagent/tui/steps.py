"""Step cards: one block per agent step, built from the engine's run events."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.markup import escape
from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Collapsible, Static

from rlmagent.tui.messages import highlighted_code

SHOWN_OUTPUT_LINES = 6
MAX_OUTPUT_CHARS = 20_000

GENERATING, RUNNING, DONE, FAILED = "generating", "running", "done", "failed"


@dataclass
class Step:
    """What a card shows for one step of one agent."""

    agent: str
    number: int
    depth: int = 0
    model: str = ""
    state: str = GENERATING
    code: str = ""
    output: str = ""
    tokens: int = 0
    cost: float | None = None
    started: float | None = None
    ended: float | None = None
    #: The kernel's variables after this step, when the engine reported them.
    variables: list = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int]:
        return (self.agent, self.number)

    @property
    def short(self) -> str:
        return short_id(self.agent)

    @property
    def seconds(self) -> float | None:
        if self.started is None or self.ended is None:
            return None
        return max(0.0, self.ended - self.started)


def short_id(run_id: str) -> str:
    """Six characters that tell agents apart; the root keeps its name."""
    if run_id == "root" or len(run_id) <= 6:
        return run_id
    return run_id[-6:]


@dataclass
class Agent:
    run_id: str
    parent: str | None = None
    depth: int = 0
    model: str = ""
    status: str = RUNNING
    steps: list[int] = field(default_factory=list)
    child_id: str | None = None


def _moment(value) -> float | None:
    """Seconds since the epoch, from a number or the ISO text the engine records."""
    if value is None or isinstance(value, (int, float)):
        return value
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


def _usage(value) -> tuple[int, float | None]:
    if value is None:
        return 0, None
    if isinstance(value, dict):
        tokens = value.get("total_tokens") or (
            (value.get("prompt_tokens") or 0) + (value.get("completion_tokens") or 0))
        return int(tokens or 0), value.get("cost")
    tokens = getattr(value, "total_tokens", 0) or (
        (getattr(value, "prompt_tokens", 0) or 0) + (getattr(value, "completion_tokens", 0) or 0))
    return int(tokens or 0), getattr(value, "cost", None)


class StepBook:
    """Folds engine events into steps and agents. Holds no widgets."""

    def __init__(self) -> None:
        #: Bumped on every change, so a view redraws only when something moved.
        self.version = 0
        self.steps: dict[tuple[str, int], Step] = {}
        self.agents: dict[str, Agent] = {}
        self.order: list[tuple[str, int]] = []
        self.child_names: dict[str, str] = {}
        self.child_notes: dict[str, str] = {}

    def clear(self) -> None:
        self.version += 1
        self.steps.clear()
        self.agents.clear()
        self.order.clear()
        self.child_names.clear()
        self.child_notes.clear()

    def feed(self, event: dict) -> Step | None:
        """Apply one event; return the step it changed, if it changed one worth showing."""
        self.version += 1
        kind = event.get("kind")
        run_id = event.get("run_id") or "root"
        child_id = event.get("child_id")
        if kind == "run_started":
            agent = self.agents.setdefault(run_id, Agent(run_id))
            agent.parent = event.get("parent_run_id")
            agent.depth = int(event.get("depth") or 0)
            agent.model = str(event.get("model") or "")
            agent.status = RUNNING
            if child_id:
                agent.child_id = child_id
            return None
        if kind == "child_bound" and child_id:
            self.agents.setdefault(run_id, Agent(run_id)).child_id = child_id
        if kind in ("child_bound", "child_named") and child_id and event.get("name"):
            self.child_names[child_id] = event["name"]
        if kind == "child_progress" and child_id:
            self.child_notes[child_id] = " ".join(str(event.get("text", ""))[:500].splitlines())
            return None
        if kind == "child_status":
            if run_id in self.agents:
                self.agents[run_id].status = event["status"]
            return None
        if kind in ("run_completed", "run_failed"):
            agent = self.agents.get(run_id)
            if agent is not None:
                agent.status = DONE if kind == "run_completed" else FAILED
            return None
        number = event.get("step")
        if not isinstance(number, int) or number <= 0:
            return None  # step 0 only loads the context
        if kind == "namespace_changed":
            step = self.steps.get((run_id, number))
            if step is not None:
                step.variables = [row for row in event.get("variables") or [] if isinstance(row, dict)]
            return None
        step = self._step(run_id, number)
        if kind == "step_started":
            step.state = GENERATING
            step.started = _moment(event.get("started") or event.get("timestamp"))
        elif kind == "code_generated":
            step.code = str(event.get("code") or "")
            step.state = RUNNING if step.code else FAILED
        elif kind == "output_received":
            step.output = str(event.get("output") or "")[-MAX_OUTPUT_CHARS:]
            if event.get("error"):
                step.state = FAILED
        elif kind == "step_completed":
            step.tokens, step.cost = _usage(event.get("usage"))
            step.state = FAILED if event.get("error") else DONE
            step.ended = _moment(event.get("ended") or event.get("timestamp"))
        else:
            return None
        return step

    def _step(self, run_id: str, number: int) -> Step:
        key = (run_id, number)
        step = self.steps.get(key)
        if step is None:
            agent = self.agents.get(run_id) or Agent(run_id)
            self.agents.setdefault(run_id, agent)
            step = Step(run_id, number, agent.depth, agent.model)
            self.steps[key] = step
            self.order.append(key)
            agent.steps.append(number)
        return step

    def totals(self) -> dict:
        steps = self.steps.values()
        costs = [s.cost for s in steps if s.cost is not None]
        return {
            "steps": len(self.steps),
            "tokens": sum(s.tokens for s in steps),
            "cost": sum(costs) if costs else None,
            "agents": len(self.agents),
            "running": sum(1 for a in self.agents.values() if a.status == RUNNING),
            "depth": max((a.depth for a in self.agents.values()), default=0),
        }


def header_text(step: Step) -> str:
    """The card's title line, e.g. ``RLM d0 · root · step 1 · running``."""
    return f"RLM d{step.depth} · {step.short} · step {step.number} · {step.state}"


def footer_text(step: Step) -> str:
    """Tokens, cost and time under a finished step."""
    parts = [f"{step.tokens / 1000:.1f}k tokens" if step.tokens >= 1000 else f"{step.tokens} tokens"]
    parts.append("cost unknown" if step.cost is None else f"${step.cost:.4f}")
    if step.seconds is not None:
        parts.append(f"{step.seconds:.1f}s")
    return " · ".join(parts)


# Greys, the violet accent for work in progress, red only for a failure.
LINE_MARKS = {GENERATING: ("●", "#c9a4ff"), RUNNING: ("●", "#c9a4ff"),
              DONE: ("✓", "#7a7a7a"), FAILED: ("✗", "#a05050")}


def _lines(text: str) -> int:
    return len(text.rstrip("\n").splitlines()) if text.strip() else 0


def line_text(step: Step, now: float | None = None, *, open_: bool = False) -> Text:
    """One step as one row: state, the code's first line, size in and out, time.

    A step still going shows how long it has been at it instead of a size.
    """
    import time

    mark, colour = LINE_MARKS.get(step.state, ("·", "#7d8590"))
    first = next((line.strip() for line in step.code.splitlines() if line.strip()), "")
    row = Text()
    row.append("  " * step.depth)
    row.append(f"{mark} ", style=colour)
    # The name in bold, everything else quiet: a sub-agent's id, or "python".
    row.append(step.short if step.depth else "python", style="bold #c8c8c8")
    row.append("  ")
    if first:
        row.append(first if len(first) <= 70 else first[:69] + "…", style="#7a7a7a")
        row.append("  ")
    if step.state in (DONE, FAILED):
        detail = f"↑{_lines(step.code)} ↓{_lines(step.output)} lines"
        if step.seconds is not None:
            detail += f" · {step.seconds:.1f}s"
        row.append(detail, style="#4a4a4a")
    else:
        started = step.started or (now or time.time())
        elapsed = max(0.0, (now or time.time()) - started)
        word = "writing code" if step.state == GENERATING else "running"
        row.append(f"{word} {elapsed:.0f}s", style="#7a7a7a")
    return row


def output_preview(output: str) -> tuple[str, int]:
    """The first lines of output and how many were left out."""
    lines = output.rstrip("\n").splitlines()
    hidden = max(0, len(lines) - SHOWN_OUTPUT_LINES)
    return "\n".join(lines[:SHOWN_OUTPUT_LINES]), hidden


class StepCard(Vertical, can_focus=True):
    """One step on screen: title, code, output preview, usage. Enter or a click opens it."""

    BINDINGS = [("enter", "inspect", "Open step"), ("space", "toggle_details", "Toggle details")]

    class Inspect(Message):
        def __init__(self, step: Step) -> None:
            super().__init__()
            self.step = step

    def action_inspect(self) -> None:
        self.post_message(self.Inspect(self.step))

    def action_toggle_details(self) -> None:
        self.set_open(self._collapsed)

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def set_open(self, opened: bool) -> None:
        """Show the code and output under the row, or fold back to the row."""
        self._collapsed = not opened
        self.refresh_step(self.step)

    def on_click(self) -> None:
        self.action_inspect()

    def __init__(self, step: Step) -> None:
        super().__init__(classes="step-card")
        self.step = step
        # One row per step by default; Space opens the code and output in place.
        self._collapsed = True
        self.styles.margin = (0, 0, 0, 2 * step.depth)
        self._head = Static(classes="step-head")
        self._code = Static(classes="step-code")
        self._out = Static(classes="step-output")
        self._more: Collapsible | None = None
        self._foot = Static(classes="step-foot")

    def compose(self):
        yield self._head
        yield self._code
        yield self._out
        yield self._foot

    def on_mount(self) -> None:
        self.refresh_step(self.step)

    def refresh_step(self, step: Step) -> None:
        self.step = step
        self._head.update(line_text(step, open_=not self._collapsed) if self._collapsed
                          else Text(header_text(step)))
        self.set_class(self._collapsed, "step-compact")
        self.set_class(step.state == FAILED, "step-failed")
        self.set_class(step.state in (GENERATING, RUNNING), "step-live")
        self._code.update(highlighted_code(step.code) if step.code else "")
        self._code.display = bool(step.code) and not self._collapsed
        shown, hidden = output_preview(step.output)
        self._out.update(escape(shown) if shown else "")
        self._out.display = bool(shown) and not self._collapsed
        if hidden and self._more is None:
            self._more = Collapsible(Static(escape(step.output)), collapsed=True,
                                     title=f"{hidden} more lines")
            self.mount(self._more, before=self._foot)
        elif self._more is not None:
            self._more.title = f"{hidden} more lines"
            self._more.query_one(Static).update(escape(step.output))
        if self._more is not None:
            self._more.display = bool(hidden) and not self._collapsed
        done = step.state in (DONE, FAILED)
        self._foot.update(footer_text(step) if done else "")
        self._foot.display = done and not self._collapsed



__all__ = [
    "Step",
    "StepBook",
    "StepCard",
    "footer_text",
    "header_text",
    "line_text",
    "output_preview",
    "short_id",
]
