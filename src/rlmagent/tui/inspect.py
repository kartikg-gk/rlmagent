"""Full views opened from the transcript: one step in full, and the agent tree."""

from __future__ import annotations

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, Tree

from rlmagent.tui.messages import highlighted_code
from rlmagent.tui.steps import Step, StepBook, footer_text, header_text, short_id


class StepScreen(ModalScreen[None]):
    """Everything one step has: code, the whole output, usage and time."""

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def __init__(self, step: Step) -> None:
        super().__init__(classes="inspect-screen")
        self.step = step

    def compose(self) -> ComposeResult:
        step = self.step
        with VerticalScroll(classes="inspect-body"):
            yield Static(Text(header_text(step)), classes="step-head")
            yield Static(Text(f"agent {step.agent} · model {step.model or 'unknown'}"),
                         classes="step-foot")
            yield Static(highlighted_code(step.code) if step.code else "(no code)",
                         classes="step-code")
            yield Static(escape(step.output) if step.output else "(no output)",
                         classes="step-output")
            yield Static(Text(footer_text(step)), classes="step-foot")
            yield Static("Esc to close", classes="step-foot")


def agent_label(book: StepBook, run_id: str) -> str:
    agent = book.agents[run_id]
    steps = [book.steps[(run_id, n)] for n in agent.steps]
    tokens = sum(s.tokens for s in steps)
    costs = [s.cost for s in steps if s.cost is not None]
    spent = f" · ${sum(costs):.4f}" if costs else ""
    return (f"{short_id(run_id)} · d{agent.depth} · {agent.status} · "
            f"{len(steps)} steps · {tokens} tokens{spent}")


class AgentTreeScreen(ModalScreen[None]):
    """Who started whom, with each agent's state and spend."""

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def __init__(self, book: StepBook) -> None:
        super().__init__(classes="inspect-screen")
        self.book = book

    def compose(self) -> ComposeResult:
        tree: Tree[str] = Tree("agents", classes="inspect-body")
        tree.show_root = False
        nodes = {}
        pending = list(self.book.agents.values())
        # Parents first, so every child has a node to hang from.
        pending.sort(key=lambda agent: agent.depth)
        for agent in pending:
            parent = nodes.get(agent.parent, tree.root)
            nodes[agent.run_id] = parent.add(agent_label(self.book, agent.run_id),
                                             data=agent.run_id, expand=True)
        if not self.book.agents:
            tree.root.add_leaf("no agents have run yet")
        tree.root.expand()
        yield tree
        yield Static("Esc to close", classes="step-foot")


BAR_WIDTH = 30


def timeline_rows(book: StepBook) -> list[str]:
    """One line per step in start order: offset, duration, a bar to scale, and who ran it."""
    steps = [s for s in book.steps.values() if s.started is not None]
    if not steps:
        return ["No timed steps yet."]
    steps.sort(key=lambda s: s.started)
    first = steps[0].started
    last = max((s.ended or s.started) for s in steps)
    span = max(last - first, 1e-6)
    rows = []
    for step in steps:
        start = int((step.started - first) / span * BAR_WIDTH)
        length = 1 if step.ended is None else max(1, round((step.ended - step.started) / span * BAR_WIDTH))
        bar = " " * start + ("█" if step.ended is not None else "▒") * length
        took = "running" if step.seconds is None else f"{step.seconds:6.1f}s"
        rows.append(f"+{step.started - first:7.1f}s {took:>8} |{bar:<{BAR_WIDTH + 1}}| "
                    f"d{step.depth} {step.short} step {step.number} {step.state}")
    return rows


class TimelineScreen(ModalScreen[None]):
    """When each step ran and for how long, across every agent."""

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def __init__(self, book: StepBook) -> None:
        super().__init__(classes="inspect-screen")
        self.book = book

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="inspect-body"):
            yield Static(Text("\n".join(timeline_rows(self.book))))
            yield Static("Esc to close", classes="step-foot")


def worker_rows() -> list[str]:
    """Every worker: whether it runs, its last heartbeat, and its waiting and scheduled jobs."""
    import time

    from rlmagent import worker

    base = worker.root()
    names = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not names:
        return ["No workers. Start one with: rlm-agent worker start NAME"]
    rows = []
    for name in names:
        meta = worker.state_of(name)
        state = meta.get("state", "?") if meta["alive"] else "not running"
        beat = meta.get("beat")
        ago = f"beat {time.time() - beat:.0f}s ago" if beat else "no heartbeat"
        jobs = worker.jobs_of(name)
        waiting = [j for j in jobs if j.get("status") == "waiting"]
        rows.append(f"{name}: {state}, {ago}, {len(waiting)} waiting")
        for job in waiting[:5]:
            when = time.strftime("%m-%d %H:%M", time.localtime(job["due"]))
            again = f", every {job['every']:.0f}s" if job.get("every") else ""
            rows.append(f"    {job['id']} at {when}{again}: {job['text'][:80]}")
    return rows


class WorkersScreen(ModalScreen[None]):
    """Background workers and their jobs."""

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="inspect-body"):
            yield Static(Text("\n".join(worker_rows())))
            yield Static("Esc to close", classes="step-foot")


__all__ = ["AgentTreeScreen", "StepScreen", "TimelineScreen", "WorkersScreen", "agent_label",
           "timeline_rows", "worker_rows"]
