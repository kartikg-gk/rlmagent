"""Saved runs read back: step cards for a past session, and per-agent totals."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Static

from rlmagent.tui.steps import StepBook, short_id


def events_path(sessions: Path, session_id: str) -> Path:
    return sessions / f"{session_id}.backend.events.jsonl"


def load_events(path: Path) -> list[dict]:
    """The run events a session saved, skipping lines that do not parse."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def runs_of(events: list[dict]) -> list[StepBook]:
    """One book per run: a root run_started begins the next one."""
    books: list[StepBook] = []
    for event in events:
        if event.get("kind") == "run_started" and not event.get("parent_run_id"):
            books.append(StepBook())
        if not books:
            books.append(StepBook())
        books[-1].feed(event)
    return books


def agent_rows(books: list[StepBook]) -> list[dict]:
    """Per-agent steps, tokens, cost and seconds over every run of a session."""
    rows = []
    for run, book in enumerate(books, 1):
        for agent in book.agents.values():
            steps = [book.steps[(agent.run_id, n)] for n in agent.steps]
            costs = [s.cost for s in steps if s.cost is not None]
            seconds = [s.seconds for s in steps if s.seconds is not None]
            rows.append({
                "run": run,
                "agent": agent.run_id,
                "parent": agent.parent,
                "depth": agent.depth,
                "status": agent.status,
                "steps": len(steps),
                "failed_steps": sum(1 for s in steps if s.state == "failed"),
                "tokens": sum(s.tokens for s in steps),
                "cost": round(sum(costs), 6) if costs else None,
                "seconds": round(sum(seconds), 2) if seconds else None,
            })
    return rows


def format_rows(rows: list[dict]) -> str:
    if not rows:
        return "No agent steps were saved for this session."
    lines = [f"{'run':>3}  {'agent':<8} {'depth':>5} {'status':<9} {'steps':>5} "
             f"{'tokens':>8} {'cost':>9} {'seconds':>8}"]
    for row in rows:
        cost = "?" if row["cost"] is None else f"${row['cost']:.4f}"
        seconds = "?" if row["seconds"] is None else f"{row['seconds']:.1f}"
        lines.append(f"{row['run']:>3}  {short_id(row['agent']):<8} {row['depth']:>5} "
                     f"{row['status']:<9} {row['steps']:>5} {row['tokens']:>8} "
                     f"{cost:>9} {seconds:>8}")
    tokens = sum(r["tokens"] for r in rows)
    costs = [r["cost"] for r in rows if r["cost"] is not None]
    total = f"${sum(costs):.4f}" if costs else "unknown"
    lines.append(f"total: {len(rows)} agents, {sum(r['steps'] for r in rows)} steps, "
                 f"{tokens} tokens, cost {total}")
    return "\n".join(lines)


class ReplayApp(App[None]):
    """A saved session's steps as cards, read only."""

    CSS_PATH = Path(__file__).with_name("rlm_agent.tcss")
    TITLE = "rlm-agent replay"
    BINDINGS = [Binding("q", "quit", "Quit"), Binding("f2", "agent_tree", "Agents"),
                Binding("f3", "timeline", "Timeline")]

    def __init__(self, session_id: str, books: list[StepBook]) -> None:
        super().__init__()
        self.session_id = session_id
        self.books = books

    def compose(self) -> ComposeResult:
        from rlmagent.tui.transcript import TranscriptView

        yield Static(f"session {self.session_id} · {len(self.books)} runs", classes="step-foot")
        yield TranscriptView()
        yield Footer()

    def on_mount(self) -> None:
        from rlmagent.tui.transcript import TranscriptView

        view = self.query_one(TranscriptView)
        for run, book in enumerate(self.books, 1):
            view.add_notice(f"run {run}")
            view.new_run()
            for key in book.order:
                view.show_step(book.steps[key])

    def on_step_card_inspect(self, event) -> None:
        from rlmagent.tui.inspect import StepScreen

        self.push_screen(StepScreen(event.step))

    def _merged(self) -> StepBook:
        merged = StepBook()
        for book in self.books:
            merged.agents.update(book.agents)
            merged.steps.update(book.steps)
        return merged

    def action_agent_tree(self) -> None:
        from rlmagent.tui.inspect import AgentTreeScreen

        self.push_screen(AgentTreeScreen(self._merged()))

    def action_timeline(self) -> None:
        from rlmagent.tui.inspect import TimelineScreen

        self.push_screen(TimelineScreen(self.books[-1] if self.books else StepBook()))


__all__ = ["ReplayApp", "agent_rows", "events_path", "format_rows", "load_events", "runs_of"]
