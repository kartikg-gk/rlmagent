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


__all__ = ["AgentTreeScreen", "StepScreen", "agent_label"]
