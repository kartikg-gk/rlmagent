"""The side panel beside the chat: the session's skills and its sub-agents.

Two short lists next to the conversation; the full detail of any step is one
key away in the dashboard (F5).
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, TabbedContent, TabPane

from rlmagent.tui.steps import DONE, FAILED, StepBook, short_id

TEXT, DIM, FAINT, ACCENT, BAD = "#c8c8c8", "#7a7a7a", "#4a4a4a", "#c9a4ff", "#a05050"
TABS = ("tab-skills", "tab-agents")


def _mark(state: str) -> tuple[str, str]:
    if state == DONE:
        return "✓", DIM
    if state == FAILED:
        return "✗", BAD
    return "●", ACCENT


def _clip(text: object, width: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


class SidePanel(Vertical):
    """Tabs over the live step book and the session's skills."""

    def __init__(self) -> None:
        super().__init__(id="side-panel")
        self._drawn = -1
        self._skills: tuple = ()

    def compose(self) -> ComposeResult:
        with TabbedContent(id="side-tabs", initial="tab-skills"):
            for pane, label, body in (("tab-skills", "skills", "side-skills"),
                                      ("tab-agents", "subagents", "side-agents")):
                with TabPane(label, id=pane):
                    with VerticalScroll():
                        yield Static("", id=body)

    def show(self, book: StepBook, skills=()) -> None:
        """Redraw from *book*; does nothing when nothing changed."""
        names = tuple((getattr(s, "name", str(s)), getattr(s, "description", "") or "")
                      for s in skills)
        if book.version == self._drawn and names == self._skills:
            return
        self._drawn, self._skills = book.version, names
        children = [a for a in book.agents.values() if a.parent is not None]
        self._label("tab-skills", "skills", len(names))
        self._label("tab-agents", "subagents", len(children))
        self._draw_skills(names)
        self._draw_agents(book, children)

    def _label(self, pane: str, word: str, count: int) -> None:
        self.query_one(TabbedContent).get_tab(pane).label = f"{word} {count}" if count else word

    def _draw_skills(self, names) -> None:
        text = Text()
        for name, description in names:
            text.append(f"{name}\n", style="bold " + TEXT)
            if description:
                text.append(f"  {_clip(description, 40)}\n", style=DIM)
        self.query_one("#side-skills", Static).update(text or Text("no skills loaded", style=FAINT))

    def _draw_agents(self, book: StepBook, children) -> None:
        text = Text()
        for agent in children:
            mark, colour = _mark({"done": DONE, "failed": FAILED}.get(agent.status, "running"))
            name = book.child_names.get(agent.child_id or "", "") or short_id(agent.run_id)
            text.append("  " * max(0, agent.depth - 1))
            text.append(f"{mark} ", style=colour)
            text.append(_clip(name, 18), style="bold " + TEXT)
            text.append(f"  d{agent.depth} · {len(agent.steps)} cells · {agent.status}\n", style=DIM)
            note = book.child_notes.get(agent.child_id or "", "")
            if note:
                text.append("    " + _clip(note, 38) + "\n", style=FAINT)
        self.query_one("#side-agents", Static).update(text or Text("no sub-agents", style=FAINT))

    def next_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = TABS[(TABS.index(tabs.active) + 1) % len(TABS)]


__all__ = ["SidePanel"]
