"""Left sidebar — the session switcher, grouped by what each session is doing."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Label, ListItem, ListView, Static

from rlmagent.tui.adapter import SessionEntry
from rlmagent.tui.search_box import SearchBox

#: Section order and headings. Each session sits under exactly one.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("running", "Running"),
    ("idle", "Idle"),
    ("inactive", "Inactive"),
)


class SessionItem(ListItem):
    """A session row. Identity is carried as an attribute, not a widget id."""

    def __init__(self, label: str, session_id: str | None) -> None:
        super().__init__(Label(label, markup=False))
        self.session_id = session_id

    @property
    def is_new_action(self) -> bool:
        """Whether this row creates a new session."""
        return self.session_id is None


class SectionHeading(ListItem):
    """A section title inside the list; it cannot be chosen."""

    def __init__(self, title: str, count: int) -> None:
        super().__init__(Label(f"{title} ({count})", markup=False), classes="session-heading")
        self.disabled = True


class SessionList(ListView):
    """Selectable list of sessions, grouped by status, newest first in each group."""

    def __init__(self) -> None:
        super().__init__(id="session-list")


def matches(entry: SessionEntry, query: str) -> bool:
    """Whether *entry* fits a search by title or id, case-insensitively."""
    query = query.strip().lower()
    return not query or query in entry.label.lower() or query in entry.session_id.lower()


class Sidebar(Vertical):
    """Fixed-width column holding the session switcher."""

    class SwitchRequested(Message):
        """User picked an existing session."""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    class NewRequested(Message):
        """User asked for a fresh session."""

    def __init__(self) -> None:
        super().__init__()
        #: What the list was last built from, so an unchanged refresh only restyles.
        self._shape: tuple = ()
        self._entries: list[SessionEntry] = []
        #: Active session id at the last repaint, so the cursor only moves
        #: when the active session actually changes.
        self._active: str | None = None

    def compose(self) -> ComposeResult:
        yield Label("Sessions", id="sidebar-title")
        yield SearchBox(placeholder="search sessions", id="session-search")
        yield SessionList()
        yield Static("", id="task-panel")

    def show_tasks(self, items: list[dict]) -> None:
        """The agent's task list under the sessions; hidden while it is empty."""
        panel = self.query_one("#task-panel", Static)
        lines = [("✓ " if item["done"] else "· ") + escape(item["text"]) for item in items]
        panel.update("Tasks\n" + "\n".join(lines) if lines else "")
        panel.display = bool(lines)

    @property
    def query_text(self) -> str:
        return self.query_one("#session-search", Input).value

    async def show(self, entries: list[SessionEntry]) -> None:
        """Repaint the list, highlighting the active session."""
        self._entries = list(entries)
        await self._rebuild()

    async def _rebuild(self) -> None:
        listing = self.query_one(SessionList)
        query = self.query_text
        shown = [e for e in self._entries if matches(e, query)]
        active = next((e.session_id for e in self._entries if e.is_active), None)
        shape = (query, tuple((e.session_id, e.status) for e in shown))

        if shape == self._shape:
            self._restyle(listing, shown)
            if active != self._active:
                self._active = active
                listing.index = self._active_index(listing)
            return

        # clear() removes children lazily, so await it or the appends race
        # still-mounted widgets.
        await listing.clear()
        if not query:
            await listing.append(SessionItem("+ New session", None))
        for status, title in SECTIONS:
            group = [e for e in shown if e.status == status]
            if not group:
                continue
            await listing.append(SectionHeading(title, len(group)))
            for entry in group:
                await listing.append(SessionItem(self._label_for(entry), entry.session_id))
        if query and not shown:
            await listing.append(SectionHeading("No match", 0))
        self._shape = shape
        self._active = active
        self._restyle(listing, shown)
        listing.index = self._active_index(listing)

    @staticmethod
    def _label_for(entry: SessionEntry) -> str:
        marker = "●" if entry.is_active else " "
        return f"{marker} {entry.label}"

    def _active_index(self, listing: SessionList) -> int:
        for i, child in enumerate(listing.children):
            if isinstance(child, SessionItem) and child.has_class("active-session"):
                return i
        return 0

    def _restyle(self, listing: SessionList, entries: list[SessionEntry]) -> None:
        """Refresh labels and the active marker without touching the cursor."""
        by_id = {e.session_id: e for e in entries}
        for child in listing.children:
            if not isinstance(child, SessionItem) or child.is_new_action:
                continue
            entry = by_id.get(child.session_id or "")
            if entry is None:
                continue
            child.set_class(entry.is_active, "active-session")
            child.query(Label).first().update(self._label_for(entry))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "session-search":
            event.stop()
            await self._rebuild()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box opens the first match."""
        if event.input.id != "session-search":
            return
        event.stop()
        first = next((e for e in self._entries if matches(e, event.value)), None)
        if first is not None and not first.is_active:
            self.post_message(self.SwitchRequested(first.session_id))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Translate a row activation into an intent message."""
        event.stop()
        item = event.item
        if not isinstance(item, SessionItem):
            return
        if item.is_new_action:
            self.post_message(self.NewRequested())
        else:
            self.post_message(self.SwitchRequested(item.session_id or ""))


__all__ = ["SECTIONS", "SectionHeading", "SessionItem", "SessionList", "Sidebar", "matches"]
