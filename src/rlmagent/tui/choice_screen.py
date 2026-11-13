"""A small modal list: pick one value with the arrows and Enter, Esc to leave."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView


class _Option(ListItem):
    def __init__(self, value: str, current: bool) -> None:
        super().__init__(Label(("● " if current else "  ") + value, markup=False))
        self.value = value


class ChoiceScreen(ModalScreen[str | None]):
    """Returns the chosen value, or ``None`` when dismissed."""

    BINDINGS = [("escape", "leave", "Cancel")]

    def __init__(self, title: str, options: list[str], current: str | None = None) -> None:
        super().__init__()
        self.title_text = title
        self.options = list(options)
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label(self.title_text, id="approval-title")
            yield ListView(*[_Option(o, o == self.current) for o in self.options],
                           id="choice-list")

    def on_mount(self) -> None:
        view = self.query_one("#choice-list", ListView)
        if self.current in self.options:
            view.index = self.options.index(self.current)
        view.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.value)

    def action_leave(self) -> None:
        self.dismiss(None)


__all__ = ["ChoiceScreen"]
