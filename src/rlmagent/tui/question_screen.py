"""Modal for a question the agent asks the user."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, SelectionList, Static


class _Choice(ListItem):
    def __init__(self, text: str) -> None:
        super().__init__(Label(text, markup=False))
        self.value = text


class QuestionScreen(ModalScreen[str | list[str] | None]):
    BINDINGS = [("escape", "skip", "Don't answer")]

    def __init__(self, question: str, choices: list[str] | None, multiple: bool = False) -> None:
        super().__init__()
        self.question = question
        self.choices = choices or []
        self.multiple = multiple and bool(self.choices)

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("The agent asks:", id="approval-title")
            yield Static(self.question, id="approval-body", markup=False)
            if self.multiple:
                yield SelectionList(*[(c, c) for c in self.choices], id="approval-list")
                yield Label("Space picks; Enter in the box below sends the picks or your text.")
            elif self.choices:
                yield ListView(*[_Choice(c) for c in self.choices], id="approval-list")
            yield Input(placeholder="Type an answer, Esc to skip", id="question-answer")

    def on_mount(self) -> None:
        target = "#approval-list" if self.choices else "#question-answer"
        self.query_one(target).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        typed = event.value.strip()
        if self.multiple and not typed:
            self.dismiss(list(self.query_one(SelectionList).selected) or None)
        else:
            self.dismiss(typed or None)

    def action_skip(self) -> None:
        self.dismiss(None)
