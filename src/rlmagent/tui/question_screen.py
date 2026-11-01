"""Modal for a question the agent asks the user."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static


class _Choice(ListItem):
    def __init__(self, text: str) -> None:
        super().__init__(Label(text, markup=False))
        self.value = text


class QuestionScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "skip", "Don't answer")]

    def __init__(self, question: str, choices: list[str] | None) -> None:
        super().__init__()
        self.question = question
        self.choices = choices or []

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("The agent asks:", id="approval-title")
            yield Static(self.question, id="approval-body", markup=False)
            if self.choices:
                yield ListView(*[_Choice(c) for c in self.choices], id="approval-list")
            yield Input(placeholder="Type an answer, Esc to skip", id="question-answer")

    def on_mount(self) -> None:
        target = "#approval-list" if self.choices else "#question-answer"
        self.query_one(target).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_skip(self) -> None:
        self.dismiss(None)
