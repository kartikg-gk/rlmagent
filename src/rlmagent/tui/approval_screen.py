"""Modal that asks before the sandbox writes a file or runs a command."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from rlmagent.agents.approval import ALWAYS, AUTO, DENY, ONCE, Request


class _Choice(ListItem):
    def __init__(self, text: str, value: str) -> None:
        super().__init__(Label(text))
        self.value = value


class ApprovalScreen(ModalScreen[str]):
    BINDINGS = [("escape", "deny", "Deny"), ("shift+tab", "auto", "Auto-accept")]

    def __init__(self, request: Request) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        always = "this command" if self.request.kind == "run" else "this file"
        with Vertical(id="approval-dialog"):
            yield Label("The agent wants to:", id="approval-title")
            yield Static(self.request.describe(), id="approval-body")
            yield ListView(
                _Choice("Allow once", ONCE),
                _Choice(f"Always allow {always} this session", ALWAYS),
                _Choice("Allow, and auto-accept the rest of this session", AUTO),
                _Choice("Deny", DENY),
                id="approval-list",
            )

    def on_mount(self) -> None:
        self.query_one("#approval-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.value)

    def action_auto(self) -> None:
        self.dismiss(AUTO)

    def action_deny(self) -> None:
        self.dismiss(DENY)
