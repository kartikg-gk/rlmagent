"""A one-line search field with undo and Emacs-style kill and yank.

Ctrl+Z undoes the last change. Ctrl+K cuts to the end, Ctrl+U to the start,
Ctrl+W the word before the cursor; Ctrl+Y puts the last cut back at the
cursor. Cuts in a row join into one, as a terminal user would expect.
"""

from __future__ import annotations

from textual.binding import Binding
from textual.widgets import Input

#: How many earlier values Ctrl+Z can step back through.
UNDO_DEPTH = 50


class SearchBox(Input):
    """``Input`` plus undo history and a single-slot kill ring."""

    BINDINGS = [
        Binding("ctrl+z", "undo", "Undo", show=False),
        Binding("ctrl+k", "kill_to_end", "Cut to end", show=False),
        Binding("ctrl+u", "kill_to_start", "Cut to start", show=False),
        Binding("ctrl+w", "kill_word", "Cut word", show=False),
        Binding("ctrl+y", "yank", "Paste cut", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._last = ""
        self._undoing = False
        self.killed = ""
        self._killing = False

    def on_input_changed(self, event: Input.Changed) -> None:
        # Record the value before each change so Ctrl+Z can return to it.
        if event.input is self and not self._undoing and event.value != self._last:
            self._history.append(self._last)
            del self._history[:-UNDO_DEPTH]
        self._last = event.value
        self._undoing = False

    def action_undo(self) -> None:
        if self._history:
            self._undoing = True
            self.value = self._history.pop()
            self.cursor_position = len(self.value)

    def _cut(self, start: int, end: int, *, before: bool) -> None:
        piece = self.value[start:end]
        if not piece:
            return
        # Consecutive cuts build one piece, in reading order.
        if self._killing:
            self.killed = piece + self.killed if before else self.killed + piece
        else:
            self.killed = piece
        self.value = self.value[:start] + self.value[end:]
        self.cursor_position = start
        self._killing = True

    def action_kill_to_end(self) -> None:
        self._cut(self.cursor_position, len(self.value), before=False)

    def action_kill_to_start(self) -> None:
        self._cut(0, self.cursor_position, before=True)

    def action_kill_word(self) -> None:
        end = self.cursor_position
        start = end
        while start and self.value[start - 1] == " ":
            start -= 1
        while start and self.value[start - 1] != " ":
            start -= 1
        self._cut(start, end, before=True)

    def action_yank(self) -> None:
        if self.killed:
            at = self.cursor_position
            self.value = self.value[:at] + self.killed + self.value[at:]
            self.cursor_position = at + len(self.killed)
        self._killing = False

    def on_key(self, event) -> None:
        # Any other key ends a run of cuts.
        if event.key not in ("ctrl+k", "ctrl+u", "ctrl+w"):
            self._killing = False


__all__ = ["SearchBox", "UNDO_DEPTH"]
