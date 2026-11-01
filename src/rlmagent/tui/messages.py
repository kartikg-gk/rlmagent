"""Transcript block widgets: user turns, assistant turns, and tool lines."""

from __future__ import annotations

import re

from rich.console import Group
from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Static

_MAX_INLINE_OUTPUT = 400

RLM_AGENT_MARK = "›"  # prompt marker for user turns
ACCENT = "#c9a4ff"  # violet accent, shared by markup and rlm_agent.tcss


_CLOSED_FENCE = re.compile(r"```(?:python|py)\n(.*?)\n```", re.DOTALL)


def highlighted_code(code: str) -> Syntax:
    """Python as the terminal shows it: coloured, wrapped, on the app's background."""
    return Syntax(code, "python", theme="monokai", word_wrap=True, background_color="default")


def render_message(text: str) -> Text | Group:
    """Message text with its finished Python blocks coloured; literal, never markup."""
    parts: list[Text | Syntax] = []
    position = 0
    for fence in _CLOSED_FENCE.finditer(text):
        if fence.start() > position:
            parts.append(Text(text[position:fence.start()].strip("\n")))
        parts.append(highlighted_code(fence.group(1)))
        position = fence.end()
    if not parts:
        return Text(text)
    if text[position:].strip("\n"):
        parts.append(Text(text[position:].strip("\n")))
    return Group(*parts)


class UserBlock(Static):
    """One user message."""

    def __init__(self, text: str) -> None:
        super().__init__(f"{RLM_AGENT_MARK} {escape(text)}", classes="user-block")
        self._text = text

    @property
    def text(self) -> str:
        """The message this block displays."""
        return self._text


class AssistantBlock(Static):
    """One assistant message, appended to incrementally while streaming."""

    def __init__(self, text: str = "") -> None:
        super().__init__(render_message(text), classes="assistant-block")
        self._buffer = text

    @property
    def text(self) -> str:
        """Accumulated text so far."""
        return self._buffer

    def append(self, delta: str) -> None:
        """Append streamed text and re-render."""
        self._buffer += delta
        self.update(render_message(self._buffer))

    def finalize(self, text: str) -> None:
        """Replace the buffer with the completed message."""
        self._buffer = text
        self.update(render_message(text))


class ToolLine(Static):
    """A compact one-line record of a tool invocation."""

    def __init__(self, call_id: str, summary: str) -> None:
        super().__init__(f"· {escape(summary)}", classes="tool-line")
        self.call_id = call_id
        self._summary = summary

    def mark_done(self, *, is_error: bool) -> None:
        """Flip the pending marker to success or failure."""
        if is_error:
            self.update(f"✗ {escape(self._summary)}")
            self.add_class("tool-error")
        else:
            self.update(f"✓ {escape(self._summary)}")


class ToolCode(Static):
    """The Python a tool call runs."""

    def __init__(self, code: str) -> None:
        super().__init__(highlighted_code(code), classes="tool-code")


class ToolOutput(Vertical):
    """Collapsible container for a tool's stdout when it is large."""

    def __init__(self, output: str) -> None:
        super().__init__(classes="tool-output")
        self._output = output

    def compose(self) -> ComposeResult:
        lines = self._output.splitlines()
        title = f"{len(lines)} lines of output"
        with Collapsible(title=title, collapsed=True):
            yield Static(escape(self._output))


def should_collapse(output: str) -> bool:
    """Whether *output* is large enough to warrant a collapsible block."""
    return len(output) > _MAX_INLINE_OUTPUT or output.count("\n") > 6


class NoticeBlock(Static):
    """A non-conversational notice: command output, errors, cancellations."""

    def __init__(self, text: str, *, error: bool = False) -> None:
        super().__init__(escape(text), classes="notice-block")
        if error:
            self.add_class("tool-error")


__all__ = [
    "AssistantBlock",
    "NoticeBlock",
    "ToolCode",
    "ToolLine",
    "ToolOutput",
    "UserBlock",
    "should_collapse",
]
