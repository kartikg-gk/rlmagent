"""Transcript block widgets: user turns, assistant turns, and tool lines."""

from __future__ import annotations

from rich.markdown import Markdown
from rich.markup import escape
from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Static

_MAX_INLINE_OUTPUT = 400

RLM_AGENT_MARK = "›"  # prompt marker for user turns
ACCENT = "#c9a4ff"  # violet accent, shared by markup and rlm_agent.tcss


def highlighted_code(code: str) -> Syntax:
    """Python as the terminal shows it: coloured, wrapped, on the app's background."""
    return Syntax(code, "python", theme="monokai", word_wrap=True, background_color="default")


def render_message(text: str) -> Markdown:
    """Message text as formatted Markdown; never read as console markup."""
    return Markdown(text, code_theme="monokai")


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
        # Progress streamed before the answer (code, output) stays on screen above it.
        if self._buffer.strip() and text not in self._buffer:
            text = f"{self._buffer.rstrip()}\n\n{text}" if text else self._buffer
        elif text in self._buffer:
            text = self._buffer
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


#: Lines a shell result keeps on screen; earlier ones are counted, not shown.
SHELL_TAIL = 20


class ErrorBlock(Vertical):
    """An error: its first line always shown, the rest folded underneath."""

    def __init__(self, text: str) -> None:
        super().__init__(classes="error-block")
        self._text = text.strip() or "error"

    @property
    def headline(self) -> str:
        return self._text.splitlines()[0]

    def compose(self) -> ComposeResult:
        first, _, rest = self._text.partition("\n")
        yield Static(f"✗ {escape(first)}", classes="tool-error")
        if rest.strip():
            with Collapsible(title=f"details ({len(rest.splitlines())} lines)", collapsed=True):
                yield Static(escape(rest.strip()))


class ShellBlock(Static):
    """A shell command's result in a box, showing only its last lines."""

    def __init__(self, command: str, output: str, exit_code: int | None = None) -> None:
        lines = output.rstrip("\n").splitlines()
        hidden = max(0, len(lines) - SHELL_TAIL)
        body = "\n".join(lines[-SHELL_TAIL:]) or "(no output)"
        if hidden:
            body = f"… {hidden} earlier line{'s' if hidden != 1 else ''}\n{body}"
        super().__init__(escape(body), classes="shell-block")
        self.border_title = f"$ {command}"
        self.border_subtitle = "" if exit_code is None else f"exit {exit_code}"
        if exit_code not in (None, 0):
            self.add_class("shell-failed")
        self.hidden_lines = hidden


class CommandEcho(Static):
    """The slash command the user ran, with its name picked out."""

    def __init__(self, text: str) -> None:
        name, _, rest = text.strip().partition(" ")
        markup = f"[{ACCENT}]{escape(name)}[/]" + (f" {escape(rest)}" if rest else "")
        super().__init__(f"{RLM_AGENT_MARK} {markup}", classes="command-echo")


class SideAnswer(Static):
    """An answer to a /btw side question, set apart from the main thread."""

    def __init__(self, question: str, answer: str) -> None:
        super().__init__(classes="side-answer")
        self.update(render_message(f"**btw** {question}\n\n{answer}"))


class KindNotice(Static):
    """A notice with a kind label: compaction, refinement, skill, injected."""

    def __init__(self, kind: str, text: str) -> None:
        super().__init__(f"[{ACCENT}]{escape(kind)}[/] {escape(text)}",
                         classes=f"kind-notice kind-{kind.replace(' ', '-')}")


class QueuedStrip(Static):
    """Messages waiting for the current run to end, shown above the prompt."""

    def __init__(self) -> None:
        super().__init__("", id="queued-strip")
        self.display = False

    def show(self, queued: list[str]) -> None:
        self.display = bool(queued)
        if queued:
            rows = [f"{i}. {escape(text.splitlines()[0][:120])}" for i, text in enumerate(queued, 1)]
            self.update(f"queued ({len(queued)}) — sent when this run ends\n" + "\n".join(rows))


__all__ = [
    "AssistantBlock",
    "CommandEcho",
    "ErrorBlock",
    "KindNotice",
    "QueuedStrip",
    "SHELL_TAIL",
    "ShellBlock",
    "SideAnswer",
    "NoticeBlock",
    "ToolCode",
    "ToolLine",
    "ToolOutput",
    "UserBlock",
    "should_collapse",
]
