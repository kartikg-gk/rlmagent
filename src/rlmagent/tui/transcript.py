"""Main transcript area — scrollable conversation view."""

from __future__ import annotations

from textual.containers import VerticalScroll

from rlmagent.tui.adapter import summarize_call
from rlmagent.tui.messages import (
    AssistantBlock,
    CommandEcho,
    ErrorBlock,
    KindNotice,
    NoticeBlock,
    ShellBlock,
    SideAnswer,
    ToolCode,
    ToolLine,
    ToolOutput,
    UserBlock,
    should_collapse,
)
from rlmagent.tui.steps import StepCard


class TranscriptView(VerticalScroll):
    """Scrollable region holding the conversation."""

    def __init__(self) -> None:
        super().__init__()
        self._live: AssistantBlock | None = None
        self._tools: dict[str, ToolLine] = {}
        self._cards: dict[tuple[str, int], StepCard] = {}

    # -- appending ----------------------------------------------------------

    def add_user(self, text: str) -> None:
        """Append a user turn."""
        self.mount(UserBlock(text))
        self._scroll()

    def add_notice(self, text: str, *, error: bool = False) -> None:
        """Append a notice (command output, error, cancellation).

        A multi-line error shows its first line and folds the rest.
        """
        if error and "\n" in text.strip():
            self.mount(ErrorBlock(text))
        else:
            self.mount(NoticeBlock(text, error=error))
        self._scroll()

    def add_command(self, text: str) -> None:
        """Echo a slash command the user ran."""
        self.mount(CommandEcho(text))
        self._scroll()

    def add_shell(self, command: str, output: str, exit_code: int | None = None) -> None:
        """A shell result in its box, last lines only."""
        self.mount(ShellBlock(command, output, exit_code))
        self._scroll()

    def add_side_answer(self, question: str, answer: str) -> None:
        self.mount(SideAnswer(question, answer))
        self._scroll()

    def add_kind(self, kind: str, text: str) -> None:
        """A labelled notice: compaction, refinement, skill, injected prompt."""
        self.mount(KindNotice(kind, text))
        self._scroll()

    def start_tool(self, call_id: str, summary: str, code: str = "") -> None:
        """Append a pending tool line, and the code it runs if any."""
        line = ToolLine(call_id, summary)
        self._tools[call_id] = line
        self.mount(line)
        if code:
            self.mount(ToolCode(code))
        self._scroll()

    def finish_tool(self, call_id: str, output: str, *, is_error: bool) -> None:
        """Resolve a pending tool line and attach large output."""
        line = self._tools.pop(call_id, None)
        if line is not None:
            line.mark_done(is_error=is_error)
        if output and should_collapse(output):
            self.mount(ToolOutput(output))
        self._scroll()

    def show_step(self, step) -> StepCard:
        """Add a card for a new step, or redraw the card it already has."""
        card = self._cards.get(step.key)
        if card is None:
            card = StepCard(step)
            self._cards[step.key] = card
            if self._live is not None and not self._live.text.strip():
                self.mount(card, before=self._live)
            else:
                self.mount(card)
        else:
            card.refresh_step(step)
        self._scroll()
        return card

    def new_run(self) -> None:
        """Later steps get new cards; the last run's cards stay as they are."""
        self._cards.clear()

    @property
    def cards(self) -> list[StepCard]:
        return list(self._cards.values())

    # -- streaming ----------------------------------------------------------

    def stream_delta(self, delta: str) -> None:
        """Append streamed assistant text, opening a block on first delta."""
        if self._live is None:
            self._live = AssistantBlock()
            self.mount(self._live)
        self._live.append(delta)
        self._scroll()

    @property
    def live_text(self) -> str:
        """Text streamed into the open assistant block so far."""
        return self._live.text if self._live is not None else ""

    def finish_assistant(self, text: str) -> None:
        """Close the live assistant block."""
        if self._live is None:
            if text:
                self.mount(AssistantBlock(text))
        else:
            self._live.finalize(text)
        self._live = None
        self._scroll()

    def reset(self) -> None:
        """Clear the transcript (used when switching sessions)."""
        self._live = None
        self._tools.clear()
        self._cards.clear()
        self.remove_children()

    def replay(self, entries) -> None:
        """Repaint the view from a session's stored transcript."""
        self.reset()
        for entry in entries:
            role = getattr(entry, "role", None)
            if role == "user":
                self.mount(UserBlock(entry.text))
            elif role == "assistant":
                for call in entry.tool_calls:
                    line = ToolLine(call.id, summarize_call(call.name, call.arguments))
                    line.mark_done(is_error=False)
                    self.mount(line)
                    code = call.arguments.get("code") if call.name == "python" else None
                    if isinstance(code, str) and code:
                        self.mount(ToolCode(code))
                if entry.text:
                    self.mount(AssistantBlock(entry.text))
            elif role == "compactionSummary":
                self.mount(KindNotice("compacted", entry.summary))
        self._scroll()

    # -- internal -----------------------------------------------------------

    def _scroll(self) -> None:
        """Keep the newest content in view."""
        self.scroll_end(animate=False)


__all__ = ["TranscriptView"]
