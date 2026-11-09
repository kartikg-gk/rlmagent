"""Textual client for a worker that owns its conversation and execution."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.widgets import Footer, Static
from pydantic import TypeAdapter, ValidationError

from rlmagent import worker
from rlmagent.harness.contracts.transcript import TranscriptEntry
from rlmagent.tui.adapter import summarize_call
from rlmagent.tui.prompt import PromptInput
from rlmagent.tui.transcript import TranscriptView

_TRANSCRIPT = TypeAdapter(list[TranscriptEntry])


class WorkerAttachApp(App[None]):
    BINDINGS = [("ctrl+d", "quit", "Detach"), Binding("ctrl+c", "interrupt", "Interrupt", priority=True)]
    CSS = """
    TranscriptView { height: 1fr; }
    PromptInput { height: 4; }
    #worker-status { height: 1; }
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.worker_name = name
        self.cursor = worker.WorkerLog(name, recent_only=True)
        self.history_path = worker.folder(name) / "history.json"
        self.history_modified = None
        self.history_session = None
        self.last_model_reply = None

    def compose(self) -> ComposeResult:
        yield Static("", id="worker-status", markup=False)
        yield TranscriptView()
        yield PromptInput()
        yield Footer()

    def on_mount(self) -> None:
        self.poll_worker()
        self.set_interval(0.3, self.poll_worker)

    def poll_worker(self) -> None:
        try:
            transcript = self.query_one(TranscriptView)
            status = self.query_one("#worker-status", Static)
        except NoMatches:
            return  # A queued poll can run during teardown.
        try:
            modified = self.history_path.stat().st_mtime_ns
        except OSError:
            modified = None
        history = worker._read(self.history_path) if modified != self.history_modified else None
        if history is not None:
            self.history_modified = modified
        if history and history.get("session") != self.history_session:
            try:
                entries = _TRANSCRIPT.validate_python(history["entries"])
                offset = int(history["log_offset"])
            except (ValidationError, KeyError, TypeError, ValueError):
                self.history_session = history.get("session")
                transcript.add_notice("Could not read the saved session history.", error=True)
            else:
                self.history_session = history["session"]
                self.last_model_reply = None
                self.cursor = worker.WorkerLog(self.worker_name, start_at=max(0, offset))
                transcript.replay(entries)
        for row in self.cursor.read():
            kind, text = row["kind"], str(row["text"])
            if kind == "prompt":
                if transcript.live_text:
                    transcript.finish_assistant(transcript.live_text)
                self.last_model_reply = None
                transcript.add_user(text)
            elif kind == "model_delta":
                transcript.stream_delta(text)
            elif kind == "model_end":
                partial = transcript.live_text
                if partial and text.startswith(partial) and len(text) > len(partial):
                    transcript.stream_delta(text[len(partial):])
                transcript.finish_assistant(text)
                self.last_model_reply = text
            elif kind == "reply" and self.last_model_reply != text:
                transcript.finish_assistant(text)
            elif kind == "tool_start":
                args = row.get("args", {})
                code = str(args.get("code") or "") if text == "python" else ""
                transcript.start_tool(row["call_id"], summarize_call(text, args), code)
            elif kind == "tool_end":
                transcript.finish_tool(row["call_id"], text, is_error=bool(row["is_error"]))
            elif kind != "reply":
                transcript.add_notice(text, error=kind == "error")
        state = worker.state_of(self.worker_name)
        label = state.get("state", "starting") if state["alive"] else "offline · prompts queue"
        if state.get("state") == "passive" and not (worker.folder(self.worker_name) / "stop").exists():
            label = "passive · prompts queue"
        status.update(f"{self.worker_name} · {label}")

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        if event.text in ("/detach", "/quit", "/exit", "/q"):
            self.exit()
            return
        try:
            result = worker.answer_input(self.worker_name, event.text)
            if result is None:
                worker.send(self.worker_name, event.text)
            else:
                self.query_one(TranscriptView).add_notice(result)
        except OSError as error:
            self.query_one(TranscriptView).add_notice(str(error), error=True)
            return
        self.query_one(PromptInput).clear_prompt()

    def action_interrupt(self) -> None:
        if not worker.interrupt(self.worker_name):
            self.exit()
