"""Textual application for rlm-agent."""

from __future__ import annotations

import asyncio

import argparse
from contextlib import suppress
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Footer, Header

from rlmagent.tui.adapter import (
    AgentNote,
    RetryNotice,
    SessionBridge,
    StepChanged,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from rlmagent.tui.completions import CompletionBar, match_commands
from rlmagent.tui.composer import Composer, ModelBadge
from rlmagent.tui.messages import ACCENT, QueuedStrip
from rlmagent.tui.model_screen import ModelChoice, ModelScreen
from rlmagent.tui.prompt import PromptInput
from rlmagent.tui.sidebar import Sidebar
from rlmagent.tui.status import StatusBar
from rlmagent.tui.transcript import TranscriptView

_CSS_PATH = Path(__file__).with_name("rlm_agent.tcss")


def build_id() -> str:
    """Short fingerprint of the UI source actually loaded."""
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    if _CSS_PATH.exists():
        digest.update(_CSS_PATH.read_bytes())
    return digest.hexdigest()[:7]



def _is_registered_command(text: str) -> bool:
    """Whether *text* names a slash command rather than being a prompt."""
    from rlmagent.conversation import COMMAND_REGISTRY
    from rlmagent.directives import parse_command

    parsed = parse_command(text)
    return parsed is not None and COMMAND_REGISTRY.get(parsed[0]) is not None

class RlmAgentApp(App[None]):
    """rlm-agent's terminal UI."""

    CSS_PATH = _CSS_PATH
    TITLE = "rlm-agent"

    BINDINGS = [
        Binding("escape", "escape", "Stop / Quit", id="stop"),
        Binding("ctrl+c", "cancel", "Stop / Quit", id="cancel"),
        Binding("ctrl+n", "new_session", "New session", id="new_session"),
        Binding("ctrl+q", "quit", "Quit", id="quit"),
        Binding("shift+tab", "toggle_auto", "Auto-accept", priority=True, id="toggle_auto"),
        Binding("f2", "agent_tree", "Agents", id="agents", priority=True),
        Binding("f3", "timeline", "Timeline", id="timeline", priority=True),
        Binding("f4", "workers", "Workers", id="workers", priority=True),
        Binding("f1", "show_help", "Help", id="help", priority=True),
        Binding("ctrl+shift+y", "copy_answer", "Copy answer", id="copy_answer", priority=True),
    ]

    def __init__(self, bridge: SessionBridge | None = None) -> None:
        super().__init__()
        self._bridge = bridge
        self._busy = False
        #: Messages typed while the agent was working, sent in order afterwards.
        self._queued: list[str] = []
        #: Set by Ctrl+C so updates still in flight are not rendered.
        self._cancelled = False
        #: When Ctrl+C was last pressed; a second press soon after quits outright.
        self._last_interrupt = 0.0

    # -- composition --------------------------------------------------------

    def format_title(self, title: str, sub_title: str) -> Content:
        """Render the header title with a violet accent and no build suffix."""
        return Content.from_markup(f"[{ACCENT}]rlm[/]-agent")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield Sidebar()
            with Vertical(id="main"):
                yield TranscriptView()
                yield CompletionBar()
                yield QueuedStrip()
                yield Composer()
        yield StatusBar()
        yield Footer()

    async def on_mount(self) -> None:
        import json

        from rlmagent.plugins import rlm_agent_home

        try:
            keymap = json.loads((rlm_agent_home() / "keybindings.json").read_text(encoding="utf-8"))
            known = {binding.id for binding in self.BINDINGS}
            if not isinstance(keymap, dict) or any(
                key not in known or not isinstance(value, str) for key, value in keymap.items()
            ):
                raise ValueError("Keybindings must map known action IDs to key strings")
            assigned = {}
            for binding in self.BINDINGS:
                for key in keymap.get(binding.id, binding.key).split(","):
                    key = key.strip().lower()
                    if key and key in assigned and assigned[key] != binding.id:
                        raise ValueError(f"Shortcut {key} is assigned to multiple actions")
                    assigned[key] = binding.id
            self.set_keymap(keymap)
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as exc:
            self.notify(f"Could not load keybindings: {exc}", severity="error")
        await self._refresh_sidebar()
        self._refresh_status()
        if self._bridge is not None:
            self.query_one(TranscriptView).replay(self._bridge.session.transcript)
            tree = getattr(self._bridge.session, "agent_tree", None)
            if tree is not None and tree.approver.policy in ("ask", "auto"):
                tree.approver.ask = self._ask_approval
            if tree is not None:
                tree.approver.question = self._ask_question

    def action_toggle_auto(self) -> None:
        """Shift+Tab: stop or resume asking before file changes and commands, this session only."""
        tree = getattr(self._bridge.session, "agent_tree", None) if self._bridge else None
        if tree is None:
            return
        from rlmagent.tui.approval_screen import ApprovalScreen

        if isinstance(self.screen, ApprovalScreen):
            # On a waiting approval it lets that call through and turns auto-accept on.
            self.screen.action_auto()
            return
        on = tree.approver.toggle_auto()
        self.notify("auto-accept on: file changes and commands go ahead unasked" if on
                    else "auto-accept off")
        self._refresh_status()

    async def _ask_question(self, question: str, choices: list[str] | None,
                            multiple: bool = False):
        """Show a question from the agent and wait for the answer."""
        from rlmagent.tui.question_screen import QuestionScreen

        answer = asyncio.get_running_loop().create_future()
        self.push_screen(QuestionScreen(question, choices, multiple),
                         lambda text: answer.done() or answer.set_result(text))
        return await answer

    async def _ask_approval(self, request) -> str:
        """Show what the sandbox wants to change and wait for the answer."""
        from rlmagent.agents.approval import AUTO, DENY
        from rlmagent.tui.approval_screen import ApprovalScreen

        answer = asyncio.get_running_loop().create_future()
        self.push_screen(ApprovalScreen(request),
                         lambda choice: answer.done() or answer.set_result(choice or DENY))
        choice = await answer
        if choice == AUTO:
            self._refresh_status()
        return choice

    # -- user actions -------------------------------------------------------

    def on_prompt_input_text_changed(self, event: PromptInput.TextChanged) -> None:
        """Offer slash-command completions as the user types."""
        event.stop()
        commands = (
            self._bridge.session.COMMANDS if self._bridge is not None else ()
        )
        bar = self.query_one(CompletionBar)
        bar.offer(match_commands(event.text, commands))

    def _is_exact_command(self, text: str) -> bool:
        """Whether *text* is already a complete slash command name."""
        if self._bridge is None or not text.startswith("/"):
            return False
        name = text[1:].strip().lower()
        return any(name == cmd for cmd, _ in self._bridge.session.COMMANDS)

    def on_prompt_input_navigate(self, event: PromptInput.Navigate) -> None:
        """Arrow keys move the completion highlight when the bar is open."""
        event.stop()
        self.query_one(CompletionBar).move(event.delta)

    def on_prompt_input_complete(self, event: PromptInput.Complete) -> None:
        """Tab accepts the highlighted completion."""
        event.stop()
        bar = self.query_one(CompletionBar)
        chosen = bar.selected
        if chosen is None:
            return
        prompt = self.query_one(PromptInput)
        prompt.text = chosen.insertion
        prompt.cursor_location = (0, len(prompt.text))
        bar.dismiss()

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle Enter in the composer."""
        event.stop()
        bar = self.query_one(CompletionBar)
        if bar.active and not self._is_exact_command(event.text):
            # Still a partial word — Enter fills it in. A fully typed command
            # submits straight away rather than needing a second Enter.
            self.on_prompt_input_complete(PromptInput.Complete())
            return
        bar.dismiss()
        text = event.text
        if self._busy:
            if text.strip() and not _is_registered_command(text):
                self.query_one(PromptInput).clear_prompt()
                self._queued.append(text)
                self._show_queue()
            return
        self.query_one(PromptInput).clear_prompt()

        if self._bridge is None:
            self.query_one(TranscriptView).add_notice(
                "No session attached — running in layout-only mode.", error=True,
            )
            return

        if _is_registered_command(text):
            await self._run_command(text)
            return

        self.query_one(TranscriptView).add_user(text)
        self.run_worker(self._stream(text), exclusive=True)

    def on_step_card_inspect(self, event) -> None:
        from rlmagent.tui.inspect import StepScreen

        self.push_screen(StepScreen(event.step))

    def action_agent_tree(self) -> None:
        from rlmagent.tui.inspect import AgentTreeScreen
        from rlmagent.tui.steps import StepBook

        book = self._bridge.steps if self._bridge is not None else StepBook()
        self.push_screen(AgentTreeScreen(book))

    def action_timeline(self) -> None:
        from rlmagent.tui.inspect import TimelineScreen
        from rlmagent.tui.steps import StepBook

        self.push_screen(TimelineScreen(self._bridge.steps if self._bridge else StepBook()))

    def action_workers(self) -> None:
        from rlmagent.tui.inspect import WorkersScreen

        self.push_screen(WorkersScreen())

    def action_show_help(self) -> None:
        from rlmagent.tui.inspect import HelpScreen

        self.push_screen(HelpScreen())

    def action_copy_answer(self) -> None:
        if self._bridge is None:
            return
        answer = next((entry for entry in reversed(list(self._bridge.session.transcript))
                       if entry.role == "assistant"), None)
        if answer is None or not answer.text:
            self.notify("There is no answer to copy.")
            return
        from rlmagent.clipboard import copy

        try:
            def write(value):
                if self._driver is None:
                    raise ValueError("No terminal driver is attached")
                self._driver.write(value)
            self.notify(copy(answer.text, write))
        except ValueError as error:
            self.notify(str(error), severity="error")

    def action_escape(self) -> None:
        """Esc stops a running turn; when idle with an empty prompt it leaves the app."""
        if not self.action_stop() and not self.query_one(PromptInput).text.strip():
            self.exit()

    def action_stop(self) -> bool:
        """Stop the running generation. Returns whether anything was stopped."""
        if not self._busy or self._bridge is None:
            return False
        self._cancelled = True
        self._queued.clear()
        self._show_queue()
        self._bridge.cancel()
        transcript = self.query_one(TranscriptView)
        transcript.finish_assistant(transcript.live_text)
        transcript.add_notice("Stopped.")
        self.query_one(PromptInput).focus()
        self._refresh_status()
        return True

    def action_cancel(self) -> None:
        """Ctrl+C: stop the generation if running, otherwise quit.

        Pressed twice within a second and a half, it quits even mid-run.
        """
        import time

        now = time.monotonic()
        twice = now - self._last_interrupt < 1.5
        self._last_interrupt = now
        if twice:
            self.exit()
            return
        if self.action_stop():
            self.notify("Stopped. Press Ctrl+C again to quit.", timeout=1.5)
        else:
            self.exit()

    def _show_queue(self) -> None:
        with suppress(NoMatches):
            self.screen_stack[0].query_one(QueuedStrip).show(self._queued)

    def action_new_session(self) -> None:
        """Ctrl+N: start a fresh session."""
        self.post_message(Sidebar.NewRequested())

    async def on_sidebar_switch_requested(
        self, event: Sidebar.SwitchRequested,
    ) -> None:
        """Switch to an existing session."""
        event.stop()
        await self._swap(lambda b: b.switch_to(event.session_id))

    async def on_sidebar_new_requested(self, event: Sidebar.NewRequested) -> None:
        """Create a fresh session."""
        event.stop()
        await self._swap(lambda b: b.start_new())

    async def _swap(self, action) -> None:
        """Run a session change, then repaint every dependent surface."""
        if self._bridge is None:
            return
        transcript = self.query_one(TranscriptView)
        if self._busy:
            self._bridge.cancel()
            self._busy = False

        try:
            await action(self._bridge)
        except Exception as exc:  # noqa: BLE001 - keep the UI alive on failure
            transcript.add_notice(f"Could not switch session: {exc}", error=True)
            return

        # Stale state must not survive the swap.
        transcript.replay(self._bridge.session.transcript)
        self.query_one(PromptInput).clear_prompt()
        await self._refresh_sidebar()
        self._refresh_status()

    # -- runtime plumbing ---------------------------------------------------

    async def _run_command(self, text: str) -> None:
        """Delegate a slash command to the runtime."""
        transcript = self.query_one(TranscriptView)
        if text in {"/quit", "/exit", "/q"}:
            self.exit()
            return
        if text.strip() == "/version":
            from rlmagent.cli.main import get_version

            transcript.add_notice(f"rlm-agent v{get_version()}")
            return
        if text.strip() in {"/continue", "/resume"}:
            if self._busy:
                return
            self.run_worker(self._stream("", resume=True), exclusive=True)
            return
        if text.strip() in {"/model", "/provider"}:
            # push_screen_wait requires a worker context.
            self.run_worker(self._open_model_picker())
            return
        assert self._bridge is not None
        from rlmagent.conversation import COMMAND_REGISTRY
        from rlmagent.directives import parse_command

        name, arg = parse_command(text) or ("", "")
        command = COMMAND_REGISTRY.get(name)
        primary = command.name if command is not None else name
        transcript.add_command(text)
        if primary == "shell" and arg and hasattr(self._bridge.session, "run_shell"):
            try:
                entry = await self._bridge.session.run_shell(arg)
            except Exception as exc:  # noqa: BLE001
                transcript.add_notice(f"Shell failed: {exc}", error=True)
            else:
                transcript.add_shell(arg, entry.output, entry.exit_code)
            return
        if primary == "effort" or (primary == "think" and not arg):
            self.run_worker(self._open_effort_picker())
            return
        if primary == "btw" and arg and arg.lower() not in ("clear", "new"):
            response = await self._bridge.handle_command(text)
            transcript.add_side_answer(arg, response or "")
            return
        try:
            response = await self._bridge.handle_command(text)
        except Exception as exc:  # noqa: BLE001 - a bad command must not kill the UI
            transcript.add_notice(f"Command failed: {exc}", error=True)
            return
        if response is not None:
            transcript.add_notice(response)
        if not self._busy and text.split() in (["/autonomous"], ["/autonomous", "on"]):
            pilot = getattr(self._bridge.session, "autopilot", None)
            if pilot is not None and (following := pilot.next_prompt()):
                transcript.add_notice(f"autonomous turn {pilot.turns}")
                self.run_worker(self._stream(following), exclusive=True)
        await self._refresh_sidebar()
        self._refresh_status()

    async def _open_effort_picker(self) -> None:
        """Pick a thinking level from a list instead of typing it."""
        from rlmagent.reasoning import DEFAULT_THINKING_LEVEL, THINKING_LEVELS
        from rlmagent.tui.choice_screen import ChoiceScreen

        if self._bridge is None:
            return
        current = self._bridge.session.thinking_level or DEFAULT_THINKING_LEVEL
        options = ["off", *THINKING_LEVELS]
        level = await self.push_screen_wait(ChoiceScreen("Reasoning effort", options, current))
        if level is None:
            return
        response = await self._bridge.handle_command(f"/think {level}")
        self.query_one(TranscriptView).add_notice(response or f"effort: {level}")
        self._refresh_status()

    async def _open_model_picker(self) -> None:
        """Show the provider/model dialog and apply whatever comes back."""
        if self._bridge is None:
            return
        session = self._bridge.session
        choice = await self.push_screen_wait(
            ModelScreen(session.provider_name, session.model)
        )
        if choice is None:
            return
        await self._apply_model_choice(choice)

    async def _apply_model_choice(self, choice: ModelChoice) -> None:
        """Persist the selection and switch the runtime onto it."""
        assert self._bridge is not None
        transcript = self.query_one(TranscriptView)
        try:
            summary = await self._bridge.apply_model_choice(
                choice.provider, choice.model, choice.api_key, **({"base_url": choice.base_url} if choice.base_url else {}),
            )
        except Exception as exc:  # noqa: BLE001 - a bad key must not kill the UI
            transcript.add_notice(f"Could not switch: {exc}", error=True)
            return
        transcript.add_notice(summary)
        self._refresh_status()

    async def _stream(self, text: str, *, resume: bool = False) -> None:
        """Consume the runtime's event stream and render it."""
        assert self._bridge is not None
        transcript = self.screen_stack[0].query_one(TranscriptView)
        transcript.new_run()
        self._busy = True
        self._cancelled = False
        self._refresh_status()
        try:
            source = (
                self._bridge.resume() if resume else self._bridge.submit(text)
            )
            async for update in source:
                if self._cancelled:
                    # Ctrl+C already closed the block and posted a notice;
                    # trailing updates must not reopen it.
                    break
                if isinstance(update, TextDelta):
                    transcript.stream_delta(update.text)
                elif isinstance(update, ToolStarted):
                    transcript.start_tool(update.call_id, update.summary, update.code)
                elif isinstance(update, ToolFinished):
                    transcript.finish_tool(
                        update.call_id, update.output, is_error=update.is_error,
                    )
                elif isinstance(update, StepChanged):
                    transcript.show_step(update.step)
                    self._refresh_status()
                elif isinstance(update, RetryNotice):
                    transcript.add_notice(update.message)
                elif isinstance(update, AgentNote):
                    transcript.add_notice(update.text)
                elif isinstance(update, TurnFinished):
                    transcript.finish_assistant(update.text)
                    if update.error:
                        transcript.add_notice(update.error, error=True)
        except Exception as exc:  # noqa: BLE001 - surface provider errors in-UI
            transcript.finish_assistant("")
            transcript.add_notice(str(exc), error=True)
        finally:
            self._busy = False
            # The app may be tearing down (quit during generation), in which
            # case these widgets are already gone.
            with suppress(NoMatches):
                await self._refresh_sidebar()
                self._refresh_status()
            pilot = getattr(self._bridge.session, "autopilot", None)
            if self._queued and not self._cancelled:
                following = self._queued.pop(0)
                self._show_queue()
                with suppress(NoMatches):
                    transcript.add_user(following)
                    self.run_worker(self._stream(following), exclusive=True)
            elif pilot is not None and self._cancelled:
                pilot.stop_autonomous("stopped by the user")
            elif pilot is not None and (following := pilot.next_prompt()):
                with suppress(NoMatches):
                    transcript.add_notice(f"autonomous turn {pilot.turns}")
                    self.run_worker(self._stream(following), exclusive=True)
            elif pilot is not None and pilot.stopped_because:
                with suppress(NoMatches):
                    transcript.add_notice(f"autonomous mode stopped: {pilot.stopped_because}")
                pilot.stopped_because = ""

    def _refresh_status(self) -> None:
        """Update the footer readout and the composer's model/effort badge."""
        if self._bridge is None:
            return
        from rlmagent.reasoning import DEFAULT_THINKING_LEVEL

        session = self._bridge.session
        main = self.screen_stack[0]
        pilot = getattr(session, "autopilot", None)
        goal = getattr(pilot, "goal", None)
        goal_text = goal.objective if goal is not None and goal.status == "active" else ""
        main.query_one(StatusBar).show(self._bridge.status(), busy=self._busy, goal=goal_text)
        tasks = getattr(session, "tasks", None)
        if tasks is not None:
            main.query_one(Sidebar).show_tasks(tasks.items)
        effort = session.thinking_level or DEFAULT_THINKING_LEVEL
        tree = getattr(session, "agent_tree", None)
        auto = tree is not None and tree.approver.policy == "auto"
        main.query_one(ModelBadge).show(session.model, effort, auto)

    async def _refresh_sidebar(self) -> None:
        """Repaint the session list from the catalog."""
        if self._bridge is None:
            return
        await self.screen_stack[0].query_one(Sidebar).show(self._bridge.sessions())


def needs_setup(ns: argparse.Namespace) -> bool:
    """No usable provider yet, so a first run should open the setup page."""
    from rlmagent.cli.main import _resolve_provider_name
    from rlmagent.config.config import resolve_provider
    from rlmagent.config.loader import apply_config
    from rlmagent.model.settings import ConfigError

    if getattr(ns, "provider", None):
        return False
    try:
        name = _resolve_provider_name(None)
        apply_config(name)
        resolve_provider(name)
    except (ConfigError, KeyError, ValueError):
        return True
    return False


class SetupApp(App):
    """Just the model page, shown before the first session exists."""

    choice = None

    def on_mount(self) -> None:
        from rlmagent.tui.model_screen import ModelScreen

        def done(choice) -> None:
            self.choice = choice
            self.exit()

        self.push_screen(ModelScreen(), done)


def save_setup(choice) -> None:
    from rlmagent.config.store import load_config, remember_model, save_config

    config = load_config()
    if choice.api_key:
        config.api_keys[choice.provider] = choice.api_key
    if choice.base_url:
        config.base_urls[choice.provider] = choice.base_url
    config.provider, config.model = choice.provider, choice.model
    save_config(config)
    remember_model(choice.provider, choice.model)


async def setup_if_needed(ns: argparse.Namespace) -> bool:
    if needs_setup(ns):
        setup = SetupApp()
        await setup.run_async()
        if setup.choice is None:
            return False
        save_setup(setup.choice)
    return True


async def run_app(ns: argparse.Namespace) -> int:
    """Build the shared runtime session and run the TUI over it."""
    from rlmagent.runtime import build_session, settle_project_trust
    from rlmagent.tui.trust_dialog import ask_in_dialog

    # Settled before the session exists: its answer decides what gets loaded.
    await settle_project_trust(ns, ask=ask_in_dialog)
    if not await setup_if_needed(ns):
        return 1
    bridge = SessionBridge(await build_session(ns))
    app = RlmAgentApp(bridge)
    try:
        await app.run_async()
    finally:
        # The bridge's session changes when the user switches; shut down
        # whichever one is current.
        await bridge.shutdown()
    return 0


def run_tui(ns: argparse.Namespace | None = None) -> int:
    """Launch the TUI. Returns a process exit code."""
    import asyncio

    if ns is None:
        RlmAgentApp().run()
        return 0
    return asyncio.run(run_app(ns))


__all__ = ["RlmAgentApp", "run_app", "run_tui"]
