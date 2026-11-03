"""Adapter boundary: ``AgentEvent`` stream -> TUI state."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import time

from rlmagent.conversation import CodingSession
from rlmagent.harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    RetryEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
)
from rlmagent.harness.contracts.transcript import ModelEntry
from rlmagent.tui.steps import StepBook

# ---------------------------------------------------------------------------
# View-model updates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Incremental assistant text to append to the live block."""

    text: str


@dataclass(frozen=True, slots=True)
class TurnFinished:
    """The assistant message completed."""

    text: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """A tool invocation began."""

    call_id: str
    name: str
    summary: str
    code: str = ""


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """A tool invocation completed."""

    call_id: str
    name: str
    output: str
    is_error: bool


@dataclass(frozen=True, slots=True)
class StepChanged:
    """An agent step started, produced code or output, or finished."""

    step: object  # rlmagent.tui.steps.Step


@dataclass(frozen=True, slots=True)
class RetryNotice:
    """A failed attempt is being retried; shown as progress, not as an error."""

    message: str


Update = TextDelta | TurnFinished | ToolStarted | ToolFinished | RetryNotice | StepChanged


# ---------------------------------------------------------------------------
# Status snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """One row in the session sidebar."""

    session_id: str
    title: str | None
    updated_at: float
    is_active: bool

    @property
    def label(self) -> str:
        """Display name — the title, or a neutral placeholder when unnamed."""
        return self.title or "Untitled session"


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Everything the footer displays."""

    provider: str
    model: str
    context_used: int
    context_limit: int
    branch: str | None
    plan_mode: bool = False
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    budget: str = ""

    @property
    def context_percent(self) -> int:
        """Context consumption as a whole percentage."""
        if self.context_limit <= 0:
            return 0
        return min(100, round(self.context_used * 100 / self.context_limit))


# ---------------------------------------------------------------------------
# Tool call summaries
# ---------------------------------------------------------------------------

_PATH_ARGS = ("file_path", "path", "notebook_path")


def summarize_call(name: str, args: dict) -> str:
    """One-line description of a tool call, e.g. ``Read foo.py``."""
    for key in _PATH_ARGS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return f"{name} {value.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}"
    command = args.get("command")
    if isinstance(command, str) and command:
        first = command.strip().splitlines()[0]
        return f"{name} {first[:60]}"
    pattern = args.get("pattern")
    if isinstance(pattern, str) and pattern:
        return f"{name} {pattern[:40]}"
    return name


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class SessionBridge:
    """Translates a ``CodingSession`` event stream into view-model updates."""

    __slots__ = ("_session", "_streamed", "steps")

    def __init__(self, session: CodingSession) -> None:
        self._session = session
        self._streamed = 0
        self.steps = StepBook()

    # -- read-only views ----------------------------------------------------

    @property
    def session(self) -> CodingSession:
        """The underlying runtime session."""
        return self._session

    @property
    def active(self) -> bool:
        """Whether a generation is currently running."""
        return self._session.active

    def status(self) -> StatusSnapshot:
        """Snapshot the footer state."""
        from rlmagent.runtime import git_branch

        usage = self._session.usage
        return StatusSnapshot(
            provider=self._session.provider_name,
            model=self._session.model,
            context_used=usage.estimated_context_tokens,
            context_limit=usage.context_limit,
            branch=git_branch(self._session.cwd),
            plan_mode=self._session.plan_mode,
            input_tokens=usage.total_input_tokens,
            cached_tokens=usage.cache_read_tokens,
            output_tokens=usage.total_output_tokens,
            cost=usage.total_cost_usd,
            budget=self.budget_line(),
        )

    def budget_line(self) -> str:
        """Steps, agents, depth and spend of the current run against its limits."""
        totals = self.steps.totals()
        if not totals["steps"]:
            return ""
        backend = getattr(self._session, "backend", None)
        limits = getattr(getattr(backend, "tree", None), "allowance", None)

        def of(used, limit):
            return f"{used}/{limit}" if limit is not None else f"{used}"

        parts = [f"steps {of(totals['steps'], getattr(limits, 'max_calls', None))}"]
        if totals["running"]:
            parts.append(f"{totals['running']} running")
        parts.append(f"depth {of(totals['depth'], getattr(limits, 'max_depth', None))}")
        if totals["cost"] is not None:
            cap = getattr(limits, "max_cost", None)
            parts.append(f"${totals['cost']:.4f}" + (f"/{cap:.2f}" if cap is not None else ""))
        return " ".join(parts)

    def sessions(self) -> list[SessionEntry]:
        """List every known session in a stable order, newest first."""
        catalog = self._session.catalog
        if catalog is None:
            return []
        active = self._session.session_id
        # A session belongs to the project it started in; never offer another project's here.
        here = Path(self._session.cwd).resolve()
        metas = sorted(
            (m for m in catalog.list_all() if not m.cwd or Path(m.cwd).resolve() == here),
            key=lambda m: m.created_at, reverse=True,
        )
        entries = [
            SessionEntry(
                session_id=meta.session_id,
                title=meta.title,
                updated_at=meta.updated_at,
                is_active=meta.session_id == active,
            )
            for meta in metas
        ]
        # A new session is only saved once something happens in it; until
        # then it is still the one being worked in, so it leads the list.
        if not any(entry.is_active for entry in entries):
            entries.insert(0, SessionEntry(
                session_id=active, title=self._session.title,
                updated_at=time(), is_active=True,
            ))
        return entries

    # -- actions ------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel the active generation via the runtime's own mechanism."""
        self._session.abort()

    async def switch_to(self, session_id: str) -> None:
        """Switch to an existing session via the runtime, reusing the provider."""
        if session_id == self._session.session_id:
            return
        self._session = await self._session.replace_session(session_id)
        self._streamed = 0

    async def start_new(self) -> None:
        """Create a fresh session via the runtime, reusing the provider."""
        self._session = await self._session.new_session()
        self._streamed = 0

    async def apply_model_choice(
        self, provider: str, model: str, api_key: str | None, base_url: str | None = None,
    ) -> str:
        """Persist credentials, rebuild the provider, and switch the session."""
        from rlmagent.config.config import resolve_provider
        from rlmagent.config.loader import apply_config
        from rlmagent.config.store import load_config, provider_info, remember_model, save_config

        try:
            info = provider_info(provider)
        except KeyError:
            # Not one of the configurable backends (e.g. a scripted provider
            # in tests). Only the model can change.
            await self._session.switch_model(model)
            return f"Switched to {provider} · {model}"

        # Save the key (and the selection) so the next launch reuses them.
        config = load_config()
        if api_key:
            config.api_keys[provider] = api_key
        if base_url is not None:
            config.base_urls[provider] = base_url
        config.provider = provider
        config.model = model
        save_config(config)
        remember_model(provider, model)

        # Export credentials for this process. These are set unconditionally:
        # an explicit switch must override whatever the previous provider left
        # behind (notably the base URL shared by OpenAI-compatible backends).
        import os

        if provider == "anthropic":
            key_var, url_var = "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"
        else:
            key_var, url_var = "OPENAI_API_KEY", "OPENAI_BASE_URL"

        key = None if provider == "vertex" else api_key or config.api_keys.get(provider) or os.environ.get(info.key_env)
        if key:
            os.environ[key_var] = key
        elif not info.needs_key and provider != "vertex":
            os.environ[key_var] = provider  # e.g. Ollama needs a placeholder
        if provider != "vertex":
            os.environ[url_var] = config.base_urls.get(provider) or info.base_url
        apply_config(provider, config)

        if provider == self._session.provider_name and base_url is None:
            await self._session.switch_model(model)
        else:
            await self._session.switch_provider(
                resolve_provider(provider), provider, model,
            )
        return f"Switched to {info.label} · {model}"

    async def shutdown(self) -> None:
        """Shut down the session currently held."""
        await self._session.shutdown()

    async def handle_command(self, text: str) -> str | None:
        """Delegate a slash command to the runtime."""
        return await self._session.handle_command(text)

    async def submit(self, text: str) -> AsyncIterator[Update]:
        """Send *text* and yield view-model updates as the runtime streams."""
        async for update in self._relay(self._session.submit(text)):
            yield update

    async def resume(self) -> AsyncIterator[Update]:
        """Continue the agent loop without adding a new user message."""
        async for update in self._relay(self._session.resume_run()):
            yield update

    async def _relay(self, events: AsyncIterator[AgentEvent]) -> AsyncIterator[Update]:
        """Translate a runtime event stream into view-model updates.

        Step events arrive on the engine's thread while the reply streams, so
        both are put on one queue and handed on in the order they came.
        """
        self._streamed = 0
        self.steps = StepBook()
        backend = getattr(self._session, "backend", None)
        if backend is None or not hasattr(backend, "listeners"):
            async for event in events:
                update = self._translate(event)
                if update is not None:
                    yield update
            return
        loop = asyncio.get_running_loop()
        merged: asyncio.Queue = asyncio.Queue()
        finished = object()

        def on_event(event: dict) -> None:
            loop.call_soon_threadsafe(merged.put_nowait, ("step", event))

        async def pump() -> None:
            try:
                async for event in events:
                    await merged.put(("agent", event))
            except BaseException as error:  # handed to the reader below
                await merged.put(("error", error))
            finally:
                await merged.put(("done", finished))

        backend.listeners.append(on_event)
        backend.cards = True
        task = asyncio.create_task(pump())
        try:
            while True:
                kind, item = await merged.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise item
                update = (StepChanged(step) if (step := self.steps.feed(item)) else None)                     if kind == "step" else self._translate(item)
                if update is not None:
                    yield update
            while not merged.empty():  # steps that landed after the reply ended
                kind, item = merged.get_nowait()
                if kind == "step" and (step := self.steps.feed(item)):
                    yield StepChanged(step)
        finally:
            backend.listeners.remove(on_event)
            backend.cards = False
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

    # -- translation --------------------------------------------------------

    def _translate(self, event: AgentEvent) -> Update | None:
        """Map one ``AgentEvent`` onto a view-model update."""
        if isinstance(event, MessageUpdateEvent):
            return self._on_update(event)
        if isinstance(event, MessageEndEvent):
            return self._on_end(event)
        if isinstance(event, ToolRunStartEvent):
            return ToolStarted(
                call_id=event.tool_call_id,
                name=event.tool_name,
                summary=summarize_call(event.tool_name, event.args),
                code=str(event.args.get("code") or "") if event.tool_name == "python" else "",
            )
        if isinstance(event, RetryEvent):
            return RetryNotice(event.message)
        if isinstance(event, ToolRunEndEvent):
            return ToolFinished(
                call_id=event.tool_call_id,
                name=event.tool_name,
                output=event.result.text,
                is_error=event.is_error,
            )
        return None

    def _on_update(self, event: MessageUpdateEvent) -> Update | None:
        """Emit only the newly streamed characters."""
        message = event.message
        if not isinstance(message, ModelEntry):
            return None
        full = message.text
        if len(full) <= self._streamed:
            return None
        delta = full[self._streamed:]
        self._streamed = len(full)
        return TextDelta(delta)

    def _on_end(self, event: MessageEndEvent) -> Update | None:
        """Finalize the assistant block, flushing any untransmitted tail."""
        message = event.message
        if not isinstance(message, ModelEntry):
            return None
        self._streamed = 0
        return TurnFinished(text=message.text, error=message.error_message)


__all__ = [
    "RetryNotice",
    "SessionBridge",
    "SessionEntry",
    "StatusSnapshot",
    "StepChanged",
    "TextDelta",
    "ToolFinished",
    "ToolStarted",
    "TurnFinished",
    "Update",
    "summarize_call",
]
