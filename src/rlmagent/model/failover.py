"""Fall back to other provider/model pairs when the main one fails outright.

A request moves to the next pair only when it failed before producing any
output. Once text, reasoning or a tool call has streamed, the reply belongs
to that provider and its failure is reported as it is: replaying a half-sent
reply on another model would show the user two answers.

Pairs come from ``RLM_AGENT_FALLBACK`` as ``provider:model`` items separated
by commas, for example ``openrouter:qwen/qwen3-coder,anthropic:claude-sonnet-4-5``,
or from ``fallback`` in a trusted project's ``.rlm-agent/settings.toml``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Sequence

from rlmagent.harness.contracts.tooling import CancelToken, ToolSpec
from rlmagent.harness.contracts.transcript import TranscriptEntry
from rlmagent.harness.provider.base import ModelProvider
from rlmagent.harness.provider.wire import (
    SourceRetryEvent,
    StreamFaultEvent,
    StreamOpenEvent,
    WireEvent,
)

FALLBACK_ENV = "RLM_AGENT_FALLBACK"

# Events that carry no output: a request that failed after only these can be
# sent elsewhere without the user having seen anything from it.
_SILENT = (StreamOpenEvent, SourceRetryEvent)


def parse_fallbacks(text: str | None) -> list[tuple[str, str]]:
    """``"a:m1, b:m2"`` -> ``[("a", "m1"), ("b", "m2")]``; bad items are skipped."""
    pairs: list[tuple[str, str]] = []
    for item in (text or "").split(","):
        name, sep, model = item.strip().partition(":")
        if sep and name.strip() and model.strip():
            pairs.append((name.strip().lower(), model.strip()))
    return pairs


class FailoverProvider:
    """Wraps a provider; tries each fallback pair after an outright failure."""

    def __init__(
        self,
        primary: ModelProvider,
        fallbacks: Sequence[tuple[str, str]],
        resolve: Callable[[str], ModelProvider],
    ) -> None:
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self._resolve = resolve
        self._built: dict[str, ModelProvider] = {}
        self.last_route: tuple[str, str] | None = None

    def __getattr__(self, name: str):
        # Anything else (service tiers, shutdown hooks…) is the primary's.
        return getattr(self.primary, name)

    def _provider(self, name: str) -> ModelProvider | None:
        if name not in self._built:
            try:
                self._built[name] = self._resolve(name)
            except Exception:  # a fallback without credentials is skipped
                return None
        return self._built[name]

    async def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        signal: CancelToken | None = None,
        cache_key: str | None = None,
    ) -> AsyncIterator[WireEvent]:
        routes: list[tuple[str | None, str]] = [(None, model), *self.fallbacks]
        for index, (name, route_model) in enumerate(routes):
            provider = self.primary if name is None else self._provider(name)
            if provider is None:
                continue
            last = index == len(routes) - 1
            held: list[WireEvent] = []
            spoke = False
            failed = False
            async for event in provider.stream_response(
                model=route_model, system=system, messages=messages, tools=tools,
                signal=signal, cache_key=cache_key,
            ):
                if spoke:
                    yield event
                    continue
                if isinstance(event, _SILENT):
                    held.append(event)
                    continue
                if (isinstance(event, StreamFaultEvent) and event.reason == "error"
                        and not last and not (signal and signal.is_cancelled())):
                    failed = True
                    break
                spoke = True
                self.last_route = (name or "", route_model)
                for early in held:
                    yield early
                yield event
            if not failed:
                if not spoke:
                    for early in held:
                        yield early
                return


def with_fallbacks(
    primary: ModelProvider, resolve: Callable[[str], ModelProvider],
    env: dict[str, str] | None = None,
) -> ModelProvider:
    """Wrap *primary* when fallbacks are configured; otherwise return it as is."""
    source = env if env is not None else os.environ
    text = source.get(FALLBACK_ENV)
    if text is None and env is None:
        from rlmagent.config.loader import project_settings

        text = project_settings().get("fallback")
    pairs = parse_fallbacks(text)
    return FailoverProvider(primary, pairs, resolve) if pairs else primary
