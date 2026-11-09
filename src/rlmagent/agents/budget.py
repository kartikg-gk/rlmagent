"""Every model call in the tree passes through one shared allowance."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass

from rlmness import Allowance, AllowanceSpent, Spend

from rlmagent.harness.contracts.transcript import ModelEntry, TextSegment
from rlmagent.harness.contracts.transcript.diagnostics import UsageStats
from rlmagent.harness.provider.wire import StreamCloseEvent, StreamFaultEvent


def to_spend(usage: UsageStats) -> Spend:
    cost = usage.cost.total
    return Spend(
        prompt_tokens=usage.input,
        completion_tokens=usage.output,
        total_tokens=usage.total_tokens or usage.input + usage.output,
        # A provider that reports no price is not free; say "unknown".
        cost=cost if cost > 0 else None,
        cached_tokens=usage.cache_read or None,
        reasoning_tokens=usage.reasoning,
    )


REFUSED = "budget reached"

#: Characters per token assumed when sizing a request before it is sent. Lower
#: than any tokenizer's real average, so the estimate errs high.
CHARS_PER_TOKEN = 3
#: Reply tokens assumed when the provider's own reply limit cannot be read.
DEFAULT_REPLY_TOKENS = 8192


def _text_of(value) -> str:
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_text_of(part) for part in content)
    return str(value) if value is not None else ""


def estimate_prompt_tokens(system, messages) -> int:
    """An upper estimate of a request's prompt, all of it counted as uncached."""
    chars = len(_text_of(system)) + sum(len(_text_of(m)) for m in messages or ())
    return chars // CHARS_PER_TOKEN + 1


def reply_limit(provider) -> int:
    """The most tokens the provider will write in one reply, as far as it says."""
    seen = set()
    while provider is not None and id(provider) not in seen:
        seen.add(id(provider))
        profile = getattr(provider, "_profile", None)
        limit = getattr(profile, "max_tokens", None)
        if isinstance(limit, int) and limit > 0:
            return limit
        provider = getattr(provider, "inner", None) or getattr(provider, "_inner", None)
    return DEFAULT_REPLY_TOKENS


class NewTokenBudget:
    """Reserve the unmeasured balance for one provider request at a time."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self.unmeasured = False
        self.lock = asyncio.Lock()

    def refusal(self, upcoming: int = 0) -> str | None:
        """Why the next request may not go out; `upcoming` is its estimated size."""
        if self.unmeasured:
            return "new tokens could not be measured for the preceding request"
        if self.used >= self.limit:
            return f"{self.used} new tokens reaches the limit of {self.limit}"
        if upcoming and self.used + upcoming > self.limit:
            return (f"the next request could use about {upcoming} new tokens, and "
                    f"{self.used} of {self.limit} are already used")
        return None

    def settle(self, usage: UsageStats) -> None:
        tokens = max(usage.input + usage.cache_write + usage.output,
                     usage.total_tokens - usage.cache_read)
        if not (tokens or usage.cache_read or usage.total_tokens):
            self.unmeasured = True
        self.used += tokens


@dataclass
class ContractAllowance(Allowance):
    new_token_budget: NewTokenBudget | None = None


class BudgetedProvider:
    def __init__(self, inner, allowance: Allowance) -> None:
        self.inner = inner
        self.allowance = allowance
        self.refusal: str | None = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def stream_response(self, *, model, **kwargs) -> AsyncIterator:
        if getattr(self.allowance, "new_token_budget", None) is None:
            return self._reserved_stream(model=model, **kwargs)
        upcoming = self.request_size(kwargs.get("system"), kwargs.get("messages"))
        return self.measured_stream(model, lambda: self._reserved_stream(model=model, **kwargs),
                                    upcoming)

    def request_size(self, system, messages) -> int:
        """Estimated new tokens of a request: its whole prompt plus the longest reply."""
        return estimate_prompt_tokens(system, messages) + reply_limit(self.inner)

    def _refuse(self, model, message, usage=None):
        self.refusal = message
        entry = ModelEntry(model=model, content=[TextSegment(text="")], stop_reason="error",
                           error_message=f"{REFUSED}: {message}", usage=usage)
        return StreamFaultEvent(reason="error", error=entry)

    async def measured_stream(self, model, start, upcoming: int = 0) -> AsyncIterator:
        budget = self.allowance.new_token_budget
        terminal = None
        async with budget.lock:
            if (reason := budget.refusal(upcoming)) is not None:
                terminal = self._refuse(model, reason)
            else:
                settled = False
                try:
                    async with aclosing(start()) as stream:
                        async for event in stream:
                            if isinstance(event, StreamCloseEvent):
                                budget.settle(event.message.usage)
                                settled = True
                                terminal = event
                                if budget.used > budget.limit:
                                    terminal = self._refuse(model, budget.refusal(), event.message.usage)
                                break
                            if isinstance(event, StreamFaultEvent):
                                if not (event.error.error_message or "").startswith(REFUSED):
                                    budget.settle(event.error.usage)
                                settled = True
                                terminal = event
                                break
                            yield event
                finally:
                    if not settled:
                        budget.unmeasured = True
        if terminal is not None:
            yield terminal

    async def _reserved_stream(self, *, model, system, messages, tools, signal=None, **extra) -> AsyncIterator:
        try:
            self.allowance.reserve()
        except AllowanceSpent as exc:
            # No allowance.cancel(): later calls would say "abandoned", not the limit.
            yield self._refuse(model, str(exc))
            return
        async for event in self.inner.stream_response(
            model=model, system=system, messages=messages, tools=tools, signal=signal, **extra
        ):
            if isinstance(event, StreamCloseEvent):
                self.allowance.settle(to_spend(event.message.usage))
            yield event
