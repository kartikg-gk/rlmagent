"""Every model call in the tree passes through one shared allowance."""

from __future__ import annotations

from collections.abc import AsyncIterator

from rlmness import Allowance, AllowanceSpent, Spend

from rlmagent_harness.contracts.transcript import ModelEntry, TextSegment
from rlmagent_harness.contracts.transcript.diagnostics import UsageStats
from rlmagent_harness.provider.wire import StreamCloseEvent, StreamFaultEvent


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


class BudgetedProvider:
    def __init__(self, inner, allowance: Allowance) -> None:
        self.inner = inner
        self.allowance = allowance
        self.refusal: str | None = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def stream_response(self, *, model, system, messages, tools, signal=None, **extra) -> AsyncIterator:
        try:
            self.allowance.reserve()
        except AllowanceSpent as exc:
            # No allowance.cancel(): later calls would say "abandoned", not the limit.
            self.refusal = str(exc)
            entry = ModelEntry(
                content=[TextSegment(text="")],
                stop_reason="error",
                error_message=f"budget reached: {exc}",
            )
            yield StreamFaultEvent(reason="error", error=entry)
            return
        async for event in self.inner.stream_response(
            model=model, system=system, messages=messages, tools=tools, signal=signal, **extra
        ):
            if isinstance(event, StreamCloseEvent):
                self.allowance.settle(to_spend(event.message.usage))
            yield event
