"""Exponential backoff with cancellation support for model adapters."""

from __future__ import annotations

from asyncio import sleep
from collections import deque
from time import monotonic

from rlmagent.harness.contracts.tooling import CancelToken
from rlmagent.harness.contracts.values import JValue
from rlmagent.harness.provider.wire import SourceRetryEvent

RETRY_POLL_SECONDS = 0.05
RETRY_BASE_DELAY_SECONDS = 0.25
#: Retries one provider may spend in any rolling window, across all requests.
#: A provider that keeps failing should fail fast, not stall every turn.
RETRY_BUDGET = 8
RETRY_BUDGET_WINDOW_SECONDS = 600.0


class RetryBudget:
    """Caps retries across every request a provider makes, per rolling window."""

    def __init__(self, limit: int = RETRY_BUDGET,
                 window_seconds: float = RETRY_BUDGET_WINDOW_SECONDS, clock=monotonic) -> None:
        self.limit, self.window, self._clock = limit, window_seconds, clock
        self._spent: deque[float] = deque()

    def take(self) -> bool:
        """Spend one retry if the window allows it."""
        now = self._clock()
        while self._spent and now - self._spent[0] >= self.window:
            self._spent.popleft()
        if len(self._spent) >= self.limit:
            return False
        self._spent.append(now)
        return True


def compute_delay(attempt: int, *, max_delay_seconds: float) -> float:
    """Calculate a capped exponential delay for the given attempt number."""
    if max_delay_seconds <= 0:
        return 0.0
    base_delay = min(RETRY_BASE_DELAY_SECONDS, max_delay_seconds)
    return float(min(max_delay_seconds, base_delay * (2**attempt)))


def build_retry_event(
    *,
    attempt: int,
    max_retries: int,
    delay_seconds: float,
    reason: str,
    data: dict[str, JValue] | None = None,
) -> SourceRetryEvent:
    """Assemble a ``SourceRetryEvent`` describing the upcoming reattempt."""
    next_attempt = attempt + 2
    max_attempts = max_retries + 1
    delay_suffix = f" in {delay_seconds:g}s" if delay_seconds else ""
    return SourceRetryEvent(
        attempt=next_attempt,
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        message=(
            f"Reattempting request {next_attempt}/{max_attempts} after {reason}{delay_suffix}."
        ),
        data=data,
    )


async def pause_for_retry(
    delay_seconds: float,
    *,
    signal: CancelToken | None,
) -> bool:
    """Wait out a backoff interval, returning False early if the signal fires."""
    if delay_seconds <= 0:
        return signal is None or not signal.is_cancelled()

    remaining = delay_seconds
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        step = min(RETRY_POLL_SECONDS, remaining)
        await sleep(step)
        remaining -= step
    return signal is None or not signal.is_cancelled()
