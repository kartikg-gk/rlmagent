"""A counted request with a provider-enforced single-attempt reply cap."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass


class TokenBoundError(ValueError):
    """The provider cannot establish a bound before generation."""


@dataclass(frozen=True)
class TokenReservation:
    input_tokens: int
    output_tokens: int
    start: Callable[[], AsyncIterator]
