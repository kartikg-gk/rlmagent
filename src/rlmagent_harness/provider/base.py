"""Structural protocol for model adapters consumed by the agent loop."""
#provider
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from rlmagent_harness.contracts.tooling import CancelToken, ToolSpec
from rlmagent_harness.contracts.transcript import TranscriptEntry
from rlmagent_harness.provider.wire import WireEvent


class ModelProvider(Protocol):
    """Structural interface for anything that can produce model completions."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        signal: CancelToken | None = None,
        cache_key: str | None = None,
    ) -> AsyncIterator[WireEvent]:
        """Stream typed wire events for one model round-trip."""
        ...
