"""HTTP transport for the Anthropic Messages API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rlmagent.harness.contracts.tooling import CancelToken

# ---------------------------------------------------------------------------
# Retriable status codes
# ---------------------------------------------------------------------------

_TRANSIENT_CODES = frozenset({408, 409, 425, 429})


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ApiRejection(Exception):
    """Non-200 HTTP response from the Anthropic Messages API."""

    def __init__(
        self,
        status: int,
        detail: str,
        *,
        wait_hint: float | None = None,
    ) -> None:
        self.status = status
        self.detail = detail
        self.wait_hint = wait_hint
        super().__init__(f"HTTP {status}")

    @property
    def retriable(self) -> bool:
        """Return whether this status code warrants an automatic reattempt."""
        return self.status in _TRANSIENT_CODES or self.status >= 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _read_wait_hint(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Streaming transport
# ---------------------------------------------------------------------------


async def relay_sse(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    *,
    signal: CancelToken | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """POST *body* as streaming JSON and yield ``(event_type, payload)`` pairs."""
    async with client.stream("POST", url, json=body, headers=headers) as response:
        if response.status_code != 200:
            raw = (await response.aread()).decode("utf-8", errors="replace")
            raise ApiRejection(
                response.status_code,
                raw,
                wait_hint=_read_wait_hint(response),
            )

        event_name = ""
        data_parts: list[str] = []

        async for raw_line in response.aiter_lines():
            if signal is not None and signal.is_cancelled():
                return

            line = raw_line.rstrip("\r\n")

            if not line:
                # Empty line fires the accumulated event
                if data_parts:
                    payload = _parse_object("\n".join(data_parts))
                    if payload is not None:
                        yield (event_name, payload)
                    event_name = ""
                    data_parts = []
                continue

            if line.startswith(":"):
                continue  # comment / keep-alive

            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_parts.append(line[5:].lstrip())
