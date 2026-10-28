"""Inference-route pinning for the Hugging Face router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ROUTED_PROVIDER = "huggingface"

RouteMode = Literal["automatic", "fixed"]


@dataclass(frozen=True, slots=True)
class RoutePin:
    """Which backend requests are sent to, and whether it may change."""

    mode: RouteMode = "automatic"
    backend: str | None = None

    def model_for(self, model: str) -> str:
        """The model id to send: suffixed with the backend when pinned."""
        return f"{model}:{self.backend}" if self.backend else model

    def describe(self) -> str:
        if self.mode == "fixed" and self.backend:
            return f"{self.backend} (fixed)"
        if self.backend:
            return f"automatic (currently {self.backend})"
        return "automatic"


def pin_from_storage(backend: str | None, mode: str | None) -> RoutePin:
    """Rebuild a pin from saved metadata."""
    if mode in ("automatic", "fixed"):
        return RoutePin(mode, backend)  # type: ignore[arg-type]
    return RoutePin("fixed", backend) if backend else RoutePin()


def is_route_failure(status: int | None) -> bool:
    """Statuses worth abandoning a learned backend for (after its own retries)."""
    if status is None:
        return False
    return status in (408, 409, 425, 429) or status >= 500


__all__ = [
    "ROUTED_PROVIDER",
    "RouteMode",
    "RoutePin",
    "is_route_failure",
    "pin_from_storage",
]
