"""rlmagent.harness: the core."""

from __future__ import annotations

from rlmagent.harness.contracts.stream import (
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RunEndEvent,
    RunStartEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
    ToolRunUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from rlmagent.harness.contracts.tooling import (
    CancelToken,
    RunHandler,
    ToolOutcome,
    ToolSpec,
)
from rlmagent.harness.contracts.transcript import (
    CallBlock,
    HumanEntry,
    ModelEntry,
    TextSegment,
    ToolOutcomeEntry,
    TranscriptEntry,
)
from rlmagent.harness.contracts.values import JObject, JPrimitive, JValue

__all__ = [
    # values
    "JObject",
    "JPrimitive",
    "JValue",
    # transcript
    "CallBlock",
    "HumanEntry",
    "ModelEntry",
    "TextSegment",
    "ToolOutcomeEntry",
    "TranscriptEntry",
    # tooling
    "CancelToken",
    "RunHandler",
    "ToolOutcome",
    "ToolSpec",
    # stream
    "AgentEvent",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "RunEndEvent",
    "RunStartEvent",
    "ToolRunEndEvent",
    "ToolRunStartEvent",
    "ToolRunUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
