"""Provider-neutral content and transcript message models."""

from __future__ import annotations

from rlmagent_harness.contracts.transcript._base import StrictModel, now_ms
from rlmagent_harness.contracts.transcript.diagnostics import (
    CostBreakdown,
    FaultInfo,
    HaltReason,
    TurnDiagnostic,
    UsageStats,
)
from rlmagent_harness.contracts.transcript.entries import (
    ExtensionEntry,
    ForkSummaryEntry,
    HumanEntry,
    ModelEntry,
    PruneSummaryEntry,
    ShellResultEntry,
    ToolOutcomeEntry,
    TranscriptEntry,
    entry_to_human,
    surface_text,
)
from rlmagent_harness.contracts.transcript.segments import (
    CallBlock,
    ImageSegment,
    InputContent,
    ReplyContent,
    ResultContent,
    TextSegment,
    ThoughtSegment,
    build_model_blocks,
    gather_text,
)

__all__ = [
    # base
    "StrictModel",
    "now_ms",
    # segments
    "CallBlock",
    "ImageSegment",
    "InputContent",
    "ReplyContent",
    "ResultContent",
    "TextSegment",
    "ThoughtSegment",
    "build_model_blocks",
    "gather_text",
    # diagnostics
    "CostBreakdown",
    "FaultInfo",
    "HaltReason",
    "TurnDiagnostic",
    "UsageStats",
    # entries
    "ExtensionEntry",
    "ForkSummaryEntry",
    "HumanEntry",
    "ModelEntry",
    "PruneSummaryEntry",
    "ShellResultEntry",
    "ToolOutcomeEntry",
    "TranscriptEntry",
    "entry_to_human",
    "surface_text",
]
