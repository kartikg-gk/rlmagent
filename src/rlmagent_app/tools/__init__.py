"""Coding tools (files, the python kernel) + tool registry/assembly."""

from __future__ import annotations

from rlmagent_app.tools._shared import (
    CallBlock,
    CancelToken,
    InputShaper,
    InvocationPrinter,
    JValue,
    OutcomePresenter,
    Parallelism,
    ProgressNotifier,
    RunHandler,
    ToolOutcome,
    ToolSpec,
)
from rlmagent_app.tools.files import file_tools
from rlmagent_app.tools.kernel import make_python_tool


def build_tool_registry(kernel=None) -> list[ToolSpec]:
    """File tools, plus the python tool when the agent has a kernel."""
    tools: list[ToolSpec] = []
    tools.extend(file_tools())
    if kernel is not None:
        tools.append(make_python_tool(kernel))
    return tools


__all__ = [
    "CallBlock",
    "CancelToken",
    "InputShaper",
    "InvocationPrinter",
    "JValue",
    "OutcomePresenter",
    "Parallelism",
    "ProgressNotifier",
    "RunHandler",
    "ToolOutcome",
    "ToolSpec",
    "build_tool_registry",
    "file_tools",
    "make_python_tool",
]
