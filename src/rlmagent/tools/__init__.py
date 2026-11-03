"""Coding tools (files, the python kernel) + tool registry/assembly."""

from __future__ import annotations

from rlmagent.tools._shared import (
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
from rlmagent.tools.files import file_tools
from rlmagent.tools.kernel import make_python_tool
from rlmagent.tools.search import search_tools


def build_tool_registry(kernel=None) -> list[ToolSpec]:
    """File tools, plus the python tool when the agent has a kernel."""
    from rlmagent.kernel.wasm import WasmKernel

    tools: list[ToolSpec] = []
    # A sandboxed agent changes files only through its approved workspace functions.
    sandboxed = isinstance(kernel, WasmKernel) or getattr(kernel, "kernel_kind", None) == "wasm"
    tools.extend(t for t in file_tools() if not sandboxed or t.name == "Read")
    tools.extend(search_tools())
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
    "search_tools",
]
