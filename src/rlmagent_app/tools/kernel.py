"""The python tool: run a cell in this agent's persistent kernel."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from rlmagent_app.kernel import CellResult, KernelSession
from rlmagent_app.tools._shared import (
    CancelToken,
    JValue,
    ProgressNotifier,
    ToolOutcome,
    ToolSpec,
)

_DESCRIPTION = (
    "Run Python in a persistent IPython kernel whose working directory is the "
    "project root. Variables, imports and functions stay defined between calls. "
    "Run shell commands with a leading `!` or with subprocess. Top-level `await` works."
)

_PARAMETERS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {"code": {"type": "string", "description": "Python code to run."}},
    "required": ["code"],
}


def format_cell(result: CellResult) -> str:
    parts: list[str] = []
    if result.restarted:
        parts.append(
            "[The kernel had stopped and was restarted: variables from before are gone.]"
        )
    if result.output:
        parts.append(result.output.rstrip("\n"))
    if result.error:
        parts.append(result.error)
    if result.timed_out:
        parts.append("[The cell timed out and was interrupted; state up to that point is kept.]")
    return "\n".join(parts) if parts else "(no output)"


def make_python_tool(kernel: KernelSession) -> ToolSpec:
    async def run(
        tool_call_id: str,
        arguments: Mapping[str, JValue],
        signal: CancelToken | None = None,
        on_update: ProgressNotifier | None = None,
    ) -> ToolOutcome:
        code = str(arguments.get("code", ""))
        task = asyncio.create_task(kernel.run(code))
        while not task.done():
            if signal is not None and signal.is_cancelled():
                await kernel.interrupt()
                break
            await asyncio.sleep(0.2)
        result = await task
        return ToolOutcome(content=format_cell(result))

    return ToolSpec(
        name="python",
        label="Python",
        description=_DESCRIPTION,
        parameters=_PARAMETERS,
        run=run,
        parallelism="sequential",
    )
