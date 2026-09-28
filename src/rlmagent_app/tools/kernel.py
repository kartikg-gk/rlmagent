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
    if result.restarted and not result.timed_out:
        parts.append(
            "[The kernel had stopped and was restarted: variables from before are gone.]"
        )
    if result.output:
        parts.append(result.output.rstrip("\n"))
    if result.error:
        parts.append(result.error)
    if result.timed_out and result.restarted:
        parts.append(
            "[The cell timed out and ignored the interrupt, so the kernel was restarted: "
            "variables from before are gone.]"
        )
    elif result.timed_out:
        parts.append("[The cell timed out and was interrupted; state up to that point is kept.]")
    return "\n".join(parts) if parts else "(no output)"


_CANCEL_GRACE = 3.0


async def _cancel(kernel: KernelSession, task: asyncio.Task) -> str:
    """Stop a running cell for a cancel, and never wait long for it."""
    await kernel.interrupt()
    done, _ = await asyncio.wait({task}, timeout=_CANCEL_GRACE)
    if done:
        return format_cell(task.result()) + "\n[Cancelled by the user.]"
    await kernel.kill()
    await asyncio.gather(task, return_exceptions=True)
    return (
        "[Cancelled by the user. The cell ignored the interrupt, so the kernel was "
        "stopped: variables from before are gone.]"
    )


def make_python_tool(kernel: KernelSession) -> ToolSpec:
    async def run(
        tool_call_id: str,
        arguments: Mapping[str, JValue],
        signal: CancelToken | None = None,
        on_update: ProgressNotifier | None = None,
    ) -> ToolOutcome:
        code = str(arguments.get("code", ""))
        task = asyncio.create_task(kernel.run(code))
        try:
            while not task.done():
                if signal is not None and signal.is_cancelled():
                    return ToolOutcome(content=await _cancel(kernel, task))
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            await kernel.interrupt()
            raise
        return ToolOutcome(content=format_cell(await task))

    return ToolSpec(
        name="python",
        label="Python",
        description=_DESCRIPTION,
        parameters=_PARAMETERS,
        run=run,
        parallelism="sequential",
    )
