"""One IPython kernel, started on first use, that keeps its variables between cells."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from jupyter_client.manager import AsyncKernelManager

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_POLL = 0.5
_STUCK = "__still_running_after_interrupt__"


@dataclass
class CellResult:
    output: str
    error: str | None = None
    timed_out: bool = False
    restarted: bool = False


def clip(text: str, limit: int) -> str:
    """Keep the start and the end of long output, and say how much was dropped."""
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n... [{dropped:,} characters omitted] ...\n{text[-half:]}"


class KernelSession:
    def __init__(
        self,
        cwd: str,
        env: Mapping[str, str] | None = None,
        startup_code: str = "",
        timeout: float = 300.0,
        output_limit: int = 20_000,
        interrupt_grace: float = 10.0,
    ) -> None:
        self.cwd = cwd
        self.env = dict(env or {})
        self.startup_code = startup_code
        self.timeout = timeout
        self.output_limit = output_limit
        self.interrupt_grace = interrupt_grace
        self._waiting_outside = 0
        self._manager: AsyncKernelManager | None = None
        self._client = None
        self._lock = asyncio.Lock()

    @contextlib.contextmanager
    def waiting_outside(self):
        """Mark the running cell as waiting on work done elsewhere.

        The cell timeout is for code that runs too long. A cell awaiting a
        sub-agent is idle while the sub-agent works, which is bounded by the
        budget instead; its clock starts again once the wait ends.
        """
        self._waiting_outside += 1
        try:
            yield
        finally:
            self._waiting_outside -= 1

    @property
    def started(self) -> bool:
        return self._manager is not None

    async def _start(self) -> str | None:
        """Start the kernel; return the setup code's error, if any."""
        manager = AsyncKernelManager(kernel_name="python3")
        try:
            await manager.start_kernel(
                cwd=self.cwd,
                env={**os.environ, **self.env},
                extra_arguments=["--IPKernelApp.log_level=ERROR"],
            )
            client = manager.client()
            client.start_channels()
            await client.wait_for_ready(timeout=60)
        except Exception:
            try:
                await manager.shutdown_kernel(now=True)
            except Exception:
                pass
            raise
        self._manager, self._client = manager, client
        if self.startup_code:
            return (await self._execute(self.startup_code, self.timeout)).error
        return None

    async def _alive(self) -> bool:
        return self._manager is not None and await self._manager.is_alive()

    async def _restart(self) -> str | None:
        await self.shutdown()
        return await self._start()

    async def run(self, code: str, timeout: float | None = None) -> CellResult:
        async with self._lock:
            restarted = False
            try:
                if self._manager is None:
                    setup_error = await self._start()
                elif not await self._alive():
                    setup_error = await self._restart()
                    restarted = True
                else:
                    setup_error = None
            except Exception as exc:
                return CellResult(output="", error=f"The kernel could not start: {exc}")
            if setup_error:
                return CellResult(output="", error=f"Kernel setup failed:\n{setup_error}")
            result = await self._execute(code, timeout or self.timeout)
            if result.error == _STUCK:
                # Ignored the interrupt (a blocking call can): a fresh kernel beats a stuck one.
                await self._restart()
                result.error = None
                restarted = True
            if result.error == "__kernel_died__":
                # The cell itself killed the kernel; the next cell restarts it.
                result.error = "The kernel process exited while running this cell."
            result.restarted = restarted
            result.output = clip(result.output, self.output_limit)
            return result

    async def _execute(self, code: str, timeout: float) -> CellResult:
        client = self._client
        msg_id = client.execute(code, store_history=True, allow_stdin=False)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        parts: list[str] = []
        error: str | None = None
        timed_out = False
        while True:
            if self._waiting_outside and not timed_out:
                deadline = loop.time() + timeout
            remaining = deadline - loop.time()
            if remaining <= 0 and not timed_out:
                timed_out = True
                await self._manager.interrupt_kernel()
                deadline = loop.time() + self.interrupt_grace
                continue
            if remaining <= 0:
                return CellResult(output="".join(parts), error=_STUCK, timed_out=True)
            try:
                msg = await asyncio.wait_for(client.get_iopub_msg(), min(_POLL, remaining))
            except (asyncio.TimeoutError, TimeoutError):
                if not await self._alive():
                    return CellResult(output="".join(parts), error="__kernel_died__")
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "stream":
                parts.append(content["text"])
            elif kind in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain")
                if text:
                    parts.append(text + "\n")
            elif kind == "error":
                error = _ANSI.sub("", "\n".join(content.get("traceback", [])))
            elif kind == "status" and content.get("execution_state") == "idle":
                break
        if timed_out and error and "KeyboardInterrupt" in error:
            error = None
        return CellResult(output="".join(parts), error=error, timed_out=timed_out)

    async def kill(self) -> None:
        """Stop the kernel process now, even while a cell holds the lock.

        The running cell then ends as "kernel died", and the next cell starts
        a fresh kernel and says so.
        """
        manager = self._manager
        if manager is not None:
            try:
                await manager.shutdown_kernel(now=True)
            except Exception:
                pass

    async def interrupt(self) -> None:
        if self._manager is not None:
            await self._manager.interrupt_kernel()

    async def shutdown(self) -> None:
        client, manager = self._client, self._manager
        self._client = self._manager = None
        if client is not None:
            client.stop_channels()
        if manager is not None:
            try:
                await manager.shutdown_kernel(now=True)
            except Exception:
                pass
