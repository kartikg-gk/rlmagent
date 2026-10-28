"""The python tool's kernel inside a WebAssembly sandbox (Pyodide, through rlmness).

Same surface as KernelSession, so the python tool and the agent tree do not care
which one they have. The sandbox has an empty filesystem, no network and no host
environment; host functions (bridges) are the only way out.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping

from rlmagent.kernel.session import CellResult, clip

Bridge = Callable[..., Awaitable[object]]

LOST = ("[The sandbox was replaced ({why}): variables from before are gone. "
        "Re-run any setup you still need.]")


def available() -> bool:
    try:
        from rlmness.wasm_runtime import wasm_available
    except ImportError:
        return False
    return wasm_available()


class WasmKernel:
    def __init__(
        self,
        cwd: str,
        bridges: Mapping[str, Bridge],
        startup_code: str = "",
        timeout: float = 300.0,
        on_final: Callable[[object], Awaitable[None]] | None = None,
        output_limit: int = 20_000,
    ) -> None:
        self.cwd = cwd
        self.env: dict[str, str] = {}
        self.bridges = dict(bridges)
        self.startup_code = startup_code
        self.timeout = timeout
        self.on_final = on_final
        self.output_limit = output_limit
        self._runtime = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    @contextlib.contextmanager
    def waiting_outside(self):
        # The sandbox does not time a cell out while one of its bridges is running.
        yield

    @property
    def started(self) -> bool:
        return self._runtime is not None

    def _blocking(self, bridge: Bridge) -> Callable[..., object]:
        """rlmness serves bridges on worker threads; the work belongs on this agent's loop."""
        def call(*args, **kwargs):
            future = asyncio.run_coroutine_threadsafe(bridge(*args, **kwargs), self._loop)
            return future.result()
        return call

    async def _start(self) -> str | None:
        from rlmness.wasm_runtime import WasmRuntime

        self._loop = asyncio.get_running_loop()
        bridges = {name: self._blocking(fn) for name, fn in self.bridges.items()}
        self._runtime = await asyncio.to_thread(WasmRuntime, None, bridges, self.timeout)
        if self.startup_code:
            outcome = await asyncio.to_thread(self._runtime.execute, self.startup_code)
            if outcome.error:
                return outcome.error
        return None

    async def _drop(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            await asyncio.to_thread(runtime.close)

    async def run(self, code: str, timeout: float | None = None) -> CellResult:
        from rlmness.runtime import CellTimeout, RuntimeGone

        async with self._lock:
            if self._runtime is None:
                failure = await self._start()
                if failure:
                    return CellResult(output="", error=f"sandbox start-up failed:\n{failure}")
            try:
                outcome = await asyncio.to_thread(self._runtime.execute, code)
            except CellTimeout:
                # A cell that never ends keeps the shared Node host busy, which stalls every
                # sandbox in it; restarting the host is the only way to stop it.
                from rlmness.wasm_runtime import reset_host

                self._runtime = None
                await asyncio.to_thread(reset_host)
                return CellResult(output=LOST.format(why=f"no answer within {self.timeout:.0f}s; "
                                                     "every agent's sandbox was restarted"),
                                  timed_out=True, restarted=True)
            except RuntimeGone:
                await self._drop()
                return CellResult(output=LOST.format(why="it stopped"), restarted=True)
        output = outcome.stdout
        if outcome.final_given and self.on_final is not None:
            await self.on_final(outcome.final)
            output += "FINAL recorded.\n"
        return CellResult(output=clip(output, self.output_limit),
                          error=clip(outcome.error, self.output_limit) if outcome.error else None)

    async def interrupt(self) -> None:
        # Pyodide cannot be interrupted from outside; stopping means replacing it.
        await self._drop()

    async def kill(self) -> None:
        await self._drop()

    async def shutdown(self) -> None:
        await self._drop()
