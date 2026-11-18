"""One IPython kernel, started on first use, that keeps its variables between cells."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from rlmness.environment import execution_environment

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_POLL = 0.5
_STUCK = "__still_running_after_interrupt__"
_UNAVAILABLE = (
    "The kernel is unavailable: it stopped again after {} restarts in a row and will "
    "not be restarted. Finish the task without the python tool."
)


# Runs inside the kernel after each finished cell: keep what can be pickled.
_SAVE = r"""
def __rlm_save(path, owned, limit):
    import pickle, sys, types
    try:
        import dill as pickler
    except ImportError:
        pickler = pickle
    ns = sys.modules["__main__"].__dict__
    values, modules, sources, dropped = {}, {}, {}, []
    for name, value in list(ns.items()):
        if name.startswith("_") or name in owned or name in ("In", "Out", "exit", "quit", "get_ipython"):
            continue
        if isinstance(value, types.ModuleType):
            modules[name] = value.__name__
            continue
        if (pickler is pickle and isinstance(value, (types.FunctionType, type))
                and getattr(value, "__module__", None) == "__main__"):
            try:
                import inspect
                sources[name] = inspect.getsource(value)
            except Exception:
                dropped.append(name)
            continue
        try:
            blob = pickler.dumps(value)
        except Exception:
            dropped.append(name)
            continue
        if len(blob) > limit:
            dropped.append(name)
            continue
        values[name] = blob
    with open(path, "wb") as fh:
        pickle.dump({"values": values, "modules": modules, "sources": sources,
                     "dropped": dropped}, fh)
"""

_LOAD = r"""
def __rlm_load(path):
    import importlib, pickle, sys
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    try:
        import dill as pickler
    except ImportError:
        pickler = pickle
    ns = sys.modules["__main__"].__dict__
    restored, missing = [], list(saved["dropped"])
    for name, module in saved["modules"].items():
        try:
            ns[name] = importlib.import_module(module)
            restored.append(name)
        except Exception:
            missing.append(name)
    for name, source in saved["sources"].items():
        try:
            exec(compile(source, "<restored " + name + ">", "exec"), ns)
            restored.append(name)
        except Exception:
            missing.append(name)
    for name, blob in saved["values"].items():
        try:
            ns[name] = pickler.loads(blob)
            restored.append(name)
        except Exception:
            missing.append(name)
    print("[Kernel checkpoint restored: " + (", ".join(sorted(restored)) or "no values")
          + "; unavailable: " + (", ".join(sorted(missing)) or "none")
          + ". Changes in the interrupted cell were not recovered.]")
"""


def _remove_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def __getattr__(name: str):
    # jupyter_client (with zmq, tornado, traitlets) costs ~0.1s to import and only the
    # session engine's local kernel needs it, so it loads on first use.
    if name == "AsyncKernelManager":
        from jupyter_client.manager import AsyncKernelManager

        globals()[name] = AsyncKernelManager
        return AsyncKernelManager
    raise AttributeError(name)


def _manager_class():
    return globals().get("AsyncKernelManager") or __getattr__("AsyncKernelManager")


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
        max_restarts: int = 3,
        restore: bool = False,
        restore_limit: int = 50_000_000,
    ) -> None:
        self.cwd = cwd
        self.env = dict(env or {})
        self.startup_code = startup_code
        self.timeout = timeout
        self.output_limit = output_limit
        self.interrupt_grace = interrupt_grace
        self.max_restarts = max_restarts
        self._restarts_in_row = 0
        self._waiting_outside = 0
        self._manager = None
        self._client = None
        self._lock = asyncio.Lock()
        self.restore = restore
        self.restore_limit = restore_limit
        self._checkpoint: str | None = None
        self._owned: list[str] | None = None

    @contextlib.contextmanager
    def waiting_outside(self):
        """Pause the cell timeout while the cell awaits sub-agents."""
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
        manager = _manager_class()(kernel_name="python3")
        try:
            await manager.start_kernel(
                cwd=self.cwd,
                env=execution_environment(self.env),
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
            error = (await self._execute(self.startup_code, self.timeout)).error
            if error:
                return error
        if self.restore and self._owned is None:
            names = await self._execute(
                "print(','.join(n for n in dir() if not n.startswith('_')))", self.timeout)
            self._owned = [n for n in names.output.strip().split(",") if n]
        return None

    async def _save_checkpoint(self) -> None:
        """Pickle the namespace after a finished cell so a restarted kernel can get it back."""
        import tempfile

        if self._checkpoint is None:
            handle, self._checkpoint = tempfile.mkstemp(prefix="rlm-kernel-", suffix=".pkl")
            os.close(handle)
            import atexit

            atexit.register(_remove_quietly, self._checkpoint)
        code =(_SAVE + f"\n__rlm_save({self._checkpoint!r}, {self._owned or []!r}, "
                f"{self.restore_limit})\ndel __rlm_save")
        await self._execute(code, self.timeout)

    async def _load_checkpoint(self) -> str:
        """Put the last checkpoint back into a fresh kernel; return its note."""
        if not (self._checkpoint and os.path.getsize(self._checkpoint)):
            return ""
        code = _LOAD + f"\n__rlm_load({self._checkpoint!r})\ndel __rlm_load"
        result = await self._execute(code, self.timeout)
        return result.output if not result.error else ""

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
                    if self._restarts_in_row >= self.max_restarts:
                        return CellResult(output="", error=_UNAVAILABLE.format(self._restarts_in_row))
                    setup_error = await self._restart()
                    self._restarts_in_row += 1
                    restarted = True
                else:
                    setup_error = None
            except Exception as exc:
                return CellResult(output="", error=f"The kernel could not start: {exc}")
            if setup_error:
                return CellResult(output="", error=f"Kernel setup failed:\n{setup_error}")
            note = await self._load_checkpoint() if restarted and self.restore else ""
            result = await self._execute(code, timeout or self.timeout)
            if result.error == _STUCK:
                # Ignored the interrupt (a blocking call can): a fresh kernel beats a stuck one.
                await self._restart()
                self._restarts_in_row += 1
                result.error = None
                restarted = True
            elif result.error == "__kernel_died__":
                result.error = "The kernel process exited while running this cell."
            else:
                self._restarts_in_row = 0
            result.restarted = restarted
            if self.restore and not result.timed_out and result.error is None:
                await self._save_checkpoint()
            result.output = clip(note + result.output, self.output_limit)
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
        """Stop the kernel now, even mid-cell; the next cell gets a fresh one."""
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
        """Stop the kernel. The checkpoint file stays until close() so a restart can use it."""
        client, manager = self._client, self._manager
        self._client = self._manager = None
        if client is not None:
            client.stop_channels()
        if manager is not None:
            try:
                await manager.shutdown_kernel(now=True)
            except Exception:
                pass
