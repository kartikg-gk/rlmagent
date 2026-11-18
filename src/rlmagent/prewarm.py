"""Start the local kernel process at launch so it is ready when the first cell runs.

The kernel is a separate Python process (rlmness's cell runner). Starting it costs
about a quarter of a second; started here, that happens while rlm-agent imports and
builds its session instead of after. Only what is cheap to import is used at module
level, because this runs before the rest of the application loads.
"""

from __future__ import annotations

import atexit
import functools
import os
import subprocess
import sys
import threading

_lock = threading.Lock()
_ready: subprocess.Popen | None = None


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


def start() -> None:
    """Launch one kernel process in the background, at most once per run."""
    global _ready

    def launch():
        global _ready
        from rlmness.environment import execution_environment
        from rlmness.runtime import _RUNNER

        environment = execution_environment({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        process = subprocess.Popen(
            [sys.executable, "-I", _RUNNER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, env=environment,
        )
        atexit.register(_stop, process)
        with _lock:
            _ready = process

    with _lock:
        if _ready is not None:
            return
    threading.Thread(target=launch, name="kernel-prewarm", daemon=True).start()


def take() -> subprocess.Popen | None:
    """The started process, once, if it is still alive."""
    global _ready
    with _lock:
        process, _ready = _ready, None
    if process is not None and process.poll() is None:
        return process
    return None


def wanted(args: list[str]) -> bool:
    """A one-shot run on the local kernel: the only case where the start is on the clock."""
    if not ({"-p", "--print"} & set(args)) or "--sandbox" in args:
        return False
    if "--kernel" in args:
        index = args.index("--kernel")
        return index + 1 < len(args) and args[index + 1] == "local"
    if "--no-sandbox" in args:
        return True
    env = os.environ.get("RLM_AGENT_KERNEL", "").strip().lower()
    if env in ("wasm", "local"):
        return env == "local"
    return _saved_kernel() != "wasm"


def _saved_kernel() -> str | None:
    # Read config.toml directly: importing the config package loads far more.
    import tomllib
    from pathlib import Path

    home = os.environ.get("RLM_AGENT_CONFIG_DIR") or os.environ.get("RLM_AGENT_HOME")
    path = (Path(home) if home else Path.home() / ".rlm-agent") / "config.toml"
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh).get("kernel")
    except (OSError, ValueError):
        return None


@functools.cache
def runtime_class():
    """rlmness's subprocess runtime, adopting the started process when there is one."""
    from rlmness.runtime import ProcessChannel, ProtocolRuntime, SubprocessRuntime

    class WarmSubprocessRuntime(SubprocessRuntime):
        def __init__(self, prompt, bridges=(), timeout=120.0, tools=(), session=None,
                     batch_only=None):
            process = take()
            if process is None:
                super().__init__(prompt, bridges, timeout, tools, session, batch_only)
                return
            ProtocolRuntime.__init__(self, ProcessChannel(process), prompt, bridges, timeout,
                                     tools, session, batch_only)

    return WarmSubprocessRuntime
