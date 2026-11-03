"""`rlm-agent status` and `rlm-agent doctor`: what is set up, what is running, what is stale."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""  # what doctor --fix would do; empty when it cannot fix this


def _config() -> Check:
    from rlmagent.config.store import config_path, load_config

    path = config_path()
    if not path.exists():
        env = next((n for n in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                                "RLM_AGENT_API_KEY") if os.environ.get(n)), None)
        if env:
            return Check("provider", True, f"no saved configuration; {env} is set")
        return Check("provider", False, "no configuration; run: rlm-agent config")
    cfg = load_config()
    has_key = bool(cfg.api_keys.get(cfg.provider or "", ""))
    return Check("provider", bool(cfg.provider),
                 f"{cfg.provider or '(unset)'} / {cfg.model or '(unset)'}, "
                 f"key {'saved' if has_key else 'not saved (environment or none)'}")


def _sandbox() -> Check:
    try:
        from rlmness.wasm_runtime import wasm_available
    except ImportError:
        return Check("sandbox", False, "rlmness is not installed")
    if shutil.which("node") is None:
        return Check("sandbox", False, "Node.js 18+ is not on PATH; install it for --kernel wasm")
    if not wasm_available():
        return Check("sandbox", False, "Pyodide is not installed", "install Pyodide (rlmness --setup)")
    return Check("sandbox", True, "Node and Pyodide found")


def _git() -> Check:
    ok = shutil.which("git") is not None
    return Check("git", ok, "found; /undo and /diff work" if ok else "not on PATH; /undo and /diff are off")


def _workers() -> list[Check]:
    from rlmagent import worker

    base = worker.root()
    rows = []
    for path in sorted(base.iterdir()) if base.is_dir() else []:
        if not path.is_dir():
            continue
        meta = worker.state_of(path.name)
        recorded = meta.get("state", "?")
        if meta["alive"]:
            rows.append(Check(f"worker {path.name}", True, f"{recorded}, session {meta.get('session')}"))
        elif recorded in ("starting", "idle", "busy"):
            rows.append(Check(f"worker {path.name}", False,
                              f"recorded as {recorded} but its process is gone", "mark it stopped"))
        else:
            rows.append(Check(f"worker {path.name}", True, recorded))
    return rows


def _sessions_open() -> list[Check]:
    from rlmagent.agentnet import _home, _read
    from rlmagent.worker import _pid_alive

    base = _home()
    rows = []
    for path in sorted(base.iterdir()) if base.is_dir() else []:
        meta = _read(path / "meta.json")
        if meta is None:
            continue
        if meta.get("pid") and _pid_alive(int(meta["pid"])):
            rows.append(Check(f"session {path.name}", True, f"open in {meta.get('cwd')}"))
        else:
            rows.append(Check(f"session {path.name}", False, "registered but its process is gone",
                              "remove the registration"))
    return rows


def _mcp() -> Check:
    from rlmagent.mcp import load_configs
    from rlmagent.plugins import rlm_agent_home

    servers, problems = load_configs(rlm_agent_home(), None)
    if problems:
        return Check("mcp", False, "; ".join(problems))
    return Check("mcp", True, f"{len(servers)} server(s) in the home mcp.json")


def _search() -> Check:
    from rlmagent.websearch import SERVICES, configured

    name = configured()
    return Check("web search", True,
                 f"{name} is set" if name else f"off; set one of {', '.join(SERVICES)}")


def checks() -> list[Check]:
    return [_config(), _sandbox(), _git(), _mcp(), _search(), *_workers(), *_sessions_open()]


def _fix(check: Check) -> str:
    if check.name == "sandbox":
        from rlmness import home

        return "installed Pyodide" if home.setup() == 0 else "Pyodide install failed"
    if check.name.startswith("worker "):
        from rlmagent import worker

        name = check.name.split(" ", 1)[1]
        meta = worker.state_of(name)
        meta.update(state="stopped", pid=None)
        meta.pop("alive", None)
        worker._write(worker.folder(name) / "meta.json", meta)
        return "marked stopped"
    if check.name.startswith("session "):
        from rlmagent.agentnet import _home

        (_home() / check.name.split(" ", 1)[1] / "meta.json").unlink(missing_ok=True)
        return "removed"
    return "nothing done"


def run(argv: list[str], *, doctor: bool) -> int:
    from rlmagent.cli.main import get_version
    from rlmagent.plugins import rlm_agent_home

    fix = "--fix" in argv
    out = sys.stdout
    out.write(f"rlm-agent {get_version()}, home {rlm_agent_home()}\n")
    found = checks()
    problems = 0
    for check in found:
        mark = "ok " if check.ok else "!! "
        out.write(f"{mark}{check.name}: {check.detail}\n")
        if not check.ok:
            problems += 1
            if doctor and check.fix:
                out.write(f"    {'fixing: ' + _fix(check) if fix else 'doctor --fix would ' + check.fix}\n")
    if doctor:
        out.write("no problems found\n" if not problems else f"{problems} problem(s)\n")
    return 1 if doctor and problems and not fix else 0


__all__ = ["Check", "checks", "run"]
