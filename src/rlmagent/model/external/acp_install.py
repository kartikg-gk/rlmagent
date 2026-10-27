"""`rlm-agent acp install` / `acp status`: choose the bridge versions acp: runs use.

Choices are saved in ~/.rlm-agent/bridges.json as {"format": 1, "pins": {agent:
{"package", "version"}}}. A run launches exactly the saved version; `install -u`
moves to the newest one. To follow a bridge that changed its npm name, edit that
agent's "package" in the file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FORMAT = 1
DEFAULT_PACKAGES = {
    "claude-code": "@agentclientprotocol/claude-agent-acp",
    "codex": "@agentclientprotocol/codex-acp",
}
MISSING = ("acp:{agent} needs its bridge, which has not been set up. "
           "Run `rlm-agent acp install` (Node and npm must be on PATH).")


def pin_file() -> Path:
    from rlmagent.plugins import rlm_agent_home

    return rlm_agent_home() / "bridges.json"


def read_pins() -> dict[str, dict] | None:
    """Saved pins per agent, or None when install has not run (or the file is unreadable)."""
    try:
        saved = json.loads(pin_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or saved.get("format") != FORMAT:
        return None
    pins = saved.get("pins")
    return pins if isinstance(pins, dict) else None


def bridge_spec(agent: str) -> str:
    """What npx should launch for `agent`: package@version, or the bare package if unpinned."""
    pins = read_pins()
    if pins is None:
        raise ValueError(MISSING.format(agent=agent))
    pin = pins.get(agent) or {}
    package = pin.get("package") or DEFAULT_PACKAGES[agent]
    return f"{package}@{pin['version']}" if pin.get("version") else package


def _npm(*args: str) -> str | None:
    """stdout of a successful npm command; None when npm is missing or fails."""
    npm = shutil.which("npm")
    if npm is None:
        return None
    try:
        done = subprocess.run([npm, *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def install(update: bool = False) -> dict[str, dict]:
    """Pin every bridge (keeping saved versions unless `update`) and pre-fetch it."""
    saved = read_pins() or {}
    pins: dict[str, dict] = {}
    for agent, default in DEFAULT_PACKAGES.items():
        before = saved.get(agent) or {}
        package = before.get("package") or default
        kept = None if update else before.get("version")
        version = kept or _npm("view", package, "version") or before.get("version")
        pins[agent] = {"package": package, "version": version}
        if version is None:
            print(f"acp:{agent}: npm did not answer; {package} stays unpinned (newest at each launch)")
            continue
        fetched = _npm("cache", "add", f"{package}@{version}") is not None
        print(f"acp:{agent}: {package}@{version}" + ("" if fetched else " (not pre-fetched)"))
    pin_file().parent.mkdir(parents=True, exist_ok=True)
    pin_file().write_text(json.dumps({"format": FORMAT, "pins": pins}, indent=2), encoding="utf-8")
    print(f"Saved to {pin_file()}")
    if shutil.which("npx") is None:
        print("Warning: npx was not found; acp: runs start their bridge with it.")
    return pins


def status() -> int:
    pins = read_pins()
    if pins is None:
        print(f"No bridges set up ({pin_file()} missing). Run: rlm-agent acp install")
        return 1
    for agent, pin in pins.items():
        version = pin.get("version") or "unpinned"
        newest = _npm("view", pin.get("package", ""), "version")
        hint = f", newer {newest} (rlm-agent acp install -u)" if newest and newest != version else ""
        print(f"acp:{agent}: {pin.get('package')}@{version}{hint}")
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command == "install":
        install(update=bool({"-u", "--update"} & set(argv)))
        return 0
    if command == "status":
        return status()
    print("usage: rlm-agent acp install [-u] | rlm-agent acp status")
    return 2
