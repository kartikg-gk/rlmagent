"""`rlm-agent acp install`: pin the ACP bridge versions once, so runs are reproducible.

The versions live in a marker file (~/.rlm-agent/acp.json), not in code: `install -u`
picks up a newer bridge with no rlm-agent release, and a plain `install` re-caches
the recorded versions without upgrading them. A renamed bridge is followed by
editing `bridge_packages` in the marker.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

MARKER_VERSION = 1

BRIDGE_PKGS = {
    "claude-code": "@agentclientprotocol/claude-agent-acp",
    "codex": "@agentclientprotocol/codex-acp",
}

NOT_INSTALLED = ("ACP support is not installed. Run `rlm-agent acp install` once before using "
                 "acp:{agent} (needs Node/npm on PATH).")


def marker_path() -> Path:
    from rlmagent.plugins import rlm_agent_home

    return rlm_agent_home() / "acp.json"


def load_marker() -> dict | None:
    try:
        data = json.loads(marker_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("marker_version") != MARKER_VERSION:
        return None
    return data


def bridge_spec(agent: str) -> str:
    """`pkg@version` for a preset's bridge; raises the not-installed error."""
    marker = load_marker()
    if marker is None:
        raise ValueError(NOT_INSTALLED.format(agent=agent))
    pkg = {**BRIDGE_PKGS, **(marker.get("bridge_packages") or {})}[agent]
    version = (marker.get("bridges") or {}).get(agent)
    return f"{pkg}@{version}" if version else pkg


def _npm(*args: str) -> str | None:
    npm = shutil.which("npm")
    if npm is None:
        return None
    try:
        out = subprocess.run([npm, *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def install(update: bool = False) -> dict:
    existing = load_marker() or {}
    old = existing.get("bridges") or {}
    pkgs = {**BRIDGE_PKGS, **(existing.get("bridge_packages") or {})}
    bridges = {}
    print("Resolving ACP bridge versions...")
    for name, pkg in pkgs.items():
        version = old.get(name) if not update else None
        version = version or _npm("view", pkg, "version") or old.get(name)
        if version is None:
            print(f"  ! could not reach npm for {pkg}; acp:{name} will resolve to latest at launch")
            continue
        bridges[name] = version
        print(f"  {pkg}@{version}  (acp:{name})")
    print("Downloading into npm's cache...")
    for name, version in bridges.items():
        if _npm("cache", "add", f"{pkgs[name]}@{version}") is None:
            print(f"  ! could not cache {pkgs[name]}@{version}; npx will fetch it on first use")
    marker = {"marker_version": MARKER_VERSION, "bridges": bridges, "bridge_packages": pkgs}
    marker_path().parent.mkdir(parents=True, exist_ok=True)
    marker_path().write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"\nACP support installed. Marker: {marker_path()}")
    if shutil.which("npx") is None:
        print("  ! npx is not on PATH; the claude-code and codex bridges need it.")
    return marker


def status() -> int:
    marker = load_marker()
    if marker is None:
        print("ACP support: not installed")
        print(f"  marker: {marker_path()} (absent)")
        print("  install with: rlm-agent acp install")
        return 1
    print("ACP support: installed")
    print(f"  marker: {marker_path()}")
    pkgs = {**BRIDGE_PKGS, **(marker.get("bridge_packages") or {})}
    for name, version in (marker.get("bridges") or {}).items():
        latest = _npm("view", pkgs[name], "version")
        note = f"  (latest {latest}; upgrade with: rlm-agent acp install -u)" if latest and latest != version else ""
        print(f"  {pkgs[name]}@{version}{note}")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["install"]:
        install(update="-u" in argv or "--update" in argv)
        return 0
    if argv[:1] == ["status"]:
        return status()
    print("usage: rlm-agent acp install [-u] | rlm-agent acp status")
    return 2
