"""Upgrade an installed application using its package manager."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rlm-agent update")
    parser.add_argument("--apply", action="store_true", help="Run the upgrade command.")
    args = parser.parse_args(argv)
    try:
        package = importlib.metadata.distribution("rlmagent")
    except importlib.metadata.PackageNotFoundError:
        parser.exit(1, "rlmagent must be installed to update it.\n")
    if package.read_text("direct_url.json"):
        parser.exit(1, "Update this local or URL installation from its original source.\n")

    prefix = Path(sys.prefix)
    if (prefix / "uv-receipt.toml").is_file():
        command = ["uv", "tool", "upgrade", "rlmagent"]
    elif (prefix / "pipx_metadata.json").is_file():
        command = ["pipx", "upgrade", "rlmagent"]
    elif (package.read_text("INSTALLER") or "").strip() == "uv":
        command = ["uv", "pip", "install", "--python", sys.executable, "--upgrade", "rlmagent"]
    else:
        command = [sys.executable, "-m", "pip", "install", "--upgrade", "rlmagent"]

    print(f"Installed rlmagent {package.version}", flush=True)
    if not args.apply:
        print("Run rlm-agent update --apply to upgrade.")
        return 0
    executable = shutil.which(command[0])
    if executable is None:
        parser.exit(1, f"The installer {command[0]} is missing from PATH.\n")
    try:
        return subprocess.run([executable, *command[1:]], check=False).returncode
    except OSError as exc:
        parser.exit(1, f"Could not start the installer: {exc}\n")
