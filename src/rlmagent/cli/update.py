"""Upgrade an installed application using its package manager."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


def select_release(document: dict, channel: str) -> str:
    """Select a published, non-yanked release from the requested channel."""
    candidates = []
    current_python = ".".join(str(part) for part in sys.version_info[:3])
    for text, files in document.get("releases", {}).items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        eligible = {
            "stable": not version.is_prerelease and not version.is_devrelease,
            "preview": version.is_prerelease and not version.is_devrelease,
            "nightly": version.is_devrelease,
        }[channel]
        if not eligible:
            continue
        for artifact in files:
            if artifact.get("yanked", False):
                continue
            try:
                supported = SpecifierSet(artifact.get("requires_python") or "")
            except InvalidSpecifier:
                continue
            if supported.contains(current_python, prereleases=True):
                candidates.append(version)
                break
    if not candidates:
        raise ValueError(f"No compatible, non-yanked {channel} release is published")
    return str(max(candidates))


def run(argv: list[str]) -> int:
    if argv[:1] == ["--run"]:
        from rlmagent.managed_release import run as managed_run

        try:
            return managed_run(argv[1:])
        except (OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(prog="rlm-agent update")
    parser.add_argument("--manifest", type=Path, help="Verified offline wheel-bundle manifest")
    parser.add_argument(
        "--managed", action="store_true", help="Use isolated installations and atomic activation"
    )
    parser.add_argument("--drain", action="store_true", help="Let workers finish before activation")
    parser.add_argument("--wait", type=float, default=60)
    parser.add_argument("--restart-workers", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Run the upgrade command.")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--check", action="store_true", help="Compare against the package index.")
    choice.add_argument("--version", dest="target", help="Install a specific package version.")
    choice.add_argument(
        "--rollback", action="store_true", help="Reinstall the previously recorded version."
    )
    parser.add_argument("--channel", choices=("stable", "preview", "nightly"), default="stable")
    args = parser.parse_args(argv)
    if args.managed or args.manifest is not None:
        if args.target is not None or args.check or (args.manifest is not None and args.rollback):
            parser.error(
                "Use one managed manifest or --managed --rollback; omit --version and --check"
            )
        from rlmagent import managed_release

        try:
            if not args.apply:
                if args.manifest is not None:
                    print(json.dumps(managed_release.manifest(args.manifest), indent=2))
                print("Use --apply to install or roll back the reviewed managed release.")
                return 0
            if args.rollback:
                record = managed_release.rollback(
                    drain=args.drain, wait=args.wait, restart=args.restart_workers
                )
            elif args.manifest is not None:
                record = managed_release.install(
                    args.manifest, drain=args.drain, wait=args.wait, restart=args.restart_workers
                )
            else:
                raise ValueError("Managed installation needs --manifest, or --rollback")
            print(f"Active managed release: {record['version']} ({record['channel']})")
            return 0
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"Managed update failed: {error}", file=sys.stderr)
            return 1
    if args.drain or args.restart_workers:
        parser.error("Worker drain and restart require a managed installation")
    try:
        package = importlib.metadata.distribution("rlmness-cli")
    except importlib.metadata.PackageNotFoundError:
        parser.exit(1, "rlmagent must be installed to update it.\n")
    if package.read_text("direct_url.json"):
        parser.exit(1, "Update this local or URL installation from its original source.\n")

    from rlmagent.plugins import rlm_agent_home

    receipt = rlm_agent_home() / "updates" / "previous.json"
    target = args.target
    if args.rollback:
        try:
            target = json.loads(receipt.read_text(encoding="utf-8"))["version"]
        except (OSError, ValueError, KeyError):
            parser.exit(1, "No successful earlier upgrade has been recorded.\n")
    if target is not None and (
        not isinstance(target, str) or not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", target)
    ):
        parser.error("invalid package version")
    if args.check or (args.channel != "stable" and target is None):
        import os

        if os.environ.get("RLM_AGENT_OFFLINE") == "1":
            parser.exit(1, "Update checks are disabled in offline startup mode.\n")
        from rlmagent.model.transport.client import fetch_json

        try:
            latest = select_release(
                fetch_json("https://pypi.org/pypi/rlmness-cli/json", timeout=15), args.channel
            )
        except ValueError as error:
            parser.exit(1, str(error) + "\n")
        except Exception:
            parser.exit(1, "Could not check the package index.\n")
        if args.check:
            print(f"Installed rlmagent {package.version}; index {args.channel} release {latest}")
            return 0
        target = latest

    prefix = Path(sys.prefix)
    if (prefix / "uv-receipt.toml").is_file():
        command = ["uv", "tool", "upgrade", "rlmness-cli"]
    elif (prefix / "pipx_metadata.json").is_file():
        command = ["pipx", "upgrade", "rlmness-cli"]
    elif (package.read_text("INSTALLER") or "").strip() == "uv":
        command = ["uv", "pip", "install", "--python", sys.executable, "--upgrade", "rlmness-cli"]
    else:
        command = [sys.executable, "-m", "pip", "install", "--upgrade", "rlmness-cli"]

    if target is not None:
        spec = f"rlmness-cli=={target}"
        if command[:3] == ["uv", "tool", "upgrade"]:
            command = ["uv", "tool", "install", "--force", spec]
        elif command[:2] == ["pipx", "upgrade"]:
            command = ["pipx", "install", "--force", spec]
        else:
            command[-1] = spec
    if args.channel in ("preview", "nightly"):
        if command[0] != "pipx":
            command.extend(["--prerelease", "allow"] if command[0] == "uv" else ["--pre"])

    print(f"Installed rlmagent {package.version}", flush=True)
    if not args.apply:
        print("Run rlm-agent update --apply with the same options to install the requested update.")
        return 0
    executable = shutil.which(command[0])
    if executable is None:
        parser.exit(1, f"The installer {command[0]} is missing from PATH.\n")
    try:
        status = subprocess.run([executable, *command[1:]], check=False).returncode
        if status == 0:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            from uuid import uuid4

            staged = receipt.with_suffix(f".{uuid4().hex}.tmp")
            try:
                staged.write_text(json.dumps({"version": package.version}), encoding="utf-8")
                staged.replace(receipt)
            finally:
                staged.unlink(missing_ok=True)
        return status
    except OSError as exc:
        parser.exit(1, f"Could not start the installer: {exc}\n")
