"""Explicitly installed capability packages and their resource directories."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from rlmagent.refinement.store import _lock

RETAIN_DAYS = 30


def prune(store: Path, records: dict, *, older_than=RETAIN_DAYS, apply=False) -> list[str]:
    """Collect only expired inactive installs directly owned by this store."""
    active = {entry["directory"] for entry in records.values()}
    cutoff = time.time() - older_than * 86400
    selected = []
    if not store.exists():
        return selected
    for candidate in store.iterdir():
        if candidate.name in active or not re.fullmatch(r"[a-f0-9]{32}", candidate.name):
            continue
        if candidate.is_symlink() or candidate.is_junction() or not candidate.is_dir():
            continue
        resolved = _location(candidate.name, store)
        if str(resolved) in sys.path or candidate.stat().st_mtime >= cutoff:
            continue
        selected.append(candidate.name)
        if apply:
            shutil.rmtree(resolved)
    return sorted(selected)


def root(*, local: bool = False, paths=None) -> Path:
    from rlmagent.discovery import _resolve_home

    if local:
        return (paths.project if paths else Path.cwd()) / ".rlm-agent" / "packages"
    return (paths.home if paths else _resolve_home()) / "packages"


def _records(store: Path | None = None) -> dict:
    store = store or root()
    try:
        entries = json.loads((store / "installed.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(entries, dict):
        raise ValueError("Invalid installed-package registry")
    for name, value in entries.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) or not isinstance(value, dict):
            raise ValueError("Invalid installed-package registry")
        _location(value["directory"], store)
    return entries


def _installer(target: Path, source: str) -> list[str]:
    """The command that installs `source` into `target`.

    Environments made by uv (including `uv tool install`) carry no pip, so uv
    installs there for this interpreter; pip is used wherever it exists.
    """
    import importlib.util

    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--target", str(target), source]
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Installing a package needs pip or uv in this environment")
    return [uv, "pip", "install", "--python", sys.executable, "--target", str(target), source]


def _location(directory: str, store: Path | None = None) -> Path:
    store = store or root()
    if not isinstance(directory, str) or not re.fullmatch(r"[a-f0-9]{32}", directory):
        raise ValueError("Invalid package directory")
    path = (store / directory).resolve()
    if not path.is_relative_to(store.resolve()):
        raise ValueError("Package directory is outside the package store")
    return path


def _save(records: dict, store: Path | None = None) -> None:
    target = (store or root()) / "installed.json"
    temporary = target.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def activate() -> None:
    from rlmagent.discovery import default_paths

    paths = default_paths()
    stores = [root(paths=paths)]
    if paths.project_enabled:
        stores.insert(0, root(local=True, paths=paths))
    for store in stores:
        for record in _records(store).values():
            directory = str(_location(record["directory"], store))
            if directory not in sys.path:
                sys.path.append(directory)


def resource_dirs(
    kind: str, *, records: dict | None = None, paths=None, store: Path | None = None
) -> list[Path]:
    if paths is not None:
        stores = [root(paths=paths)]
        if paths.project_enabled:
            stores.insert(0, root(local=True, paths=paths))
        return [
            directory for location in stores for directory in resource_dirs(kind, store=location)
        ]
    store = store or root()
    directories = []
    for record in (_records(store) if records is None else records).values():
        package = _location(record["directory"], store)
        manifest = package / "rlmagent-resources.json"
        if not manifest.is_file():
            continue
        resources = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(resources, dict) or set(resources) - {"skills", "prompts", "themes"}:
            raise ValueError("Invalid package resource manifest")
        paths = resources.get(kind, [])
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError("Package resources must be lists of relative paths")
        for path in paths:
            resolved = (package / path).resolve()
            if not resolved.is_relative_to(package) or not resolved.is_dir():
                raise ValueError("Package resource directory is absent or outside its installation")
            directories.append(resolved)
    return directories


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rlm-agent package")
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--local", action="store_true", help="Use this project's package store")
    cleanup = sub.add_parser("prune", help="Review or remove expired inactive installations")
    cleanup.add_argument("--local", action="store_true")
    cleanup.add_argument("--older-than", type=float, default=RETAIN_DAYS, metavar="DAYS")
    cleanup.add_argument("--apply", action="store_true")
    install = sub.add_parser("install")
    install.add_argument("name")
    install.add_argument("source", help="Trusted wheel, directory or package-index requirement")
    install.add_argument("--local", action="store_true", help="Install for this project only")
    for action in ("remove", "update"):
        operation = sub.add_parser(action)
        operation.add_argument("name")
        if action == "update":
            operation.add_argument(
                "--source", default=None, help="Replacement local artifact or index requirement"
            )
        operation.add_argument(
            "--local", action="store_true", help="Use this project's package store"
        )
    args = parser.parse_args(argv)
    store = root(local=args.local)
    try:
        if args.action == "list":
            print(json.dumps(_records(store), indent=2))
            return 0
        if args.action == "prune":
            import math

            if not math.isfinite(args.older_than) or args.older_than < 0:
                raise ValueError("Cache retention must be a finite nonnegative number of days")
            store.mkdir(parents=True, exist_ok=True)
            with _lock(store / "registry.lock"):
                selected = prune(
                    store, _records(store), older_than=args.older_than, apply=args.apply
                )
            print(json.dumps({"removed" if args.apply else "candidates": selected}, indent=2))
            return 0
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.name):
            parser.error("invalid package name")
        store.mkdir(parents=True, exist_ok=True)
        with _lock(store / "registry.lock"):
            records = _records(store)
            if args.action == "remove":
                if args.name not in records:
                    raise ValueError("Package is not installed")
                del records[args.name]
                _save(records, store)
            else:
                source = (
                    args.source
                    if args.action == "install"
                    else args.source or records[args.name]["source"]
                )
                if (
                    not source
                    or source.startswith("-")
                    or "://" in source
                    or any(ch in source for ch in "\r\n")
                ):
                    raise ValueError("Use a package-index requirement or a local package path")
                directory = uuid4().hex
                target = _location(directory, store)
                result = subprocess.run(_installer(target, source), check=False)
                if result.returncode:
                    return result.returncode
                # Validate declared resources before making the installation active.
                packages = [
                    {"name": distribution.metadata["Name"], "version": distribution.version}
                    for distribution in importlib.metadata.distributions(path=[str(target)])
                ]
                provenance = None
                artifact = Path(source)
                if artifact.is_file():
                    with artifact.open("rb") as stream:
                        provenance = hashlib.file_digest(stream, "sha256").hexdigest()
                records[args.name] = {
                    "source": source,
                    "directory": directory,
                    "distributions": packages,
                    "artifact_sha256": provenance,
                }
                for kind in ("skills", "prompts", "themes"):
                    resource_dirs(kind, records=records, store=store)
                _save(records, store)
            try:
                prune(store, records, apply=True)
            except OSError as error:
                print(
                    f"Package saved; cache cleanup deferred ({type(error).__name__}).",
                    file=sys.stderr,
                )
        print("Package registry updated. Restart the agent to reload Python modules.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Package operation failed: {exc}", file=sys.stderr)
        return 1
