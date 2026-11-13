"""Verified wheel bundles activated through an atomic installation pointer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import venv
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

from rlmagent.discovery import _resolve_home
from rlmagent.refinement.store import _lock


def root():
    return _resolve_home() / "updates" / "managed"


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def python_for(record):
    identity = record["slot"]
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid managed installation identity")
    directory = (root() / "slots" / identity).resolve()
    if not directory.is_relative_to((root() / "slots").resolve()):
        raise ValueError("Managed installation escapes its store")
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def manifest(path):
    data = _read(path)
    if (
        not isinstance(data, dict)
        or set(data) - {"format", "version", "channel", "artifacts", "extras"}
        or not {"format", "version", "channel", "artifacts"}.issubset(data)
        or data["format"] != "rlmagent.release/1"
        or data["channel"] not in ("stable", "preview", "nightly")
        or not isinstance(data["version"], str)
        or re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", data["version"]) is None
        or not isinstance(data["artifacts"], list)
        or not 1 <= len(data["artifacts"]) <= 256
    ):
        raise ValueError("Invalid release manifest")
    extras = data.setdefault("extras", [])
    if (
        not isinstance(extras, list)
        or not all(
            extra in ("subscription", "copilot", "mcp-keyring", "vertex") for extra in extras
        )
        or len(set(extras)) != len(extras)
    ):
        raise ValueError("Invalid release extras")
    names = set()
    for artifact in data["artifacts"]:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"filename", "sha256"}
            or not isinstance(artifact["filename"], str)
            or re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", artifact["filename"]) is None
            or artifact["filename"] in names
            or not isinstance(artifact["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"]) is None
        ):
            raise ValueError("Invalid or duplicate wheel artifact")
        names.add(artifact["filename"])
    return data


def _stage(data, source, destination):
    destination.mkdir(parents=True)
    source = source.resolve()
    for artifact in data["artifacts"]:
        wheel = (source / artifact["filename"]).resolve()
        if (
            not wheel.is_relative_to(source)
            or not wheel.is_file()
            or wheel.stat().st_size > 512 * 1024 * 1024
        ):
            raise ValueError("Wheel is absent, too large, or outside the bundle")
        target = destination / artifact["filename"]
        shutil.copyfile(wheel, target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(f"Wheel checksum mismatch: {artifact['filename']}")
        with ZipFile(target) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("Wheel needs exactly one package metadata record")
            if archive.getinfo(metadata[0]).file_size > 1024 * 1024:
                raise ValueError("Wheel metadata exceeds the limit")
            for line in archive.read(metadata[0]).decode().splitlines():
                if line.lower().startswith("requires-dist:") and "@" in line:
                    raise ValueError("Offline bundles must not depend on direct network URLs")


def _health(record):
    python = python_for(record)
    script = (
        "import importlib.metadata as m; import rlmagent.cli.main; "
        "import rlmagent.tui.app; import rlmagent.model.oai_compatible; import rlmness; "
        "assert m.version('rlmness-cli') == " + repr(record["version"])
    )
    if "subscription" in record.get("extras", []):
        script += "; import jwt, keyring, websockets"
    if "copilot" in record.get("extras", []):
        script += "; import copilot"
    env = {
        key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    reply = subprocess.run(
        [str(python), "-I", "-c", script],
        cwd=python.parent.parent,
        env=env,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if reply.returncode:
        raise ValueError("Replacement installation failed its health check")


def _recover():
    from rlmagent import worker

    journal = _read(root() / "transaction.json")
    if journal and journal["phase"] == "activating":
        previous = journal.get("previous")
        if previous is not None:
            _write(root() / "current.json", previous)
        else:
            (root() / "current.json").unlink(missing_ok=True)
        if journal.get("workers"):
            names = {state["name"] for state in journal["workers"]}
            _drain([state for state in _fleet() if state["name"] in names], 30)
            if previous is not None:
                _restart(journal["workers"], previous)
        _write(root() / "transaction.json", {**journal, "phase": "recovered"})
    if journal:
        pause = worker.root() / "admission-paused"
        if pause.exists() and pause.read_text() == journal["candidate"]["slot"]:
            pause.unlink()


def _fleet():
    from rlmagent import worker

    if not worker.root().exists():
        return []
    states = []
    for path in worker.root().iterdir():
        if not path.is_dir() or not worker._NAME.fullmatch(path.name):
            continue
        state = worker.state_of(path.name)
        if state["alive"] or worker._supervised(state):
            states.append(state)
    return states


def _drain(states, timeout):
    from rlmagent import worker

    for state in states:
        (worker.folder(state["name"]) / "stop").touch()
    deadline = time.monotonic() + timeout
    names = {state["name"] for state in states}
    while any(state["name"] in names for state in _fleet()):
        if time.monotonic() >= deadline:
            raise ValueError(
                "Worker drain timed out; active installation was preserved; no workers were killed"
            )
        time.sleep(0.2)


def _restart(states, record):
    from rlmagent import worker

    for state in states:
        launch = _read(worker.folder(state["name"]) / "launch.json")
        if not launch:
            raise ValueError("Worker has no saved launch settings")
        args = list(launch["argv"])
        if "--resume" in args:
            index = args.index("--resume")
            del args[index : index + 2]
        if state.get("session"):
            args += ["--resume", state["session"]]
        result = subprocess.run(
            [
                str(python_for(record)),
                "-m",
                "rlmagent.cli.main",
                "worker",
                "start",
                state["name"],
                *args,
            ],
            cwd=launch["cwd"],
            env={
                key: value
                for key, value in os.environ.items()
                if key not in ("PYTHONPATH", "PYTHONHOME")
            },
            check=False,
            timeout=200,
        )
        if result.returncode:
            raise ValueError(f"Worker {state['name']} could not restart; inspect its saved log")


def install(path: Path, *, drain=False, wait=60, restart=False):
    if wait < 0 or wait > 3600:
        raise ValueError("Drain wait must be between 0 and 3600 seconds")
    with _lock(root() / "install.lock"):
        _recover()
        data = manifest(path)
        previous = _read(root() / "current.json")
        states = _fleet()
        if states and not drain:
            raise ValueError(
                "Workers are running; use --drain to finish their current work before activation"
            )
        if restart:
            from rlmagent import worker

            if any(_read(worker.folder(state["name"]) / "launch.json") is None for state in states):
                raise ValueError(
                    "A worker lacks saved launch settings; restart cannot be guaranteed"
                )
        candidate = {
            "slot": uuid4().hex,
            "version": data["version"],
            "channel": data["channel"],
            "extras": data.get("extras", []),
        }
        incoming = root() / "bundles" / candidate["slot"]
        _stage(data, path.parent, incoming)
        python = python_for(candidate)
        venv.EnvBuilder(with_pip=True).create(python.parent.parent)
        requirement = "rlmness-cli" + (
            "[" + ",".join(candidate["extras"]) + "]" if candidate["extras"] else ""
        )
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(incoming),
                f"{requirement}=={data['version']}",
            ],
            check=False,
            timeout=300,
        )
        if result.returncode:
            raise ValueError("Bundle installation failed; active installation was preserved")
        _health(candidate)
        if states:
            _drain(states, wait)
        journal = {
            "phase": "activating",
            "candidate": candidate,
            "previous": previous,
            "workers": states if restart else [],
        }
        _write(root() / "transaction.json", journal)
        try:
            if restart and states:
                from rlmagent import worker

                (worker.root() / "admission-paused").write_text(candidate["slot"])
            _write(root() / "previous.json", previous)
            _write(root() / "current.json", candidate)
            _health(candidate)
            if restart:
                _restart(states, candidate)
            _write(root() / "transaction.json", {**journal, "phase": "complete"})
            _recover()
        except BaseException:
            _recover()
            raise
        return candidate


def rollback(*, drain=False, wait=60, restart=False):
    with _lock(root() / "install.lock"):
        _recover()
        previous = _read(root() / "previous.json")
        if previous is None:
            raise ValueError("No prior managed installation is available")
        _health(previous)
        states = _fleet()
        if states and not drain:
            raise ValueError("Workers are running; use --drain before rollback")
        if restart:
            from rlmagent import worker

            if any(_read(worker.folder(state["name"]) / "launch.json") is None for state in states):
                raise ValueError(
                    "A worker lacks saved launch settings; restart cannot be guaranteed"
                )
        if states:
            _drain(states, wait)
        current = _read(root() / "current.json")
        journal = {
            "phase": "activating",
            "candidate": previous,
            "previous": current,
            "workers": states if restart else [],
        }
        _write(root() / "transaction.json", journal)
        try:
            if restart and states:
                from rlmagent import worker

                (worker.root() / "admission-paused").write_text(previous["slot"])
            _write(root() / "current.json", previous)
            if restart:
                _restart(states, previous)
            _write(root() / "previous.json", current)
            _write(root() / "transaction.json", {**journal, "phase": "complete"})
            _recover()
        except BaseException:
            _recover()
            raise
        return previous


def run(args):
    with _lock(root() / "install.lock"):
        _recover()
        record = _read(root() / "current.json")
        if record is None:
            raise ValueError("No managed installation is active")
        python = python_for(record)
        _health(record)
    env = {
        key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    return subprocess.run(
        [str(python), "-m", "rlmagent.cli.main", *args], env=env, check=False
    ).returncode
