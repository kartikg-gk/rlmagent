"""Named agents that keep running after the terminal closes.

A worker is an ordinary saved session in a process of its own. It is reached
through files under the home folder rather than a socket, so sending to it or
scheduling for it works whether or not it is up, and needs no open port. With
nobody there to ask, anything needing approval is refused unless the worker was
started with --auto-approve.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

BEAT_SECONDS = 2.0
POLL_SECONDS = 0.5
#: Without a beat for this long, a process with that pid is not this worker.
STALE_SECONDS = 30.0
INPUT_TIMEOUT = 60.0
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def root() -> Path:
    from rlmagent.plugins import rlm_agent_home

    return rlm_agent_home() / "workers"


def folder(name: str) -> Path:
    if not _NAME.match(name):
        raise ValueError(f"not a worker name: {name!r}")
    return root() / name


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
    staged.write_text(json.dumps(value), encoding="utf-8")
    os.replace(staged, path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def state_of(name: str) -> dict:
    """What is on record for a worker, with `alive` worked out rather than trusted."""
    meta = _read(folder(name) / "meta.json") or {}
    beat = meta.get("beat", 0)
    meta["alive"] = bool(
        meta.get("pid") and _pid_alive(meta["pid"]) and time.time() - beat < STALE_SECONDS
    )
    meta["name"] = name
    return meta


def _supervised(meta: dict) -> bool:
    pid = meta.get("supervisor_pid")
    return bool(pid and _pid_alive(pid))


def send(name: str, text: str, source: str = "user") -> str:
    message_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
    message = {"text": text, "source": source, "time": time.time()}
    _write(folder(name) / "inbox" / f"{message_id}.json", message)
    return message_id


def schedule(name: str, text: str, due: float, every: float | None = None) -> dict:
    job = {"id": uuid.uuid4().hex[:8], "text": text, "due": due, "every": every,
           "status": "waiting", "runs": 0, "last_run": None}
    _write(folder(name) / "jobs" / f"{job['id']}.json", job)
    return job


def jobs_of(name: str) -> list[dict]:
    jobs = [_read(path) for path in (folder(name) / "jobs").glob("*.json")]
    return sorted((job for job in jobs if job), key=lambda job: job["due"])


def log_line(name: str, kind: str, text: str, **details) -> None:
    path = folder(name) / "log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), "kind": kind, "text": text, **details},
                                ensure_ascii=False) + "\n")


async def _request_input(name: str, text: str, *, approval: bool, choices=(), multiple=False):
    request_id = uuid.uuid4().hex
    base = folder(name) / "input"
    request = base / f"{request_id}.request.json"
    response = base / f"{request_id}.response.json"
    expires = time.time() + INPUT_TIMEOUT
    _write(request, {"pid": os.getpid(), "expires": expires, "approval": approval})
    options = "".join(f"\n{index}. {choice}" for index, choice in enumerate(choices, 1))
    hint = "once|always|auto|deny" if approval else "TEXT"
    try:
        log_line(name, "input", f"{text}{options}\n/answer {request_id} {hint}")
        deadline = time.monotonic() + INPUT_TIMEOUT
        while time.monotonic() < deadline and not (folder(name) / "stop").exists():
            answer = _read(response)
            if answer is not None:
                value = answer.get("value")
                if not isinstance(value, str) or not value:
                    return None
                if approval:
                    return value if value in ("once", "always", "auto", "deny") else None
                picks = [part.strip() for part in value.split(",")]
                if choices and all(p.isascii() and p.isdecimal() and len(p) <= 8
                                   and 1 <= int(p) <= len(choices) for p in picks):
                    if multiple:
                        return [choices[int(p) - 1] for p in picks]
                    if len(picks) == 1:
                        return choices[int(picks[0]) - 1]
                return value
            await asyncio.sleep(0.15)
        return None
    finally:
        request.unlink(missing_ok=True)
        response.unlink(missing_ok=True)


def answer_input(name: str, command: str) -> str | None:
    """Intercept client input replies before they can enter the prompt queue."""
    parts = command.split(maxsplit=2)
    if not parts or parts[0] != "/answer":
        return None
    if len(parts) != 3 or not re.fullmatch(r"[a-f0-9]{32}", parts[1]):
        return "Use /answer REQUEST_ID TEXT."
    _, request_id, value = parts
    base = folder(name) / "input"
    request = _read(base / f"{request_id}.request.json")
    meta = state_of(name)
    if (not request or not meta["alive"] or request.get("pid") != meta.get("pid")
            or time.time() >= request.get("expires", 0)):
        return "That input request is no longer active."
    if request.get("approval") and value not in ("once", "always", "auto", "deny"):
        return "Choose once, always, auto, or deny."
    from rlmagent.harness.session.lease import SessionBusy, acquire

    response = base / f"{request_id}.response.json"
    try:
        with acquire(base, "answer"):
            if not (base / f"{request_id}.request.json").is_file():
                return "That input request is no longer active."
            if response.exists():
                return "That input request already has an answer."
            _write(response, {"value": value})
            if not (base / f"{request_id}.request.json").is_file():
                response.unlink(missing_ok=True)
    except SessionBusy:
        return "Another client is submitting an answer. Retry if the request is still active."
    return "Answer sent to the worker."


def interrupt(name: str) -> bool:
    meta = state_of(name)
    turn_id = meta.get("turn_id")
    if not meta["alive"] or not isinstance(turn_id, str) or not re.fullmatch(r"[a-f0-9]{32}", turn_id):
        return False
    path = folder(name) / "interrupts" / turn_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return True


def _enable_interactive_input(session, name: str) -> None:
    from rlmagent.agents.approval import DENY

    approver = session.agent_tree.approver

    async def ask(request):
        return await _request_input(name, request.describe(), approval=True) or DENY

    async def question(text, choices=None, multiple=False):
        return await _request_input(name, text, approval=False, choices=choices or (), multiple=multiple)

    if approver.policy == "deny":
        approver.policy = "ask"
    approver.ask = ask
    approver.question = question


def _take_inbox(name: str, *, limit: int | None = None) -> list[dict]:
    taken = []
    base = folder(name)
    active = base / "in-flight"
    active.mkdir(parents=True, exist_ok=True)
    for path in sorted((base / "inbox").glob("*.json")):
        claimed = active / path.name
        try:
            path.replace(claimed)
        except FileNotFoundError:
            continue
        message = _read(claimed)
        if message:
            message["id"] = path.stem
            taken.append(message)
            if len(taken) == limit:
                break
        else:
            _finish_message(name, path.stem, "failed")
    return taken


def _finish_message(name: str, message_id: str, status: str) -> None:
    base = folder(name)
    receipt = base / "receipts" / f"{message_id}.json"
    if not receipt.exists():
        _write(receipt, {"id": message_id, "status": status, "time": time.time()})
    (base / "in-flight" / f"{message_id}.json").unlink(missing_ok=True)


def receipt_of(name: str, message_id: str) -> dict:
    if re.fullmatch(r"[0-9]+-[0-9a-f]{6,32}", message_id) is None:
        raise ValueError("invalid worker message ID")
    base = folder(name)
    filename = f"{message_id}.json"
    return _read(base / "receipts" / filename) or {
        "id": message_id,
        "status": ("queued" if (base / "inbox" / filename).is_file()
                   else "in-flight" if (base / "in-flight" / filename).is_file() else "unknown"),
    }


def _take_due(name: str, now: float, *, limit: int | None = None) -> list[dict]:
    """Due jobs, marked as started first: a crash must not run one a second time."""
    taken = []
    for job in jobs_of(name):
        if job["status"] != "waiting" or job["due"] > now:
            continue
        job["runs"] += 1
        job["last_run"] = now
        if job["every"]:
            job["due"] = now + job["every"]
        else:
            job["status"] = "started"
        _write(folder(name) / "jobs" / f"{job['id']}.json", job)
        taken.append(job)
        if len(taken) == limit:
            break
    return taken


def _finish_job(name: str, job: dict, status: str) -> None:
    if job["every"]:
        return
    job["status"] = status
    _write(folder(name) / "jobs" / f"{job['id']}.json", job)


async def _handle(session, name: str, text: str) -> bool:
    from rlmagent.harness.contracts.stream import (
        MessageEndEvent, MessageUpdateEvent, ToolRunEndEvent, ToolRunStartEvent, ToolRunUpdateEvent,
    )
    from rlmagent.harness.contracts.transcript import ModelEntry

    log_line(name, "prompt", text)
    if text.startswith("/"):
        response = await session.handle_command(text)
        if response is not None:
            log_line(name, "reply", str(response))
            return True
    streamed = 0
    async for event in session.submit(text):
        if isinstance(event, MessageUpdateEvent) and isinstance(event.message, ModelEntry):
            full = event.message.text
            if len(full) > streamed:
                log_line(name, "model_delta", full[streamed:])
                streamed = len(full)
        elif isinstance(event, MessageEndEvent) and isinstance(event.message, ModelEntry):
            log_line(name, "model_end", event.message.text)
            streamed = 0
        elif isinstance(event, ToolRunStartEvent):
            log_line(name, "tool_start", event.tool_name, call_id=event.tool_call_id, args=event.args)
        elif isinstance(event, ToolRunUpdateEvent):
            log_line(name, "tool_update", event.partial_result.text)
        elif isinstance(event, ToolRunEndEvent):
            log_line(name, "tool_end", event.result.text, call_id=event.tool_call_id,
                     is_error=event.is_error)
    answer = next(
        (m for m in reversed(list(session.transcript)) if getattr(m, "role", "") == "assistant"),
        None,
    )
    problem = getattr(answer, "error_message", None)
    if problem:
        log_line(name, "error", str(problem))
        return False
    log_line(name, "reply", getattr(answer, "text", "") if answer else "")
    return True


async def serve(name: str, ns: argparse.Namespace) -> int:
    """The loop inside the detached process."""
    from rlmagent.runtime import build_session

    ns.print_mode = True
    base = folder(name)
    meta = {"pid": os.getpid(), "cwd": os.getcwd(), "started": time.time(), "beat": time.time(),
            "state": "starting", "session": None, "error": None}
    supervisor = os.environ.get("RLM_AGENT_WORKER_SUPERVISOR_PID")
    if supervisor:
        meta["supervisor_pid"] = int(supervisor)
    launch_id = os.environ.get("RLM_AGENT_WORKER_LAUNCH_ID")
    if launch_id:
        meta["launch_id"] = launch_id
    _write(base / "meta.json", meta)
    try:
        if getattr(ns, "supervised", False):
            session_id = name.removeprefix("session-")
            if session_id == name or ns.no_session or (ns.resume and ns.resume != session_id):
                raise ValueError("supervised workers require a matching saved session ID")
            session = await build_session(ns, session_id=session_id)
            if not ns.resume:
                try:
                    await session._write_header()
                except BaseException:
                    await session.shutdown()
                    raise
        else:
            session = await build_session(ns)
    except BaseException as failure:  # noqa: BLE001 - the starter is waiting to hear why
        message = getattr(failure, "code", None) if isinstance(failure, SystemExit) else failure
        _write(base / "meta.json", {**meta, "state": "failed", "pid": None, "error": str(message)})
        log_line(name, "error", f"could not start: {message}")
        return 1
    meta.update(session=session.session_id, state="idle", model=session.model)
    if getattr(ns, "supervised", False):
        for stale in (base / "input").glob("*.json"):
            stale.unlink(missing_ok=True)
        _enable_interactive_input(session, name)
    history = base / "history.json"
    if (_read(history) or {}).get("session") != session.session_id:
        log = base / "log.jsonl"
        _write(history, {"session": session.session_id,
                         "log_offset": log.stat().st_size if log.exists() else 0,
                         "entries": [entry.model_dump(mode="json") for entry in session.transcript]})
    for path in (base / "in-flight").glob("*.json"):
        _finish_message(name, path.stem, "interrupted")
    pilot = getattr(session, "autopilot", None)
    if pilot is not None:
        pilot.durable = True
        pilot.load()
    net = getattr(session, "agentnet", None)
    if net is not None:
        net.leave()  # a worker is reached through its worker folder, not as a session
        net.worker = name

    async def beat():
        while True:
            if turn_id := meta.get("turn_id"):
                request = base / "interrupts" / turn_id
                if request.exists():
                    request.unlink(missing_ok=True)
                    session.abort()
                    log_line(name, "note", "Interrupt requested.")
            meta["beat"] = time.time()
            _write(base / "meta.json", meta)
            await asyncio.sleep(BEAT_SECONDS)

    beating = asyncio.create_task(beat())
    log_line(name, "note", f"started in {os.getcwd()}, session {session.session_id}")
    try:
        while not (base / "stop").exists():
            if (root() / "admission-paused").exists():
                meta["state"] = "paused"
                await asyncio.sleep(POLL_SECONDS)
                continue
            meta["state"] = "idle"
            # Claim only the task we are about to run so a crash leaves the rest queued.
            due = _take_due(name, time.time(), limit=1)
            work = [(job["text"], job, None) for job in due]
            if not work:
                work = [(message["text"], None, message["id"])
                        for message in _take_inbox(name, limit=1)]
            continuing = False
            if not work and pilot is not None and (prompt := pilot.next_prompt()):
                continuing = True
                work = [(prompt, None, None)]
            if pilot is not None and pilot.stopped_because and not pilot.autonomous:
                log_line(name, "note", f"autonomous mode stopped: {pilot.stopped_because}")
                pilot.stopped_because = ""
                pilot.save()
            for text, job, message_id in work:
                if (base / "stop").exists():
                    if job is not None:
                        _finish_job(name, job, "interrupted")
                    if message_id is not None:
                        _finish_message(name, message_id, "interrupted")
                    continue
                meta["state"] = "busy"
                turn_id = meta["turn_id"] = uuid.uuid4().hex
                _write(base / "meta.json", meta)
                tracked = pilot is not None and pilot.autonomous and not text.startswith("/")
                if tracked:
                    pilot.start_worker_turn(continuation=continuing)
                try:
                    outcome = "done" if await _handle(session, name, text) else "failed"
                except Exception as failure:  # noqa: BLE001 - one bad task must not end the worker
                    log_line(name, "error", f"{type(failure).__name__}: {failure}")
                    outcome = "failed"
                if job is not None:
                    _finish_job(name, job, outcome)
                if message_id is not None:
                    _finish_message(name, message_id, outcome)
                if tracked:
                    if outcome == "failed" and pilot.autonomous:
                        pilot.stop_autonomous("the worker turn failed")
                    pilot.finish_worker_turn()
                meta["state"] = "idle"
                meta.pop("turn_id", None)
                _write(base / "meta.json", meta)
                (base / "interrupts" / turn_id).unlink(missing_ok=True)
            await asyncio.sleep(POLL_SECONDS)
    finally:
        beating.cancel()
        await session.shutdown()
        _write(base / "meta.json", {**meta, "state": "stopped", "pid": None})
        (base / "stop").unlink(missing_ok=True)
        log_line(name, "note", "stopped")
    return 0


def _spawn(name: str, run_args: list[str], *, recovery_cwd: str | None = None,
           launch_id: str | None = None) -> subprocess.Popen:
    base = folder(name)
    base.mkdir(parents=True, exist_ok=True)
    if recovery_cwd is None:
        (base / "meta.json").unlink(missing_ok=True)
        (base / "stop").unlink(missing_ok=True)
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        options["start_new_session"] = True
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if launch_id:
        env["RLM_AGENT_WORKER_LAUNCH_ID"] = launch_id
    with (base / "output.log").open("ab") as output:
        return subprocess.Popen(
            [sys.executable, "-m", "rlmagent.cli.main", "worker", "supervise", name, *run_args],
            stdin=subprocess.DEVNULL, stdout=output, stderr=output,
            cwd=recovery_cwd or os.getcwd(),
            env=env, **options,
        )


def _supervise(name: str, run_args: list[str]) -> int:
    """Restart a failed worker using its last saved session."""
    from rlmagent.harness.session.lease import SessionBusy, acquire

    try:
        lease = acquire(folder(name), "supervisor")
    except SessionBusy:
        return 1
    with lease:
        return _supervise_locked(name, run_args)


def _supervise_locked(name: str, run_args: list[str]) -> int:
    base = folder(name)
    failures = 0
    initial_args = run_args
    while not (base / "stop").exists():
        started = time.monotonic()
        status = subprocess.run(
            [sys.executable, "-m", "rlmagent.cli.main", "worker", "serve", name, *run_args],
            check=False,
            env={**os.environ, "RLM_AGENT_WORKER_SUPERVISOR_PID": str(os.getpid())},
        ).returncode
        meta = _read(base / "meta.json") or {}
        session_id = meta.get("session")
        if status == 0 or not session_id or "--no-session" in initial_args or (base / "stop").exists():
            return status
        failures = 0 if time.monotonic() - started >= 60 else failures + 1
        if failures >= 5:
            return status
        delay = min(15, 2 ** failures)
        _write(base / "meta.json", {**meta, "state": "recovering", "pid": None,
                                    "supervisor_pid": os.getpid()})
        log_line(name, "note", f"worker exited with status {status}; recovering in {delay}s")
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if (base / "stop").exists():
                _write(base / "meta.json", {**meta, "state": "stopped", "pid": None,
                                            "supervisor_pid": None})
                return 0
            time.sleep(0.2)
        run_args = [*initial_args, "--resume", session_id]
    return 0


def _start(name: str, run_args: list[str], wait: float = 180.0) -> int:
    from rlmagent.harness.session.lease import SessionBusy, acquire

    try:
        lease = acquire(folder(name), "start")
    except SessionBusy:
        print(f"Worker {name} is already starting.", file=sys.stderr)
        return 1
    with lease:
        return _start_locked(name, run_args, wait)


def _start_locked(name: str, run_args: list[str], wait: float) -> int:
    from rlmagent.cli.main import _build_run_parser
    from rlmagent.runtime import settle_project_trust

    current = state_of(name)
    if current["alive"] or _supervised(current):
        print(f"Worker {name} is already running.", file=sys.stderr)
        return 1
    ns = _build_run_parser().parse_args(run_args)
    # Asked here, with a person present. The worker has nobody to ask.
    outcome = asyncio.run(settle_project_trust(ns))
    if not (ns.approve or ns.no_approve):
        run_args = [*run_args, "--approve" if outcome.trusted else "--no-approve"]
    _write(folder(name) / "launch.json", {"cwd": os.getcwd(), "argv": run_args})
    _spawn(name, run_args)
    limit = time.monotonic() + wait
    while time.monotonic() < limit:
        meta = _read(folder(name) / "meta.json") or {}
        if meta.get("state") == "failed":
            print(f"Worker {name} failed to start: {meta.get('error')}", file=sys.stderr)
            return 1
        if meta.get("session"):
            print(f"Worker {name} running, pid {meta['pid']}, session {meta['session']}.")
            return 0
        time.sleep(0.2)
    print(f"Worker {name} did not start in {wait:.0f}s; see {folder(name) / 'output.log'}.",
          file=sys.stderr)
    return 1


def _stop(name: str, wait: float) -> int:
    meta = state_of(name)
    if meta.get("state") == "recovering" and _supervised(meta):
        (folder(name) / "stop").touch()
        print(f"Worker {name} will stop instead of restarting.")
        return 0
    if not meta["alive"]:
        print(f"Worker {name} is not running.")
        return 0
    stop = folder(name) / "stop"
    stop.touch()
    limit = time.monotonic() + wait
    while time.monotonic() < limit:
        if not state_of(name)["alive"]:
            print(f"Worker {name} stopped.")
            return 0
        time.sleep(0.2)
    # Still beating, so the pid is still this worker.
    if state_of(name)["alive"]:
        os.kill(meta["pid"], signal.SIGTERM)
        print(f"Worker {name} did not stop in {wait:.0f}s and was ended; its current task is lost.")
    return 0


def _list() -> int:
    names = _worker_names()
    if not names:
        print("No workers.")
    for name in names:
        meta = state_of(name)
        recovering = meta.get("state") == "recovering" and _supervised(meta)
        state = meta.get("state", "?") if meta["alive"] or recovering else "not running"
        waiting = sum(job["status"] == "waiting" for job in jobs_of(name))
        print(f"{name}  {state}  session {meta.get('session')}  {meta.get('cwd', '')}  "
              f"{waiting} job(s) waiting")
    return 0


def _worker_names() -> list[str]:
    if not root().is_dir():
        return []
    return sorted(path.name for path in root().iterdir()
                  if path.is_dir() and _NAME.fullmatch(path.name))


def _stop_all(wait: float) -> int:
    import math

    if not math.isfinite(wait) or wait < 0:
        raise ValueError("wait must be finite and nonnegative")
    targets = []
    for name in _worker_names():
        meta = state_of(name)
        if meta.get("state") not in ("starting", "idle", "busy", "paused", "recovering"):
            continue
        (folder(name) / "stop").touch()
        targets.append(name)
    deadline = time.monotonic() + wait
    while targets:
        targets = [name for name in targets
                   if (meta := state_of(name))["alive"] or _supervised(meta)]
        if not targets:
            break
        if time.monotonic() >= deadline:
            print("Still stopping: " + ", ".join(targets), file=sys.stderr)
            return 1
        time.sleep(0.2)
    print("Worker fleet stopped.")
    return 0


def _recover_worker(name: str, wait: float = 180.0) -> int:
    from rlmagent.harness.session.lease import SessionBusy, acquire

    base = folder(name)
    try:
        lease = acquire(base, "start")
    except SessionBusy:
        return 0
    with lease:
        current = state_of(name)
        if not _can_recover(base, current):
            return 0
        launch = _read(base / "launch.json")
        if not launch:
            print(f"Worker {name} lacks saved launch settings.", file=sys.stderr)
            return 1
        run_args = list(launch["argv"])
        if "--no-session" in run_args:
            return 0
        if "--resume" in run_args:
            index = run_args.index("--resume")
            del run_args[index:index + 2]
        session_id = current["session"]
        launch_id = uuid.uuid4().hex
        incident_path = base / "incidents" / f"{time.time_ns()}-{launch_id}.json"
        incident = {
            "session": session_id, "recorded_at": time.time(), "status": "recovering",
            "last_heartbeat": current.get("beat"), "previous_state": current["state"],
            "previous_pid": current.get("pid"), "previous_supervisor_pid": current.get("supervisor_pid"),
            "in_flight_messages": sorted(path.stem for path in (base / "in-flight").glob("*.json")),
            "started_jobs": [job["id"] for job in jobs_of(name) if job["status"] == "started"],
        }
        _write(incident_path, incident)
        try:
            supervisor = _spawn(name, [*run_args, "--resume", session_id],
                                recovery_cwd=launch["cwd"], launch_id=launch_id)
        except OSError:
            _write(incident_path, {**incident, "status": "spawn_failed"})
            raise
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            meta = state_of(name)
            if meta["alive"] and meta.get("session") == session_id and meta.get("launch_id") == launch_id:
                _write(incident_path, {**incident, "status": "recovered",
                                       "pid": meta["pid"], "supervisor_pid": meta.get("supervisor_pid")})
                print(f"Recovered worker {name}, session {session_id}.")
                return 0
            if supervisor.poll() is not None:
                break
            time.sleep(0.2)
        _write(incident_path, {**incident, "status": "exited" if supervisor.poll() is not None
                               else "readiness_timeout"})
        print(f"Worker {name} did not recover. See {base / 'output.log'}.", file=sys.stderr)
        return 1


def _can_recover(base: Path, current: dict) -> bool:
    return (
        current.get("state") in ("starting", "idle", "busy", "paused", "recovering")
        and bool(current.get("session"))
        and not (base / "stop").exists()
        and not any(_pid_alive(pid) for pid in (current.get("pid"), current.get("supervisor_pid")) if pid)
    )


def _watch_fleet(stop: bool = False) -> int:
    """Recover saved workers on startup and after subsequent process losses."""
    from concurrent.futures import ThreadPoolExecutor

    from rlmagent.harness.session.lease import SessionBusy, acquire

    stop_file = root() / "fleet-watch.stop"
    if stop:
        root().mkdir(parents=True, exist_ok=True)
        stop_file.touch()
        return 0
    try:
        lease = acquire(root(), "fleet-watch")
    except SessionBusy:
        print("A fleet watcher is already running.", file=sys.stderr)
        return 1
    pending = {}
    retry_after = {}
    with lease, ThreadPoolExecutor(max_workers=4) as pool:
        stop_file.unlink(missing_ok=True)
        print("Watching the worker fleet.", flush=True)
        try:
            while not stop_file.exists():
                now = time.monotonic()
                for name, future in list(pending.items()):
                    if not future.done():
                        continue
                    del pending[name]
                    retry_after[name] = now + STALE_SECONDS
                    try:
                        future.result()
                    except Exception as error:
                        print(f"Worker {name} recovery failed: {error}", file=sys.stderr)
                for name in _worker_names():
                    if len(pending) == 4:
                        break
                    if name in pending or now < retry_after.get(name, 0):
                        continue
                    if _can_recover(folder(name), state_of(name)):
                        pending[name] = pool.submit(_recover_worker, name)
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            pass
    return 0


def _show(record: dict) -> None:
    if record["kind"] in ("model_delta", "model_end", "tool_start", "tool_update", "tool_end"):
        return
    stamp = time.strftime("%H:%M:%S", time.localtime(record["time"]))
    print(f"[{stamp} {record['kind']}] {record['text']}", flush=True)


class WorkerLog:
    """Read only complete log records and keep a cursor for the next poll."""

    def __init__(self, name: str, *, recent_only: bool = False, start_at: int = 0):
        self.path = folder(name) / "log.jsonl"
        self.position = start_at
        self.recent_only = recent_only
        self.identity = None

    def read(self) -> list[dict]:
        records = []
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            return records
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if (self.identity is not None and identity != self.identity) or stat.st_size < self.position:
                self.position = 0
            if self.identity is None and not self.position and self.recent_only and stat.st_size > 131_072:
                stream.seek(stat.st_size - 131_072)
                stream.readline()
                self.position = stream.tell()
            self.identity = identity
            stream.seek(self.position)
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                self.position += len(line)
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(record, dict) and {"time", "kind", "text"} <= record.keys():
                    records.append(record)
        return records


def _follow(name: str, stop: threading.Event, tail: int | None) -> None:
    cursor = WorkerLog(name)
    rows = [row for row in cursor.read()
            if row["kind"] not in ("model_delta", "model_end", "tool_start", "tool_update", "tool_end")]
    for row in rows[-tail:] if tail else rows:
        _show(row)
    while not stop.is_set():
        for row in cursor.read():
            _show(row)
        stop.wait(0.3)


def _attach(name: str, *, tui: bool = False) -> int:
    if tui:
        from rlmagent.tui.worker_attach import WorkerAttachApp

        WorkerAttachApp(name).run()
        return 0
    if not state_of(name)["alive"]:
        print(f"Worker {name} is not running; what you send waits for its next start.",
              file=sys.stderr)
    done = threading.Event()
    threading.Thread(target=_follow, args=(name, done, 20), daemon=True).start()
    print("Attached. Lines you type go to the worker. /detach leaves it running.", flush=True)
    try:
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                if interrupt(name):
                    print("Interrupt sent to the worker.", flush=True)
                    continue
                break
            if not line:
                break
            text = line.strip()
            if text in ("/detach", "/quit", "/exit", "/q"):
                break
            if text:
                result = answer_input(name, text)
                if result is None:
                    send(name, text)
                else:
                    print(result, flush=True)
    except KeyboardInterrupt:
        pass
    done.set()
    return 0


def handle_session(argv: list[str]) -> int:
    """Give a saved conversation a detached process and a stable terminal address."""
    from rlmagent import agentnet
    from rlmagent.cli.main import _build_run_parser

    action = argv[0]
    parser = argparse.ArgumentParser(prog=f"rlm-agent session {action}")
    parser.add_argument("session_id")
    if action == "attach":
        parser.add_argument("--tui", action="store_true", help="Open the Textual session view.")
    if action == "detach":
        args, run_args = parser.parse_known_args(argv[1:])
    else:
        args, run_args = parser.parse_args(argv[1:]), []
    session_id = args.session_id
    name = f"session-{session_id}"
    if not _NAME.fullmatch(name):
        parser.error("invalid session ID")
    if action == "attach":
        state = state_of(name)
        if not state["alive"] and not (args.tui and state.get("session") == session_id):
            parser.error("session is offline; start it with session detach first")
        return _attach(name, tui=args.tui)

    ns = _build_run_parser().parse_args(run_args)
    if ns.resume or ns.no_session or ns.contract or ns.prompt:
        parser.error("give the saved session ID followed by provider and runtime options")
    registered = agentnet._read(agentnet._home() / f"s-{session_id[:8]}" / "meta.json")
    if registered and registered.get("pid") and _pid_alive(registered["pid"]):
        parser.error("close the session's current terminal before detaching it")
    return _start(name, [*run_args, "--resume", session_id])


def start_interactive(run_args: list[str], *, session_id: str | None, tui: bool) -> int:
    session_id = session_id or uuid.uuid4().hex[:12]
    name = f"session-{session_id}"
    state = state_of(name)
    if state["alive"]:
        if state.get("session") != session_id:
            raise ValueError(f"Worker {name} owns a different session.")
    else:
        if tui:
            from rlmagent.cli.main import _build_run_parser
            from rlmagent.runtime import settle_project_trust
            from rlmagent.tui.app import setup_if_needed
            from rlmagent.tui.trust_dialog import ask_in_dialog

            async def prepare():
                ns = _build_run_parser().parse_args(run_args)
                trust = await settle_project_trust(ns, ask=ask_in_dialog)
                return trust.trusted if await setup_if_needed(ns) else None

            trusted = asyncio.run(prepare())
            if trusted is None:
                return 1
            ns = _build_run_parser().parse_args(run_args)
            if not (ns.approve or ns.no_approve):
                run_args = [*run_args, "--approve" if trusted else "--no-approve"]
        if "--supervised" not in run_args:
            run_args = [*run_args, "--supervised"]
        status = _start(name, run_args)
        if status:
            return status
    return _attach(name, tui=tui)


def _when(moment: str) -> float:
    return datetime.fromisoformat(moment).timestamp()


def handle(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rlm-agent worker")
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start", help="Start a named worker in this folder. Run flags follow.")
    start.add_argument("name")
    sub.add_parser("list", help="List workers.")
    sub.add_parser("incidents", help="Show recorded recovery attempts.").add_argument("name")
    recover = sub.add_parser("recover", help="Recover interrupted saved workers.")
    recover.add_argument("name", nargs="?")
    watch = sub.add_parser("watch", help="Monitor the fleet and recover interrupted saved workers.")
    watch.add_argument("--stop", action="store_true", help="Stop the fleet watcher.")
    stop_all = sub.add_parser("stop-all", help="Ask the worker fleet to finish and stop.")
    stop_all.add_argument("--wait", type=float, default=60.0)
    for action, text in (("attach", "Follow a worker and type to it."),
                         ("jobs", "List a worker's scheduled jobs."),
                         ("remove", "Delete a stopped worker's folder.")):
        worker_action = sub.add_parser(action, help=text)
        worker_action.add_argument("name")
        if action == "attach":
            worker_action.add_argument("--tui", action="store_true", help="Open the Textual worker view.")
    stop = sub.add_parser("stop", help="Stop a worker after its current task.")
    stop.add_argument("name")
    stop.add_argument("--wait", type=float, default=60.0)
    tell = sub.add_parser("send", help="Queue a prompt for a worker.")
    tell.add_argument("name")
    tell.add_argument("text")
    receipt = sub.add_parser("receipt", help="Show the outcome of an inbox message.")
    receipt.add_argument("name")
    receipt.add_argument("message_id")
    log = sub.add_parser("log", help="Show a worker's log.")
    log.add_argument("name")
    log.add_argument("--follow", action="store_true")
    log.add_argument("--tail", type=int, default=None)
    plan = sub.add_parser("schedule", help="Send a prompt later, once or repeatedly.")
    plan.add_argument("name")
    plan.add_argument("text")
    moment = plan.add_mutually_exclusive_group(required=True)
    moment.add_argument("--in", dest="delay", type=float, metavar="SECONDS")
    moment.add_argument("--at", metavar="ISO-TIME", help="For example 2027-01-02T09:00.")
    plan.add_argument("--every", type=float, metavar="SECONDS")
    cancel = sub.add_parser("unschedule", help="Remove a scheduled job.")
    cancel.add_argument("name")
    cancel.add_argument("job")

    if argv[:1] == ["supervise"]:
        return _supervise(argv[1], argv[2:])
    if argv[:1] == ["serve"]:
        from rlmagent.cli.main import _build_run_parser

        return asyncio.run(serve(argv[1], _build_run_parser().parse_args(argv[2:])))
    if argv[:1] == ["start"]:
        ns, run_args = start_parser_split(argv[1:])
        return _start(ns, run_args)

    ns = parser.parse_args(argv)
    try:
        if ns.action == "list":
            return _list()
        if ns.action == "incidents":
            records = sorted((folder(ns.name) / "incidents").glob("*.json"))
            print(json.dumps([value for path in records if (value := _read(path)) is not None]))
            return 0
        if ns.action == "recover":
            names = [ns.name] if ns.name else _worker_names()
            results = [_recover_worker(name) for name in names]
            return max(results, default=0)
        if ns.action == "watch":
            return _watch_fleet(ns.stop)
        if ns.action == "stop-all":
            return _stop_all(ns.wait)
        if ns.action == "stop":
            return _stop(ns.name, ns.wait)
        if ns.action == "send":
            message_id = send(ns.name, ns.text)
            print(f"Queued for {ns.name}. Message {message_id}.")
            return 0
        if ns.action == "receipt":
            print(json.dumps(receipt_of(ns.name, ns.message_id)))
            return 0
        if ns.action == "attach":
            return _attach(ns.name, tui=ns.tui)
        if ns.action == "log":
            done = threading.Event()
            if ns.follow:
                try:
                    _follow(ns.name, done, ns.tail)
                except KeyboardInterrupt:
                    pass
            else:
                done.set()
                _follow(ns.name, done, ns.tail)
            return 0
        if ns.action == "schedule":
            due = time.time() + ns.delay if ns.delay is not None else _when(ns.at)
            job = schedule(ns.name, ns.text, due, ns.every)
            at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(due))
            print(f"Job {job['id']} will run at {at}.")
            if not state_of(ns.name)["alive"]:
                print(f"Worker {ns.name} is not running; the job waits until it starts.")
            return 0
        if ns.action == "jobs":
            for job in jobs_of(ns.name):
                at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job["due"]))
                print(f"{job['id']}  {job['status']}  {at}  runs {job['runs']}  {job['text'][:60]}")
            return 0
        if ns.action == "unschedule":
            (folder(ns.name) / "jobs" / f"{ns.job}.json").unlink(missing_ok=True)
            return 0
        if ns.action == "remove":
            meta = state_of(ns.name)
            if meta["alive"] or _supervised(meta):
                print(f"Worker {ns.name} is running; stop it first.", file=sys.stderr)
                return 1
            shutil.rmtree(folder(ns.name), ignore_errors=True)
            return 0
    except (OSError, ValueError) as problem:
        print(f"rlm-agent: {problem}", file=sys.stderr)
        return 2
    return 2


def start_parser_split(rest: list[str]) -> tuple[str, list[str]]:
    if not rest:
        raise SystemExit("usage: rlm-agent worker start NAME [run flags]")
    return rest[0], rest[1:]
