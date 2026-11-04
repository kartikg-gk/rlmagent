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


def send(name: str, text: str, source: str = "user") -> None:
    message = {"text": text, "source": source, "time": time.time()}
    _write(folder(name) / "inbox" / f"{time.time_ns()}-{uuid.uuid4().hex[:6]}.json", message)


def schedule(name: str, text: str, due: float, every: float | None = None) -> dict:
    job = {"id": uuid.uuid4().hex[:8], "text": text, "due": due, "every": every,
           "status": "waiting", "runs": 0, "last_run": None}
    _write(folder(name) / "jobs" / f"{job['id']}.json", job)
    return job


def jobs_of(name: str) -> list[dict]:
    jobs = [_read(path) for path in (folder(name) / "jobs").glob("*.json")]
    return sorted((job for job in jobs if job), key=lambda job: job["due"])


def log_line(name: str, kind: str, text: str) -> None:
    path = folder(name) / "log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), "kind": kind, "text": text},
                                ensure_ascii=False) + "\n")


def _take_inbox(name: str, *, limit: int | None = None) -> list[dict]:
    taken = []
    for path in sorted((folder(name) / "inbox").glob("*.json")):
        message = _read(path)
        path.unlink(missing_ok=True)
        if message:
            taken.append(message)
            if len(taken) == limit:
                break
    return taken


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


async def _handle(session, name: str, text: str) -> None:
    log_line(name, "prompt", text)
    if text.startswith("/"):
        response = await session.handle_command(text)
        if response is not None:
            log_line(name, "reply", str(response))
            return
    async for _event in session.submit(text):
        pass
    answer = next(
        (m for m in reversed(list(session.transcript)) if getattr(m, "role", "") == "assistant"),
        None,
    )
    problem = getattr(answer, "error_message", None)
    if problem:
        log_line(name, "error", str(problem))
    else:
        log_line(name, "reply", getattr(answer, "text", "") if answer else "")


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
    _write(base / "meta.json", meta)
    try:
        session = await build_session(ns)
    except BaseException as failure:  # noqa: BLE001 - the starter is waiting to hear why
        message = getattr(failure, "code", None) if isinstance(failure, SystemExit) else failure
        _write(base / "meta.json", {**meta, "state": "failed", "pid": None, "error": str(message)})
        log_line(name, "error", f"could not start: {message}")
        return 1
    meta.update(session=session.session_id, state="idle", model=session.model)
    net = getattr(session, "agentnet", None)
    if net is not None:
        net.leave()  # a worker is reached through its worker folder, not as a session
        net.worker = name

    async def beat():
        while True:
            meta["beat"] = time.time()
            _write(base / "meta.json", meta)
            await asyncio.sleep(BEAT_SECONDS)

    beating = asyncio.create_task(beat())
    log_line(name, "note", f"started in {os.getcwd()}, session {session.session_id}")
    try:
        while not (base / "stop").exists():
            # Claim only the task we are about to run so a crash leaves the rest queued.
            due = _take_due(name, time.time(), limit=1)
            work = [(job["text"], job) for job in due]
            if not work:
                work = [(message["text"], None) for message in _take_inbox(name, limit=1)]
            for text, job in work:
                if (base / "stop").exists():
                    if job is not None:
                        _finish_job(name, job, "interrupted")
                    continue
                meta["state"] = "busy"
                try:
                    await _handle(session, name, text)
                    if job is not None:
                        _finish_job(name, job, "done")
                except Exception as failure:  # noqa: BLE001 - one bad task must not end the worker
                    log_line(name, "error", f"{type(failure).__name__}: {failure}")
                    if job is not None:
                        _finish_job(name, job, "failed")
                meta["state"] = "idle"
            await asyncio.sleep(POLL_SECONDS)
    finally:
        beating.cancel()
        await session.shutdown()
        _write(base / "meta.json", {**meta, "state": "stopped", "pid": None})
        (base / "stop").unlink(missing_ok=True)
        log_line(name, "note", "stopped")
    return 0


def _spawn(name: str, run_args: list[str]) -> None:
    base = folder(name)
    base.mkdir(parents=True, exist_ok=True)
    (base / "meta.json").unlink(missing_ok=True)
    (base / "stop").unlink(missing_ok=True)
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        options["start_new_session"] = True
    with (base / "output.log").open("ab") as output:
        subprocess.Popen(
            [sys.executable, "-m", "rlmagent.cli.main", "worker", "supervise", name, *run_args],
            stdin=subprocess.DEVNULL, stdout=output, stderr=output, cwd=os.getcwd(),
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, **options,
        )


def _supervise(name: str, run_args: list[str]) -> int:
    """Restart a failed worker using its last saved session."""
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
    names = sorted(path.name for path in root().iterdir() if path.is_dir()) if root().is_dir() else []
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


def _show(record: dict) -> None:
    stamp = time.strftime("%H:%M:%S", time.localtime(record["time"]))
    print(f"[{stamp} {record['kind']}] {record['text']}", flush=True)


def _follow(name: str, stop: threading.Event, tail: int | None) -> None:
    path = folder(name) / "log.jsonl"
    position = 0
    if path.exists():
        rows = path.read_text(encoding="utf-8").splitlines()
        for line in rows[-tail:] if tail else rows:
            _show(json.loads(line))
        position = path.stat().st_size
    while not stop.is_set():
        if path.exists() and path.stat().st_size > position:
            with path.open("rb") as stream:
                stream.seek(position)
                chunk = stream.read()
            end = chunk.rfind(b"\n") + 1
            for line in chunk[:end].decode("utf-8").splitlines():
                _show(json.loads(line))
            position += end
        stop.wait(0.3)


def _attach(name: str) -> int:
    if not state_of(name)["alive"]:
        print(f"Worker {name} is not running; what you send waits for its next start.",
              file=sys.stderr)
    done = threading.Event()
    threading.Thread(target=_follow, args=(name, done, 20), daemon=True).start()
    print("Attached. Lines you type go to the worker. /detach leaves it running.", flush=True)
    try:
        for line in sys.stdin:
            text = line.strip()
            if text == "/detach":
                break
            if text:
                send(name, text)
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
    if action == "detach":
        args, run_args = parser.parse_known_args(argv[1:])
    else:
        args, run_args = parser.parse_args(argv[1:]), []
    session_id = args.session_id
    name = f"session-{session_id}"
    if not _NAME.fullmatch(name):
        parser.error("invalid session ID")
    if action == "attach":
        if not state_of(name)["alive"]:
            parser.error("session is offline; start it with session detach first")
        return _attach(name)

    ns = _build_run_parser().parse_args(run_args)
    if ns.resume or ns.no_session or ns.contract or ns.prompt:
        parser.error("give the saved session ID followed by provider and runtime options")
    registered = agentnet._read(agentnet._home() / f"s-{session_id[:8]}" / "meta.json")
    if registered and registered.get("pid") and _pid_alive(registered["pid"]):
        parser.error("close the session's current terminal before detaching it")
    return _start(name, [*run_args, "--resume", session_id])


def _when(moment: str) -> float:
    return datetime.fromisoformat(moment).timestamp()


def handle(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rlm-agent worker")
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start", help="Start a named worker in this folder. Run flags follow.")
    start.add_argument("name")
    sub.add_parser("list", help="List workers.")
    for action, text in (("attach", "Follow a worker and type to it."),
                         ("jobs", "List a worker's scheduled jobs."),
                         ("remove", "Delete a stopped worker's folder.")):
        sub.add_parser(action, help=text).add_argument("name")
    stop = sub.add_parser("stop", help="Stop a worker after its current task.")
    stop.add_argument("name")
    stop.add_argument("--wait", type=float, default=60.0)
    tell = sub.add_parser("send", help="Queue a prompt for a worker.")
    tell.add_argument("name")
    tell.add_argument("text")
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
        if ns.action == "stop":
            return _stop(ns.name, ns.wait)
        if ns.action == "send":
            send(ns.name, ns.text)
            print(f"Queued for {ns.name}.")
            return 0
        if ns.action == "attach":
            return _attach(ns.name)
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
    except ValueError as problem:
        print(f"rlm-agent: {problem}", file=sys.stderr)
        return 2
    return 2


def start_parser_split(rest: list[str]) -> tuple[str, list[str]]:
    if not rest:
        raise SystemExit("usage: rlm-agent worker start NAME [run flags]")
    return rest[0], rest[1:]
