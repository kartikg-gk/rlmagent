"""Workers: the files they are reached through, and a real detached process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from rlmagent import worker


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_names_cannot_leave_the_folder(home):
    for bad in ("../x", "a/b", "", "a b", ".."):
        with pytest.raises(ValueError):
            worker.folder(bad)


def test_a_job_runs_once_even_if_the_worker_dies_after_taking_it(home):
    job = worker.schedule("w", "hello", time.time() - 1)
    assert [j["id"] for j in worker._take_due("w", time.time())] == [job["id"]]
    # A restarted worker finds it marked started, not waiting.
    assert worker._take_due("w", time.time()) == []
    assert worker.jobs_of("w")[0]["status"] == "started"


def test_a_future_job_waits_and_a_repeating_one_comes_back(home):
    worker.schedule("w", "later", time.time() + 100)
    assert worker._take_due("w", time.time()) == []
    worker.schedule("r", "again", time.time() - 1, every=50)
    assert len(worker._take_due("r", time.time())) == 1
    assert worker._take_due("r", time.time()) == []
    assert len(worker._take_due("r", time.time() + 60)) == 1
    assert worker.jobs_of("r")[0]["runs"] == 2


def test_the_inbox_keeps_order_and_hands_each_message_over_once(home):
    first = worker.send("w", "one")
    worker.send("w", "two")
    assert worker.receipt_of("w", first)["status"] == "queued"
    assert [m["text"] for m in worker._take_inbox("w")] == ["one", "two"]
    assert worker._take_inbox("w") == []
    assert worker.receipt_of("w", first)["status"] == "in-flight"
    worker._finish_message("w", first, "done")
    worker._finish_message("w", first, "interrupted")
    assert worker.receipt_of("w", first)["status"] == "done"


def test_a_dead_or_silent_worker_is_not_alive(home):
    worker._write(worker.folder("w") / "meta.json", {"pid": 999999, "beat": time.time()})
    assert not worker.state_of("w")["alive"]
    # This process is alive, but a silent record is not this worker.
    worker._write(worker.folder("s") / "meta.json", {"pid": os.getpid(), "beat": time.time() - 999})
    assert not worker.state_of("s")["alive"]
    worker._write(worker.folder("a") / "meta.json", {"pid": os.getpid(), "beat": time.time()})
    assert worker.state_of("a")["alive"]


def test_cli_schedule_and_jobs(home, capsys):
    assert worker.handle(["schedule", "w", "do it", "--in", "100"]) == 0
    out = capsys.readouterr().out
    assert "will run at" in out and "not running" in out
    assert worker.handle(["jobs", "w"]) == 0
    assert "do it" in capsys.readouterr().out
    assert worker.handle(["schedule", "../bad", "x", "--in", "1"]) == 2


@pytest.mark.parametrize(("expression", "zone", "after", "expected"), [
    ("30 2 * * *", "America/New_York", "2027-03-13T07:30+00:00", "2027-03-15T06:30+00:00"),
    ("30 1 * * *", "America/New_York", "2027-11-07T05:30+00:00", "2027-11-07T06:30+00:00"),
    ("0 9 1 * 7", "UTC", "2027-01-01T09:00+00:00", "2027-08-01T09:00+00:00"),
    ("@hourly", "UTC", "2027-01-01T09:00+00:00", "2027-01-01T10:00+00:00"),
    ("5,10-20/5 9 * * 1-5", "Asia/Kolkata", "2027-01-04T03:40+00:00", "2027-01-04T03:45+00:00"),
])
def test_cron_finds_next_real_local_time(expression, zone, after, expected):
    from rlmagent.cron import next_run

    assert next_run(expression, zone, datetime.fromisoformat(after).timestamp()) == (
        datetime.fromisoformat(expected).timestamp()
    )


def test_worker_cron_advances_once_after_downtime_and_can_be_unscheduled(home, capsys, monkeypatch):
    from rlmagent.harness.session.lease import acquire

    monkeypatch.setattr(worker.time, "time", lambda: 1_800_000_000)
    assert worker.handle(["schedule", "w", "nightly", "--cron", "*/10 * * * *"]) == 0
    job = worker.jobs_of("w")[0]
    assert job["timezone"] == "UTC"
    assert worker.handle(["jobs", "w"]) == 0
    assert "cron */10 * * * * (UTC)" in capsys.readouterr().out
    with acquire(worker.folder("w") / "jobs", job["id"]):
        assert worker._take_due("w", job["due"]) == []
        assert worker.handle(["unschedule", "w", job["id"]]) == 2
    delayed = job["due"] + 1_305
    [taken] = worker._take_due("w", delayed)
    assert taken["due"] == job["due"] + 1_800
    assert taken["runs"] == 1
    assert worker._take_due("w", delayed) == []
    worker._finish_job("w", taken, "done")
    assert worker.handle(["unschedule", "w", job["id"]]) == 0
    assert worker._take_due("w", taken["due"]) == []


async def test_agent_tool_schedules_cron_and_rejects_invalid_combinations(home):
    from rlmagent.agentnet import AgentNet

    agent = AgentNet("session", str(home), worker="w")
    assert "job " in await agent.schedule("task", cron="0 9 * * 1-5", timezone="Asia/Kolkata")
    assert worker.jobs_of("w")[0]["cron"] == "0 9 * * 1-5"
    with pytest.raises(ValueError, match="exactly one"):
        await agent.schedule("task", in_seconds=60, cron="* * * * *")
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        await agent.schedule("task", cron="* * * * *", timezone="Invalid/Zone")
    with pytest.raises(ValueError, match="no occurrence"):
        await agent.schedule("task", cron="0 0 30 2 *")


def test_stop_all_preserves_stop_intent_for_interrupted_workers(home, monkeypatch):
    monkeypatch.setattr(worker, "_pid_alive", lambda _: False)
    for name, state in (("interrupted", "busy"), ("finished", "stopped")):
        worker._write(worker.folder(name) / "meta.json",
                      {"pid": 1, "state": state, "session": "saved"})
    assert worker.handle(["stop-all", "--wait", "0"]) == 0
    assert (worker.folder("interrupted") / "stop").is_file()
    assert not (worker.folder("finished") / "stop").exists()
    assert worker.handle(["recover", "interrupted"]) == 0


def test_supervisor_retains_session_after_a_restart_fails_before_readiness(home, monkeypatch, capsys):
    from itertools import count
    from types import SimpleNamespace

    base = worker.folder("w")
    ticks = count(step=20)
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))
    attempts = []

    def run(argv, *, check, env):
        attempts.append(argv)
        if len(attempts) == 1:
            worker._write(base / "meta.json", {"state": "busy", "session": "saved", "pid": 42})
            return SimpleNamespace(returncode=3)
        assert argv[-2:] == ["--resume", "saved"]
        if len(attempts) == 2:
            worker._write(base / "meta.json", {"state": "failed", "session": None})
            return SimpleNamespace(returncode=1)
        assert worker._read(base / "meta.json")["session"] == "saved"
        incident = base / "incidents" / f"{env['RLM_AGENT_WORKER_INCIDENT']}.json"
        worker._settle_incident(incident, "recovered", pid=43)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", run)
    assert worker._supervise_locked("w", []) == 0
    assert worker.handle(["incidents", "w"]) == 0
    first, second = json.loads(capsys.readouterr().out)
    assert first["status"] == "restart_failed" and first["exit_status"] == 3
    assert first["restart_exit_status"] == 1
    assert second["status"] == "recovered" and second["pid"] == 43
    assert first["session"] == second["session"] == "saved"


async def test_textual_worker_view_submits_and_reconnects_without_owning_session(home):
    from rlmagent.tui.messages import AssistantBlock, ToolLine, UserBlock
    from rlmagent.tui.prompt import PromptInput
    from rlmagent.tui.transcript import TranscriptView
    from rlmagent.tui.worker_attach import WorkerAttachApp

    name = "session-saved"
    worker.log_line(name, "prompt", "obsolete log prompt")
    log = worker.folder(name) / "log.jsonl"
    worker._write(worker.folder(name) / "history.json", {
        "session": "saved", "log_offset": log.stat().st_size,
        "entries": [{"role": "user", "content": "earlier prompt"}],
    })
    reply = json.dumps({"time": 0, "kind": "reply", "text": "completed reply"}).encode()
    with log.open("ab") as stream:
        stream.write(reply[:8])
    first = WorkerAttachApp(name)
    async with first.run_test() as pilot:
        assert [block.text for block in first.query(UserBlock)] == ["earlier prompt"]
        assert not first.query(AssistantBlock)
        first.query_one(PromptInput).text = "new prompt"
        await pilot.press("enter")
        assert [row["text"] for row in worker._take_inbox(name)] == ["new prompt"]
        assert [block.text for block in first.query(UserBlock)] == ["earlier prompt"]
        base = worker.folder(name)
        request_id = "a" * 32
        worker._write(base / "meta.json", {"pid": os.getpid(), "beat": time.time()})
        worker._write(base / "input" / f"{request_id}.request.json", {
            "pid": os.getpid(), "expires": time.time() + 30, "approval": True,
        })
        first.query_one(PromptInput).text = f"/answer {request_id} once"
        await pilot.press("enter")
        assert worker._read(base / "input" / f"{request_id}.response.json") == {"value": "once"}
        assert worker._take_inbox(name) == []
        assert "already has an answer" in worker.answer_input(name, f"/answer {request_id} deny")
        turn_id = "b" * 32
        worker._write(base / "meta.json", {"pid": os.getpid(), "beat": time.time(), "turn_id": turn_id})
        await pilot.press("ctrl+c")
        assert (base / "interrupts" / turn_id).is_file()
        with log.open("ab") as stream:
            stream.write(reply[8:] + b"\n")
        first.poll_worker()
        first.poll_worker()
        await pilot.pause()
        assert [block.text for block in first.query(AssistantBlock)] == ["completed reply"]
        other_client = WorkerAttachApp(name)
        async with other_client.run_test() as other_pilot:
            worker.log_line(name, "model_delta", "shared ")
            await other_pilot.pause(0.4)
            assert other_client.query_one(TranscriptView).live_text == "shared "
            worker.log_line(name, "model_end", "shared live reply")
            worker.log_line(name, "reply", "shared live reply")
            worker.log_line(name, "tool_start", "python", call_id="tool", args={"code": "pass"})
            worker.log_line(name, "tool_end", "failed output", call_id="tool", is_error=True)
            await other_pilot.pause(0.4)
            assert [block.text for block in other_client.query(AssistantBlock)] == [
                "completed reply", "shared live reply"]
            assert [block.text for block in first.query(AssistantBlock)] == [
                "completed reply", "shared live reply"]
            assert other_client.query_one(ToolLine).has_class("tool-error")

    second = WorkerAttachApp(name)
    async with second.run_test():
        assert [block.text for block in second.query(AssistantBlock)] == [
            "completed reply", "shared live reply"]
        assert not (worker.folder(name) / "stop").exists()


async def test_supervised_node_worker_retries_named_gates_and_stops_failure_cycle(home, tmp_path, monkeypatch):
    import asyncio

    from _scripted import text_turn
    from rlmagent.cli.main import _build_run_parser
    from rlmagent.kernel.wasm import available
    from rlmagent.model.scripted import ReplayProvider

    if not available():
        pytest.skip("Node and Pyodide are unavailable")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker, "POLL_SECONDS", 0.03)
    finish = text_turn("```python\nFINAL(await goal_done('verified'))\n```")
    turns = [finish, finish]
    for attempt in range(3):
        turns.extend((
            text_turn("```python\nraise ValueError('same operation failed')\n```"),
            text_turn(f"```python\nFINAL('Different wording {attempt}')\n```"),
        ))
    provider = ReplayProvider(turns)
    monkeypatch.setattr(sys.modules["rlmagent.cli.main"], "_resolve_provider", lambda _: provider)
    ns = _build_run_parser().parse_args([
        "--supervised", "--no-approve", "--no-auto-improve", "--provider", "openai",
        "--model", "fixture", "--engine", "rlmness", "--kernel", "wasm",
    ])
    name = "session-autonomous-fixture"
    base = worker.folder(name)
    task = asyncio.create_task(worker.serve(name, ns))

    async def wait_until(check):
        async with asyncio.timeout(30):
            while not check():
                if task.done():
                    pytest.fail(f"Worker stopped early with status {task.result()}")
                await asyncio.sleep(0.03)

    async def send(text):
        receipt = worker.send(name, text)
        await wait_until(lambda: worker.receipt_of(name, receipt)["status"] in ("done", "failed"))
        assert worker.receipt_of(name, receipt)["status"] == "done"

    try:
        await wait_until(lambda: worker.state_of(name)["alive"] and worker.state_of(name).get("session"))
        await send("/goal verify the project")
        await send(f'/autonomous gate add build "{sys.executable}" -c "pass"')
        await send(f'/autonomous gate add check "{sys.executable}" -c "from pathlib import Path; assert Path(\'ready\').exists()"')
        await send("verify now")
        assert "check: exited 1" in (base / "log.jsonl").read_text(encoding="utf-8")
        (tmp_path / "ready").touch()
        await send("verify again")
        assert "Goal marked done." in (base / "log.jsonl").read_text(encoding="utf-8")
        await send("/goal recover from repeated errors")
        await send("/autonomous on")
        await wait_until(lambda: "the same failed operations repeated without progress"
                         in (base / "log.jsonl").read_text(encoding="utf-8"))
        assert len(provider.calls) == 8
    finally:
        (base / "stop").touch()
        assert await asyncio.wait_for(task, 10) == 0


def _cli(env, cwd, *args):
    return subprocess.run([sys.executable, "-m", "rlmagent.cli.main", "worker", *args],
                          capture_output=True, text=True, env=env, cwd=cwd, timeout=240)


def _wait(condition, seconds=120):
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        if condition():
            return True
        time.sleep(0.2)
    return False


def test_a_detached_worker_survives_its_starter_and_runs_a_scheduled_prompt(tmp_path):
    """A real process with a local model server, started from a process that then exits."""
    import http.server
    import threading

    code = "```python\nFINAL('ran')\n```"
    turn_entered = threading.Event()
    release_turn = threading.Event()
    interrupted_requests = []

    class Model(http.server.BaseHTTPRequestHandler):
        log_message = lambda *_: None  # noqa: E731

        def do_POST(self):
            request = self.rfile.read(int(self.headers.get("content-length", 0)))
            if b"crash this scheduled turn" in request:
                interrupted_requests.append(1)
                if not turn_entered.is_set():
                    turn_entered.set()
                    release_turn.wait(timeout=20)
                    return
            reply = code
            if b"verify supervised input" in request:
                reply = ("```python\nawait write_file('approved.txt', 'ok')\n"
                         "answer = await ask_user('Choose a color', ['blue', 'green'])\n"
                         "FINAL(answer)\n```")
            chunks = [
                {"id": "x", "model": "m", "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": reply}}]},
                {"id": "x", "model": "m", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "RLM_AGENT_HOME": str(home), "PYTHONPATH": str(Path(worker.__file__).parents[1]),
           "OPENAI_API_KEY": "test-not-real",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}/v1"}
    watcher = None
    supervised_name = None
    try:
        started = _cli(env, work, "start", "w", "--provider", "openai", "--model", "m",
                       "--no-approve", "--kernel", "local")
        assert started.returncode == 0, started.stderr
        meta = json.loads((home / "workers" / "w" / "meta.json").read_text())
        assert meta["pid"] != os.getpid()
        # The process that started it has exited; it is still there.
        assert "idle" in _cli(env, work, "list").stdout or "busy" in _cli(env, work, "list").stdout
        conflict = _cli(env, work, "start", "other", "--provider", "openai", "--model", "m",
                        "--no-approve", "--resume", meta["session"])
        assert conflict.returncode != 0
        assert "active writer" in (home / "workers" / "other" / "output.log").read_text()

        assert _cli(env, work, "schedule", "w", "first prompt", "--in", "1").returncode == 0
        log = home / "workers" / "w" / "log.jsonl"

        def replied():
            return log.exists() and '"reply"' in log.read_text(encoding="utf-8")

        assert _wait(replied), log.read_text(encoding="utf-8")
        assert "first prompt" in log.read_text(encoding="utf-8")
        assert '"model_end"' in log.read_text(encoding="utf-8")
        assert "done" in _cli(env, work, "jobs", "w").stdout

        import signal

        scheduled = _cli(env, work, "schedule", "w", "crash this scheduled turn", "--in", "0")
        assert scheduled.returncode == 0, scheduled.stderr
        job_id = scheduled.stdout.split()[1]
        assert turn_entered.wait(timeout=20)
        os.kill(meta["pid"], signal.SIGTERM)
        release_turn.set()
        meta_file = home / "workers" / "w" / "meta.json"

        def recovered():
            current = worker._read(meta_file) or {}
            return (current.get("state") == "idle" and current.get("pid") != meta["pid"]
                    and current.get("session") == meta["session"])

        assert _wait(recovered, seconds=30), meta_file.read_text(encoding="utf-8")
        job = worker._read(meta_file.parent / "jobs" / f"{job_id}.json")
        assert job["status"] == "interrupted" and job["runs"] == 1
        assert _cli(env, work, "send", "w", "after recovery").returncode == 0
        assert _wait(lambda: log.read_text(encoding="utf-8").count('"reply"') >= 2, seconds=30)

        for command in ("/goal verify the workspace", "/autonomous turns 2", "/autonomous"):
            assert _cli(env, work, "send", "w", command).returncode == 0
        assert _wait(lambda: "autonomous mode stopped: reached 2 turns"
                     in log.read_text(encoding="utf-8"), seconds=30)
        assert log.read_text(encoding="utf-8").count("Continue toward the goal.") == 2

        running = worker._read(meta_file)
        os.kill(running["supervisor_pid"], signal.SIGTERM)
        os.kill(running["pid"], signal.SIGTERM)
        assert _wait(lambda: not any(worker._pid_alive(running[key])
                                    for key in ("supervisor_pid", "pid")), seconds=10)
        recovered_fleet = _cli(env, work, "recover")
        assert recovered_fleet.returncode == 0, recovered_fleet.stderr
        assert "Recovered worker w" in recovered_fleet.stdout
        restored = worker._read(meta_file)
        assert restored["session"] == running["session"]
        assert restored["supervisor_pid"] != running["supervisor_pid"]
        assert _cli(env, work, "send", "w", "after supervisor loss").returncode == 0
        assert _wait(lambda: "after supervisor loss" in log.read_text(encoding="utf-8"), seconds=30)

        os.kill(restored["supervisor_pid"], signal.SIGTERM)
        os.kill(restored["pid"], signal.SIGTERM)
        assert _wait(lambda: not any(worker._pid_alive(restored[key])
                                    for key in ("supervisor_pid", "pid")), seconds=10)
        watcher = subprocess.Popen(
            [sys.executable, "-m", "rlmagent.cli.main", "worker", "watch"],
            cwd=work, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        def watcher_recovered():
            current = worker._read(meta_file) or {}
            return (current.get("state") == "idle" and current.get("pid") != restored["pid"]
                    and current.get("session") == restored["session"])

        assert _wait(watcher_recovered, seconds=30), meta_file.read_text(encoding="utf-8")
        duplicate = _cli(env, work, "watch")
        assert duplicate.returncode == 1 and "already running" in duplicate.stderr
        incidents = json.loads(_cli(env, work, "incidents", "w").stdout)
        assert [entry["status"] for entry in incidents] == ["recovered"] * 3
        assert [entry["reason"] for entry in incidents] == ["worker_exit", "fleet_recovery", "fleet_recovery"]
        assert incidents[0]["job_id"] == job_id and incidents[0]["turn_id"]
        assert incidents[0]["started_jobs"] == [job_id]
        assert incidents[0]["previous_pid"] == meta["pid"]
        assert incidents[0]["pid"] == running["pid"]
        assert len(interrupted_requests) == 1
        assert incidents[-1]["previous_supervisor_pid"] == restored["supervisor_pid"]
        assert incidents[-1]["session"] == restored["session"]

        stopped = _cli(env, work, "stop-all")
        assert "stopped" in stopped.stdout
        assert "not running" in _cli(env, work, "list").stdout
        assert "Recovered worker w" not in _cli(env, work, "recover").stdout

        cli_with_terminal_input = (
            "import sys; from rlmagent.cli.main import main; "
            "sys.stdin.isatty = lambda: True; raise SystemExit(main())"
        )
        interactive = [sys.executable, "-c", cli_with_terminal_input, "--repl",
                       "--provider", "openai", "--model", "m", "--no-approve", "--kernel", "local"]
        launched = subprocess.run(interactive, input="/detach\n", capture_output=True,
                                  text=True, env=env, cwd=work, timeout=45)
        assert launched.returncode == 0, launched.stderr
        paths = list((home / "workers").glob("session-*/meta.json"))
        assert len(paths) == 1
        supervised_name = paths[0].parent.name
        supervised = worker._read(paths[0])
        assert supervised_name == f"session-{supervised['session']}"
        reattached = subprocess.run([*interactive, "--resume", supervised["session"]],
                                    input="continue saved session\n/detach\n", capture_output=True,
                                    text=True, env=env, cwd=work, timeout=30)
        assert reattached.returncode == 0, reattached.stderr
        worker_log = paths[0].parent / "log.jsonl"
        assert _wait(lambda: '"text": "ran"' in worker_log.read_text(encoding="utf-8"), seconds=20)
        assert worker._read(paths[0])["pid"] == supervised["pid"]

        def client(text):
            return subprocess.run([*interactive, "--resume", supervised["session"]],
                                  input=f"{text}\n/detach\n", capture_output=True,
                                  text=True, env=env, cwd=work, timeout=30)

        def pending_input(approval):
            for path in (paths[0].parent / "input").glob("*.request.json"):
                row = worker._read(path)
                if row and row["approval"] is approval:
                    return path.name.split(".")[0]
            return None

        assert client("verify supervised input").returncode == 0
        assert _wait(lambda: pending_input(True), seconds=20)
        assert not (work / "approved.txt").exists()
        assert "Answer sent" in client(f"/answer {pending_input(True)} once").stdout
        assert _wait(lambda: pending_input(False), seconds=20)
        assert (work / "approved.txt").read_text() == "ok"
        assert "Answer sent" in client(f"/answer {pending_input(False)} 2").stdout
        assert _wait(lambda: '"text": "green"' in worker_log.read_text(encoding="utf-8"), seconds=20)
    finally:
        release_turn.set()
        if supervised_name is not None:
            _cli(env, work, "stop", supervised_name, "--wait", "5")
        if watcher is not None:
            _cli(env, work, "watch", "--stop")
            watcher.wait(timeout=20)
        _cli(env, work, "stop", "w", "--wait", "5")
        server.shutdown()
