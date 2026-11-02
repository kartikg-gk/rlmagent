"""Workers: the files they are reached through, and a real detached process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
    worker.send("w", "one")
    worker.send("w", "two")
    assert [m["text"] for m in worker._take_inbox("w")] == ["one", "two"]
    assert worker._take_inbox("w") == []


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

    class Model(http.server.BaseHTTPRequestHandler):
        log_message = lambda *_: None  # noqa: E731

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            chunks = [
                {"id": "x", "model": "m", "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": code}}]},
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
    try:
        started = _cli(env, work, "start", "w", "--provider", "openai", "--model", "m",
                       "--no-approve", "--kernel", "local")
        assert started.returncode == 0, started.stderr
        meta = json.loads((home / "workers" / "w" / "meta.json").read_text())
        assert meta["pid"] != os.getpid()
        # The process that started it has exited; it is still there.
        assert "idle" in _cli(env, work, "list").stdout or "busy" in _cli(env, work, "list").stdout

        assert _cli(env, work, "schedule", "w", "first prompt", "--in", "1").returncode == 0
        log = home / "workers" / "w" / "log.jsonl"

        def replied():
            return log.exists() and '"reply"' in log.read_text(encoding="utf-8")

        assert _wait(replied), log.read_text(encoding="utf-8")
        assert "first prompt" in log.read_text(encoding="utf-8")
        assert "done" in _cli(env, work, "jobs", "w").stdout

        stopped = _cli(env, work, "stop", "w")
        assert "stopped" in stopped.stdout
        assert "not running" in _cli(env, work, "list").stdout
    finally:
        _cli(env, work, "stop", "w", "--wait", "5")
        server.shutdown()
