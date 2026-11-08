import asyncio
import sys
from types import SimpleNamespace

import pytest

from rlmagent.hints import EMPTY_CELL, hint_for
from rlmagent.jobs import Jobs
from rlmagent.policy import Policy, blocked_url, cap_output, git_refusal

PY = sys.executable.replace("\\", "/")


def _jobs(tmp_path, approve=None, check=None):
    return Jobs(str(tmp_path), tmp_path / "jobs", approve, check)


async def test_a_job_runs_in_the_background_and_is_read_from_a_cursor(tmp_path):
    jobs = _jobs(tmp_path)
    try:
        started = await jobs.job_start(f'"{PY}" -c "print(1); print(2)"')
        done = await jobs.job_wait(started["id"], timeout=30)
        assert done["done"] and done["exit"] == 0
        first = await jobs.job_read(started["id"])
        assert first["text"].split() == ["1", "2"] and first["done"]
        again = await jobs.job_read(started["id"], first["cursor"])
        assert again["text"] == "" and again["cursor"] == first["cursor"]
        assert [j["id"] for j in await jobs.jobs_list()] == [started["id"]]
        with pytest.raises(ValueError, match="no job"):
            await jobs.job_read("nope")
    finally:
        await jobs.close()


async def test_timeouts_stops_and_services(tmp_path):
    jobs = _jobs(tmp_path)
    slow = f'"{PY}" -c "import time; time.sleep(30)"'
    try:
        timed = await jobs.job_start(slow, timeout=0.5)
        assert (await jobs.job_wait(timed["id"], 30))["timed_out"]
        long = await jobs.job_start(slow)
        stopped = await jobs.job_stop(long["id"])
        assert stopped["done"] and stopped["stopped"]
        server = (f'"{PY}" -c "import time; print(\'listening on 8000\', flush=True); '
                  f'time.sleep(30)"')
        svc = await jobs.service_start("web", server, ready="listening", wait=20)
        assert svc["ready"] and svc["service"] and not svc["done"]
        with pytest.raises(ValueError, match="is up; stop it"):
            await jobs.service_start("web", server)
        assert (await jobs.job_stop("web"))["stopped"]
    finally:
        await jobs.close()


async def test_watches_see_output_paths_and_messages(tmp_path):
    jobs = _jobs(tmp_path)
    try:
        job = await jobs.job_start(f'"{PY}" -c "import time; time.sleep(1); print(5)"')
        seen = await jobs.watch(job=job["id"], timeout=20)
        assert seen["event"] in ("output", "ended")
        (tmp_path / "w").mkdir()

        async def touch():
            await asyncio.sleep(0.5)
            (tmp_path / "w" / "new.txt").write_text("x")

        asyncio.create_task(touch())
        changed = await jobs.watch(path="w", timeout=10)
        assert changed["event"] == "path" and changed["paths"][0].endswith("new.txt")
        box = [[], [{"from": "a", "text": "hi"}]]
        got = await jobs.watch(inbox=lambda: box.pop(0), timeout=5)
        assert got == {"event": "message", "messages": [{"from": "a", "text": "hi"}]}
        assert (await jobs.watch(path="w", timeout=0.3))["event"] == "timeout"
    finally:
        await jobs.close()


async def test_jobs_ask_first_and_honour_the_git_guard(tmp_path):
    asked = []

    async def approve(command):
        asked.append(command)
        raise PermissionError("refused")

    policy = Policy(allow_git=False)
    jobs = _jobs(tmp_path, approve, lambda c: git_refusal(c, policy))
    with pytest.raises(ValueError, match="git push touches other branches"):
        await jobs.job_start("git push origin main")
    with pytest.raises(PermissionError):
        await jobs.job_start("echo hi")
    assert asked == ["echo hi"]


@pytest.mark.parametrize("command,refused", [
    ("git status && git diff", False),
    ("git add -A; git commit -m x", False),
    ("git -C sub log --oneline", False),
    ("git push", True),
    ("FOO=1 git checkout other", True),
    ("echo x | git branch -a", True),
    ("git log --all", True),
    ("git config user.name x", True),
    ("ls; git frobnicate", True),
    ("echo git push", False),
])
def test_git_guard(command, refused):
    assert bool(git_refusal(command, Policy(allow_git=False))) is refused
    assert git_refusal(command, Policy()) is None


def test_output_cap_and_block_list():
    text = "a" * 5000 + "MIDDLE" + "b" * 5000
    capped = cap_output(text, 3000)
    assert len(capped) < 3200 and "bytes of output left out" in capped
    assert capped.startswith("aaa") and capped.endswith("bbb") and "MIDDLE" not in capped
    assert cap_output("short", 3000) == "short"
    policy = Policy(search_block=["example.com", "*.bad.org", "https://x.io"])
    assert blocked_url("https://docs.example.com/a", policy)
    assert blocked_url("https://a.bad.org/", policy)
    assert blocked_url("https://x.io/page", policy) and not blocked_url("http://x.io/", policy)
    assert not blocked_url("https://good.org", policy)


def test_hints_for_familiar_failures():
    assert hint_for("", "") == EMPTY_CELL
    assert "await" in hint_for("x = read_file('a')", "RuntimeWarning: coroutine 'read_file' was never awaited")
    assert "f-string" in hint_for("x", "SyntaxError: f-string: unmatched '('")
    assert "micropip" in hint_for("import yaml", "ModuleNotFoundError: No module named 'yaml'")
    assert "job_start" in hint_for("x", "CellTimeout: no response within 30s")
    assert hint_for("print(1)", "ValueError: bad") is None


def test_shaped_cells_cap_output_and_add_hints():
    from rlmagent.backend import shape_cells

    class Runtime:
        def execute(self, code):
            if code == "big":
                return SimpleNamespace(stdout="x" * 50_000, error=None)
            return SimpleNamespace(stdout="", error="NameError: name 'FINAL' is not defined")

    runtime = Runtime()
    shape_cells(runtime, Policy(max_tool_output_bytes=1000))
    assert len(runtime.execute("big").stdout) < 1200
    assert "Hint: FINAL is only defined" in runtime.execute("def f(): FINAL(1)").error


async def test_workspace_run_honours_the_git_guard(tmp_path):
    from rlmagent.agents.approval import Approver
    from rlmagent.agents.workspace import Workspace

    ws = Workspace(tmp_path, Approver("auto"))
    ws.policy = Policy(allow_git=False)
    with pytest.raises(ValueError, match="git fetch touches other branches"):
        await ws.run("git fetch")


async def test_compactions_stop_at_the_limit(tmp_path):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    session = await CodingSession.create(provider=ReplayProvider([text_turn("ok")]),
                                         provider_name="replay", model="m", system="s",
                                         tools=[], cwd=str(tmp_path))
    session.policy = Policy(max_compactions=0)
    with pytest.raises(RuntimeError, match="allows 0 compaction"):
        await session.compact()
    await session.shutdown()
