import argparse
import asyncio
import shutil
import sys
import time

import pytest

from rlmagent.agents.approval import Approver
from rlmagent.agents.workspace import Workspace
from rlmagent.harness.engine import argument_problem
from rlmagent.harness.session.index import SessionCatalog
from rlmagent.runtime import _resume_folder_problem
from rlmagent.undo import Shadow

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


async def test_a_session_does_not_resume_in_another_project(tmp_path, monkeypatch):
    project, other, store = tmp_path / "project", tmp_path / "other", tmp_path / "store"
    project.mkdir(), other.mkdir(), store.mkdir()
    catalog = SessionCatalog(store)
    catalog.upsert(catalog.prepare(session_id="abc", vault_path=store / "abc.jsonl",
                                   cwd=str(project), model="m", provider="p"))
    ns = argparse.Namespace(print_mode=True, prompt="x")

    monkeypatch.chdir(project)
    assert await _resume_folder_problem(ns, "abc", store) is None
    monkeypatch.chdir(other)
    assert "belongs to" in await _resume_folder_problem(ns, "abc", store)
    ns.resume_anywhere = True
    assert await _resume_folder_problem(ns, "abc", store) is None


async def test_cancelling_a_command_stops_what_it_started(tmp_path):
    beats = tmp_path / "beats"
    child = (tmp_path / "child.py")
    child.write_text(
        "import time\n"
        f"while True:\n    open({str(beats)!r}, 'a').write('x'); time.sleep(0.1)\n")
    python = sys.executable.replace("\\", "/")
    script = str(child).replace("\\", "/")
    run = asyncio.create_task(Workspace(tmp_path, Approver("auto")).run(f'"{python}" "{script}"; echo end'))
    for _ in range(100):
        await asyncio.sleep(0.1)
        if beats.exists() and beats.stat().st_size > 2:
            break
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    time.sleep(0.5)
    size = beats.stat().st_size
    time.sleep(0.6)
    assert beats.stat().st_size == size


@needs_git
def test_undo_notices_files_it_cannot_cover(tmp_path):
    shadow = Shadow(tmp_path / "w", tmp_path / "home")
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "big.bin").write_bytes(b"x" * 5_000_001)
    (tmp_path / "w" / "a.txt").write_text("one")
    commit = shadow.snapshot("start")
    (tmp_path / "w" / "a.txt").write_text("two")
    assert shadow.restore(commit) == ["a.txt"]
    assert (tmp_path / "w" / "a.txt").read_text() == "one"
    assert shadow.skipped == {"big.bin"}


async def test_a_failed_undo_keeps_the_turn_and_says_so(monkeypatch):
    from types import SimpleNamespace

    from rlmagent.conversation import CodingSession

    def broken(_commit):
        raise RuntimeError("these files did not come back: a.txt")

    session = SimpleNamespace(_undo_stack=[("c", None, [])], _undo_problem=None,
                              shadow=SimpleNamespace(restore=broken, available=True, skipped=set()))
    answer = await CodingSession._cmd_undo(session, "")
    assert "Undo failed" in answer and "a.txt" in answer and session._undo_stack
    session._undo_problem = "disk full"
    assert "no snapshot" in await CodingSession._cmd_undo(session, "")


async def test_binary_files_get_a_clear_answer(tmp_path):
    (tmp_path / "data.txt").write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ValueError, match="binary"):
        await Workspace(tmp_path, Approver("deny")).read_file("data.txt")

    from rlmagent.tools.files import _looks_binary

    assert _looks_binary(tmp_path / "data.txt")
    (tmp_path / "ok.txt").write_text("fine")
    assert not _looks_binary(tmp_path / "ok.txt")


def test_tool_arguments_are_checked_against_the_schema():
    schema = {"type": "object", "required": ["path"], "properties": {
        "path": {"type": "string"}, "limit": {"type": "integer"},
        "either": {"type": ["string", "null"]}}}
    assert argument_problem(schema, {"path": "a", "limit": 3, "either": None}) is None
    assert argument_problem(schema, {"path": "a", "limit": 3.0}) is None
    assert "missing" in argument_problem(schema, {"limit": 1})
    assert "'limit' must be integer" in argument_problem(schema, {"path": "a", "limit": "3"})
    assert "'limit' must be integer" in argument_problem(schema, {"path": "a", "limit": True})
    assert argument_problem(schema, {"path": "a", "unknown": 1}) is None
