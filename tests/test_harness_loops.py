"""Evidence, expiry, task-based checks, digests, capabilities and extensions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rlmagent.playbook.apply import apply_edits, validate
from rlmagent.playbook.improver import AUTO_MEMORY_TASKS, Improver
from rlmagent.playbook.store import Playbook, load, save
from rlmagent.plugins import as_tool, contributions, load_extensions, unload_extensions, EXTENSIONS_DIR_ENV
from rlmagent.refinement.controller import DIGEST_CHARS, _digest
from rlmagent.refinement.evidence import EvidenceLog


class FakeSession:
    def __init__(self, replies=()):
        self.transcript = []
        self.replies = list(replies)
        self.injected = []
        self.notes = []
        self.evidence = None

    async def _utility_completion(self, _prompt, _system):
        return self.replies.pop(0)

    def inject(self, text):
        self.injected.append(text)

    async def note_playbook_change(self, change):
        self.notes.append(change)


def message(text, role="user"):
    return SimpleNamespace(text=text, role=role)


def test_evidence_separates_saved_reached_and_what_followed(tmp_path):
    path = tmp_path / "e.jsonl"
    log = EvidenceLog(path)
    session = SimpleNamespace(transcript=[], _system="base")
    session.transcript = [SimpleNamespace(is_error=True, tool_call_id="1"),
                          SimpleNamespace(is_error=True, tool_call_id="2")]
    log.task_finished(session)
    assert log.last_failed == 2

    log.saved("c1", "refine", ["run-tests"], "keep it")
    # Saved but not sent: the report must not count it as having reached the model.
    session.transcript = [*session.transcript, SimpleNamespace(is_error=False, tool_call_id="3")]
    log.task_finished(session)
    session._system = "base\nprompt run-tests (Run tests):\nuse pytest"
    log.task_finished(session)

    assert log.rate("c1", "great").startswith("A verdict")
    assert log.rate("nope", "better").startswith("No saved")
    assert "rated better" in log.rate("c1", "better")

    again = EvidenceLog(path)
    report = again.report()
    assert "reached the model: in 1 of 2 later tasks" in report
    assert "2.00 over 1 tasks before" in report
    assert "0.00 over 2 tasks after" in report
    assert "your verdict: better" in report
    assert "not proof" in report


def test_a_name_inside_another_word_does_not_count_as_reached():
    log = EvidenceLog(None)
    log.saved("c", "refine", ["test"])
    session = SimpleNamespace(transcript=[message("please test this")], _system="prompt tester (x)")
    assert log._reached(session) == set()


def test_evidence_reads_a_backends_cells_and_host_calls():
    log = EvidenceLog(None)
    backend = SimpleNamespace(instructions="", events=[
        {"kind": "step", "step": 0, "code": "opening", "error": False},
        {"kind": "step", "step": 1, "code": "x", "error": True},
        {"kind": "host_result", "result": "Error: refused"},
        {"kind": "host_result", "result": "fine"},
    ])
    log.task_finished(SimpleNamespace(transcript=[], _system="", backend=backend))
    assert log.rows[-1]["calls"] == 3 and log.rows[-1]["failed"] == 2
    # Counted once, not again on the next task.
    log.task_finished(SimpleNamespace(transcript=[], _system="", backend=backend))
    assert log.rows[-1]["calls"] == 0


async def test_an_automatic_memory_expires_and_a_rule_stays(tmp_path):
    proposal = json.dumps({"summary": "s", "edits": [
        {"action": "create", "kind": "memory", "id": "paused", "title": "t", "content": "c"},
        {"action": "create", "kind": "rule", "id": "tests", "title": "t", "content": "c"},
    ]})
    session = FakeSession([proposal])
    improver = Improver(session, tmp_path / "local.json", None)
    await improver.improve("x", trigger="auto:tasks")
    assert improver.local.entries["paused"].expires_after_tasks == AUTO_MEMORY_TASKS
    assert improver.local.entries["tests"].expires_after_tasks is None
    improver.show_digest()
    assert "paused" in session.injected[-1]

    for _ in range(AUTO_MEMORY_TASKS - 1):
        await improver._task_finished()
    assert "paused" in improver.local.entries
    await improver._task_finished()
    assert list(improver.local.entries) == ["tests"]
    assert improver.local.history[-1]["trigger"] == "expire"
    improver.show_digest()
    assert "paused" not in session.injected[-1]
    # Survives a restart, counter included.
    assert load(tmp_path / "local.json").tasks == AUTO_MEMORY_TASKS


async def test_expiry_is_undoable_and_an_asked_for_memory_keeps(tmp_path):
    proposal = json.dumps({"summary": "s", "edits": [
        {"action": "create", "kind": "memory", "id": "fact", "title": "t", "content": "c"}]})
    session = FakeSession([proposal])
    improver = Improver(session, tmp_path / "local.json", None)
    await improver.improve("keep this", trigger="user")
    assert improver.local.entries["fact"].expires_after_tasks is None


async def test_the_last_entry_expiring_is_still_told_to_the_model(tmp_path):
    session = FakeSession()
    improver = Improver(session, tmp_path / "local.json", None)
    improver.apply([{"action": "create", "kind": "memory", "id": "a", "title": "t", "content": "c",
                     "expires_after_tasks": 1}], trigger="auto:tasks")
    improver.show_digest()
    await improver._task_finished()
    improver.show_digest()
    assert "no longer apply" in session.injected[-1]


def test_a_bad_lifetime_is_refused():
    edit = {"action": "create", "kind": "memory", "id": "a", "title": "t", "content": "c"}
    assert validate({**edit, "expires_after_tasks": 0})
    assert validate({**edit, "expires_after_tasks": "soon"})
    assert validate({**edit, "expires_after_tasks": True})
    assert validate({**edit, "expires_after_tasks": 2}) is None


def test_old_playbook_files_still_load(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"entries": [
        {"id": "a", "kind": "rule", "title": "t", "content": "c", "scope": "local"}], "history": []}))
    book = load(path)
    assert book.entries["a"].expires_after_tasks is None and book.tasks == 0
    save(path, book)
    assert load(path).entries["a"].id == "a"


async def test_checks_count_tasks_and_trouble_comes_first(tmp_path):
    session = FakeSession(['{"worth_it": false}', '{"worth_it": false}'])
    improver = Improver(session, None, None)
    improver.every, improver.cooldown = 2, 0.0
    triggers = []
    real = improver._auto

    async def spy(trigger):
        triggers.append(trigger)
        await real(trigger)

    improver._auto = spy
    # Many model turns in one task are still one task.
    session.transcript = [message("x", "assistant") for _ in range(50)]
    await improver.at_boundary()
    assert triggers == []
    await improver.at_boundary()
    assert triggers == ["auto:tasks"]
    session.evidence = SimpleNamespace(last_failed=3)
    await improver.at_boundary()
    assert triggers == ["auto:tasks", "auto:trouble"]


def test_long_guidance_goes_in_as_a_digest():
    short = SimpleNamespace(id="a", content="brief")
    long = SimpleNamespace(id="b", content="x" * (DIGEST_CHARS + 50))
    assert _digest(short) == "brief"
    digest = _digest(long)
    assert len(digest) < len(long.content) and 'await guidance("b")' in digest


@pytest.fixture
def ext_dir(tmp_path, monkeypatch):
    directory = tmp_path / "extensions"
    directory.mkdir()
    monkeypatch.setenv(EXTENSIONS_DIR_ENV, str(directory))
    yield directory
    unload_extensions()


async def test_an_extension_contributes_functions_as_tools(ext_dir):
    (ext_dir / "words.py").write_text(
        "def count(text, base=0):\n    'Count words.'\n    return base + len(text.split())\n"
        "def register():\n    return {'count': count, 'not a name': count, 'value': 3}\n",
        encoding="utf-8",
    )
    tools, functions = contributions(load_extensions(include_project=False))
    assert tools == [] and list(functions) == ["count"]
    tool = as_tool("count", functions["count"])
    assert tool.parameters["required"] == ["text"]
    outcome = await tool.run("id", {"text": "a b c"})
    assert outcome.content[0].text == "3"


async def test_a_project_extension_reaches_a_cell_and_reload_picks_up_changes(ext_dir, tmp_path, monkeypatch):
    from rlmagent.plugins import reload_extensions

    path = ext_dir / "one.py"
    path.write_text("def register():\n    return {'first': lambda: 1}\n", encoding="utf-8")
    assert "first" in contributions(load_extensions(include_project=False))[1]
    path.write_text("def register():\n    return {'second': lambda: 2}\n", encoding="utf-8")
    assert list(contributions(reload_extensions(include_project=False))[1]) == ["second"]


def test_apply_records_when_a_note_was_made():
    book = Playbook(tasks=7)
    apply_edits(book, [{"action": "create", "kind": "memory", "id": "a", "title": "t",
                        "content": "c", "expires_after_tasks": 1}],
                baseline={}, scope="local", trigger="auto:tasks", summary="")
    assert book.entries["a"].made_at_task == 7


async def test_automatic_proposals_can_wait_for_review(tmp_path):
    import json as _json
    from types import SimpleNamespace

    from rlmagent.playbook.improver import Improver

    proposal = {"summary": "keep the check", "rationale": "seen twice", "edits": [
        {"action": "create", "kind": "memory", "id": "check-cmd", "title": "Check",
         "content": "Run make check"}]}

    class Session(SimpleNamespace):
        transcript = []

        async def _utility_completion(self, request, system):
            return _json.dumps(proposal)

        def inject(self, text):
            pass

    improver = Improver(Session(), tmp_path / "local.json", None)
    improver.hold_auto = True
    note = await improver.improve(None, trigger="auto:tasks")
    assert note.startswith("Held proposal 1") and not improver.entries()
    assert "create memory check-cmd: Run make check" in await improver.command("pending")
    assert (await improver.command("accept 5")).startswith("Use /improve accept N")
    await improver.command("accept 1")
    assert [e.id for e in improver.entries()] == ["check-cmd"] and not improver.held
    await improver.improve(None, trigger="auto:tasks")
    assert (await improver.command("reject 1")).startswith("Dropped")
    assert await improver.command("pending") == "No proposals are waiting."
