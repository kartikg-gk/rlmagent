import io
from time import time

import pytest
from _scripted import text_turn
from rlmness import Allowance

from rlmagent.agents.approval import DENY, ONCE, Approver, Refused
from rlmagent.agents.patch import PatchError
from rlmagent.agents.workspace import Workspace
from rlmagent.harness.session.index import SessionCatalog
from rlmagent.prune import prune

DIFF = """--- a/one.py
+++ b/one.py
@@ -1,2 +1,2 @@
 keep
-old
+new
--- /dev/null
+++ b/two.py
@@ -0,0 +1 @@
+made
"""


async def test_a_patch_changes_every_file_after_one_approval(tmp_path):
    (tmp_path / "one.py").write_text("keep\nold\n")
    asked = []

    async def ask(request):
        asked.append(request.target)
        return ONCE
    await Workspace(tmp_path, Approver("ask", ask)).apply_patch(DIFF)
    assert (tmp_path / "one.py").read_text() == "keep\nnew\n"
    assert (tmp_path / "two.py").read_text() == "made\n"
    assert asked == ["one.py, two.py"]


async def test_a_patch_that_does_not_fit_changes_nothing(tmp_path):
    (tmp_path / "one.py").write_text("keep\nother\n")
    with pytest.raises(PatchError, match="not found"):
        await Workspace(tmp_path, Approver("auto")).apply_patch(DIFF)
    assert (tmp_path / "one.py").read_text() == "keep\nother\n"
    assert not (tmp_path / "two.py").exists()


async def test_a_patch_stays_inside_and_needs_approval(tmp_path):
    (tmp_path / "w").mkdir()
    with pytest.raises(ValueError, match="outside the workspace"):
        await Workspace(tmp_path / "w", Approver("auto")).apply_patch(
            "--- /dev/null\n+++ b/../evil.py\n@@ -0,0 +1 @@\n+x\n")
    (tmp_path / "w" / "one.py").write_text("keep\nold\n")

    async def deny(_request):
        return DENY
    with pytest.raises(Refused):
        await Workspace(tmp_path / "w", Approver("ask", deny)).apply_patch(DIFF)
    assert (tmp_path / "w" / "one.py").read_text() == "keep\nold\n"


async def test_a_moved_hunk_and_windows_line_endings_still_apply(tmp_path):
    (tmp_path / "one.py").write_bytes(b"top\r\nkeep\r\nold\r\n")
    diff = "--- a/one.py\n+++ b/one.py\n@@ -1,2 +1,2 @@\n keep\n-old\n+new\n"
    await Workspace(tmp_path, Approver("auto")).apply_patch(diff)
    assert (tmp_path / "one.py").read_bytes() == b"top\r\nkeep\r\nnew\r\n"


async def test_questions_reach_the_user_or_return_an_error(tmp_path):
    workspace = Workspace(tmp_path, Approver("deny"))
    assert (await workspace.ask_user("which?")).startswith("Error:")

    async def answer(question, choices):
        return choices[1]
    workspace.approver.question = answer
    assert await workspace.ask_user("which?", ["a", "b"]) == "b"

    async def silent(question, choices):
        return None
    workspace.approver.question = silent
    assert (await workspace.ask_user("which?")).startswith("Error:")


async def test_command_output_arrives_while_it_runs(tmp_path):
    seen = []
    workspace = Workspace(tmp_path, Approver("auto"))
    workspace.on_output = seen.append
    result = await workspace.run("echo first; sleep 1; echo second")
    assert result["exit"] == 0 and "second" in result["output"]
    assert len(seen) >= 2 and "first" in seen[0] and "second" not in seen[0]


async def _backend_with(tmp_path, turns):
    from types import SimpleNamespace

    from rlmagent.backend import BackendProvider

    model = __import__("rlmagent.model.scripted", fromlist=["x"]).ReplayProvider(turns)
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("auto"), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    return BackendProvider(model, model, tree)


async def test_saved_variables_roll_back_to_a_checkpoint(tmp_path):
    backend = await _backend_with(tmp_path, [text_turn("x")])
    backend.state.variables["a"] = {"value": 1}
    saved = backend.checkpoint_state()
    backend.state.variables["a"]["value"] = 2
    backend.state.variables["b"] = {"value": 3}
    backend.restore_state(saved)
    assert backend.state.variables == {"a": {"value": 1}}
    backend.state.variables["a"]["value"] = 9
    assert saved["variables"]["a"]["value"] == 1


async def test_too_large_variables_are_not_copied(tmp_path, monkeypatch):
    backend = await _backend_with(tmp_path, [text_turn("x")])
    monkeypatch.setattr("rlmagent.backend.MAX_CHECKPOINT_CHARS", 10)
    backend.state.variables["a"] = {"value": "x" * 100}
    assert backend.checkpoint_state() is None


def _session(catalog, base, sid, cwd, age_days):
    catalog.upsert(catalog.prepare(session_id=sid, vault_path=base / f"{sid}.jsonl",
                                   cwd=str(cwd), model="m", provider="p"))
    from dataclasses import replace

    meta = catalog.get(sid)
    entries = catalog._load_deduped()
    entries[sid] = replace(meta, updated_at=time() - age_days * 86400)
    catalog.remove({sid})
    catalog.upsert(entries[sid])
    (base / f"{sid}.jsonl").write_text("x")
    (base / f"{sid}.backend.json").write_text("{}")


def test_prune_lists_first_and_removes_only_what_is_old(tmp_path):
    base, home = tmp_path / "store", tmp_path / "home"
    base.mkdir()
    catalog = SessionCatalog(base)
    _session(catalog, base, "old1", tmp_path / "gone", 40)
    _session(catalog, base, "new1", tmp_path / "here", 1)

    out = io.StringIO()
    prune(base, home, 30, out=out)
    assert "1 of 2" in out.getvalue() and "Nothing was removed" in out.getvalue()
    assert (base / "old1.jsonl").exists()

    prune(base, home, 30, apply=True, out=io.StringIO())
    assert not (base / "old1.jsonl").exists() and not (base / "old1.backend.json").exists()
    assert (base / "new1.jsonl").exists()
    assert [m.session_id for m in catalog.list_all()] == ["new1"]




def test_pages_are_read_without_their_markup():
    from rlmagent.network import readable

    text = readable("<html><style>a{}</style><script>x()</script><h1>Title</h1><p>Some  text</p></html>")
    assert text == "Title\nSome text"


def test_prompts_find_files_without_an_extension_and_trim_links(tmp_path):
    from rlmagent.prompt_refs import references

    (tmp_path / "Makefile").write_text("all:\n")
    refs = references("fix Makefile, see (https://example.com/a).", tmp_path)
    assert [f["path"] for f in refs["files"]] == ["Makefile"]
    assert refs["links"] == ["https://example.com/a"]


async def test_instruction_documents_are_bounded(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 70_000)
    workspace = Workspace(tmp_path, Approver("deny"))
    assert await workspace.instruction_docs() == ["AGENTS.md"]
    with pytest.raises(ValueError, match="larger"):
        await workspace.instruction_docs("AGENTS.md")


async def test_undo_goes_back_several_turns_or_lists_them(tmp_path):
    from types import SimpleNamespace

    from rlmagent.conversation import CodingSession

    restored = []
    session = SimpleNamespace(
        _undo_stack=[("c1", None, ["t1"], None), ("c2", None, ["t2"], None)], _undo_problem=None,
        _vault=None, _harness=SimpleNamespace(set_messages=lambda m: None),
        shadow=SimpleNamespace(available=True, skipped=set(), label=lambda c: f"turn {c}",
                               restore=lambda c: restored.append(c) or []))
    listed = await CodingSession._cmd_undo(session, "list")
    assert "1. turn c2" in listed and "2. turn c1" in listed
    assert "Only 2" in await CodingSession._cmd_undo(session, "5")
    await CodingSession._cmd_undo(session, "2")
    assert restored == ["c1"] and session._undo_stack == []


async def test_several_answers_can_be_requested(tmp_path):
    workspace = Workspace(tmp_path, Approver("deny"))

    async def answer(question, choices, multiple=False):
        return choices if multiple else choices[0]
    workspace.approver.question = answer
    assert await workspace.ask_user("q", ["a", "b"], multiple=True) == ["a", "b"]
    assert await workspace.ask_user("q", ["a", "b"]) == "a"


async def test_edits_keep_each_files_line_endings(tmp_path):
    (tmp_path / "unix.py").write_bytes(b"a\nb\n")
    (tmp_path / "dos.py").write_bytes(b"a\r\nb\r\n")
    workspace = Workspace(tmp_path, Approver("auto"))
    await workspace.edit_file("unix.py", "a\nb", "a\nc")
    await workspace.edit_file("dos.py", "a\nb", "a\nc")
    await workspace.write_file("new.py", "x\n")
    assert (tmp_path / "unix.py").read_bytes() == b"a\nc\n"
    assert (tmp_path / "dos.py").read_bytes() == b"a\r\nc\r\n"
    assert (tmp_path / "new.py").read_bytes() == b"x\n"


def test_an_unfinished_conversation_is_noticed():
    from types import SimpleNamespace as Entry

    from rlmagent.conversation import CodingSession

    def state(*entries):
        return CodingSession.interrupted.fget(Entry(_harness=Entry(transcript=list(entries))))

    assert not state()
    assert not state(Entry(role="user"), Entry(role="assistant", stop_reason="stop"))
    assert state(Entry(role="user"))
    assert state(Entry(role="assistant", stop_reason="aborted"))
    assert state(Entry(role="assistant", stop_reason="toolUse"), Entry(role="toolResult"))


def test_provider_failures_say_what_to_change():
    from types import SimpleNamespace as Fake

    from rlmagent.tui.model_screen import _explain

    def failure(status):
        error = RuntimeError("x")
        error.response = Fake(status_code=status)
        return error

    assert "key" in _explain(failure(401)) and "base URL" in _explain(failure(404))
    assert "reach" in _explain(OSError("down"))


def test_recent_models_can_be_removed(tmp_path, monkeypatch):
    from rlmagent.config.store import forget_model, recent_models, remember_model

    monkeypatch.setenv("RLM_AGENT_CONFIG_DIR", str(tmp_path))
    remember_model("p", "a")
    remember_model("p", "b")
    forget_model("p", "a")
    assert [r["model"] for r in recent_models()] == ["b"]


def test_sessions_from_a_newer_version_are_refused_clearly():
    import json

    from rlmagent.harness.session.store import RecordParseError, deserialize_record

    old = deserialize_record(json.dumps({"type": "session_info", "id": "a", "timestamp": 1.0}))
    assert old.format == 1
    with pytest.raises(RecordParseError, match="newer"):
        deserialize_record(json.dumps({"type": "session_info", "id": "a", "timestamp": 1.0,
                                       "format": 99}))


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="process groups")
def test_stopping_a_process_that_just_ended_is_harmless(monkeypatch):
    from types import SimpleNamespace

    from rlmagent.model.external import cli

    def gone(_pid):
        raise ProcessLookupError

    monkeypatch.setattr(cli.os, "getpgid", gone)
    cli.kill_tree(SimpleNamespace(returncode=None, pid=1))
