import json
import os
import shutil

import pytest
from _scripted import code_turn, text_turn
from test_improve import _submit
from rlmness import Allowance

from rlmagent.agents.tree import AgentTree
from rlmagent.conversation import CodingSession
from rlmagent.model.scripted import ReplayProvider
from rlmagent.prompt_refs import note, references
from rlmagent.tools import build_tool_registry
from rlmagent.tools._shared import set_workspace, validate_file_path
from rlmagent.undo import Shadow

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


@pytest.fixture
def confined(tmp_path):
    set_workspace(tmp_path / "proj")
    yield tmp_path / "proj"
    set_workspace(None)


def test_file_tools_stay_inside_the_workspace(tmp_path, confined):
    confined.mkdir()
    assert validate_file_path(str(confined / "a.py"))
    with pytest.raises(ValueError, match="outside the workspace"):
        validate_file_path(str(tmp_path / "secret.txt"))
    with pytest.raises(ValueError, match="outside the workspace"):
        validate_file_path(str(confined / ".." / "secret.txt"))
    try:
        os.symlink(tmp_path, confined / "out", target_is_directory=True)
    except OSError:
        pytest.skip("no symlink permission")
    with pytest.raises(ValueError, match="outside the workspace"):
        validate_file_path(str(confined / "out" / "secret.txt"))


def test_a_prompt_brings_the_files_and_links_it_names(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text("x = 1\n")
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")
    (tmp_path.parent / "outside.py").write_text("no")
    refs = references("Fix `src/calc.py`, see @notes.md and blob.bin, ../outside.py, "
                      "missing.py and https://example.com/spec.", tmp_path)
    assert [f["path"] for f in refs["files"]] == ["src/calc.py", "notes.md"]
    assert refs["links"] == ["https://example.com/spec"]
    assert "src/calc.py" in note(refs) and note(references("nothing here", tmp_path)) == ""


@needs_git
def test_the_shadow_repo_restores_files_and_skips_secrets(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "a.py").write_text("one\n")
    (work / ".env").write_text("KEY=secret\n")
    shadow = Shadow(work, tmp_path / "home")
    commit = shadow.snapshot("before")
    (work / "a.py").write_text("two\n")
    (work / "new.py").write_text("added\n")
    changed = shadow.restore(commit)
    assert (work / "a.py").read_text() == "one\n" and not (work / "new.py").exists()
    assert set(changed) == {"a.py", "new.py"}
    assert ".env" not in shadow._git("ls-files")
    assert not (work / ".git").exists()


async def _session(tmp_path, streams):
    provider = ReplayProvider(streams)
    tree = AgentTree(provider=provider, provider_name="replay", model="m", cwd=str(tmp_path),
                     allowance=Allowance(max_cost=10.0), sessions_dir=None,
                     system_for=lambda n, t: "s", first_message_for=lambda n: n.task)
    await tree.start()
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name="replay", model="m", system="s",
        tools=build_tool_registry(tree.root_kernel()), cwd=str(tmp_path))
    await session.set_name("t")
    session.agent_tree = tree
    tree.attach("root", session)
    return session, provider


async def test_named_files_are_waiting_in_the_kernel(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    session, provider = await _session(tmp_path, [
        code_turn("print(PROMPT['files'][0]['content'])"), text_turn("ok")])
    try:
        await _submit(session, "fix calc.py")
        assert "PROMPT['files'] is a list" in json.dumps([str(m) for m in provider.calls[0][2]])
        outputs = [m.text for m in session.transcript if getattr(m, "role", "") == "toolResult"]
        assert "return a - b" in outputs[0]
    finally:
        await session.shutdown()


@needs_git
async def test_undo_puts_back_the_files_and_the_messages(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "calc.py").write_text("old\n")
    session, _ = await _session(work, [
        text_turn("first"),
        code_turn("open('calc.py', 'w').write('new\\n')"), text_turn("changed"),
    ])
    session.shadow = Shadow(work, tmp_path / "home")
    try:
        await _submit(session, "hello")
        before = len(session.transcript)
        await _submit(session, "change it")
        assert (work / "calc.py").read_text() == "new\n"
        reply = await session.handle_command("/undo")
        assert "calc.py" in reply
        assert (work / "calc.py").read_text() == "old\n"
        assert len(session.transcript) == before
        assert await session.handle_command("/undo")  # the first turn can be undone too
        assert await session.handle_command("/undo") == "Nothing to undo."
    finally:
        await session.shutdown()
