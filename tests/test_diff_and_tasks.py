import shutil

import pytest

from rlmagent.tasks import TaskList
from rlmagent.tools import build_tool_registry
from rlmagent.tools.patch_tool import make_patch_tool


def _text(outcome):
    return "".join(part.text for part in outcome.content) if isinstance(outcome.content, list) \
        else outcome.content


async def test_task_list_is_kept_in_its_file(tmp_path):
    path = tmp_path / "s.tasks.json"
    tasks = TaskList(path)
    seen = []
    tasks.on_change = lambda t: seen.append(len(t.items))
    assert await tasks.todo() == "No tasks."
    await tasks.todo("add", "read the  failing test")
    await tasks.todo("add", "fix it")
    assert await tasks.todo("done", 1) == "1. [x] read the failing test\n2. [ ] fix it"
    with pytest.raises(ValueError, match="no task 5"):
        await tasks.todo("done", 5)
    with pytest.raises(ValueError, match="action is one of"):
        await tasks.todo("rename", 1)
    assert TaskList(path).render() == tasks.render()
    await tasks.todo("clear")
    assert tasks.render() == "2. [ ] fix it".replace("2.", "1.") and seen == [1, 2, 2, 1]


async def test_apply_patch_tool_changes_files_all_or_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    tool = make_patch_tool()
    diff = ("--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n"
            "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+hello\n")
    assert "patched a.txt, new.txt" in _text(await tool.run("1", {"patch": diff}))
    assert (tmp_path / "a.txt").read_text() == "one\nTWO\n"
    assert (tmp_path / "new.txt").read_text() == "hello\n"
    bad = "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n one\n-nope\n+x\n"
    assert _text(await tool.run("2", {"patch": bad})).startswith("Patch not applied")
    assert (tmp_path / "a.txt").read_text() == "one\nTWO\n"


def test_patch_tool_only_where_files_can_change():
    assert "ApplyPatch" in [t.name for t in build_tool_registry()]
    from rlmagent.planning import MUTATING_TOOLS
    assert "ApplyPatch" in MUTATING_TOOLS


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_diff_and_tasks_commands(tmp_path):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider
    from rlmagent.undo import Shadow

    work = tmp_path / "w"
    work.mkdir()
    (work / "a.txt").write_text("one\n")
    session = await CodingSession.create(
        provider=ReplayProvider([text_turn("ok"), text_turn("ok")]), provider_name="replay",
        model="m", system="s", tools=[], cwd=str(work), sessions_dir=tmp_path / "s")
    session.shadow = Shadow(work, tmp_path / "h")
    assert await session._cmd_diff("") == "Nothing has changed yet."
    async for _ in session.submit("first"):
        pass
    (work / "a.txt").write_text("two\n")
    async for _ in session.submit("second"):
        pass
    (work / "b.txt").write_text("new\n")
    whole = await session._cmd_diff("")
    assert "-one" in whole and "+two" in whole and "b.txt" in whole
    last = await session._cmd_diff("last")
    assert "b.txt" in last and "a.txt" not in last
    assert "2 files changed" in await session._cmd_diff("stat")
    assert (await session._cmd_diff("everything")).startswith("Usage")
    assert session._cmd_tasks("") == "No task list in this session."
    session.tasks = TaskList()
    assert session._cmd_tasks("add write the test") == "1. [ ] write the test"
    assert session._cmd_tasks("done 1") == "1. [x] write the test"
    assert session._cmd_tasks("done 9").startswith("there is no task 9")
    await session.shutdown()


async def test_sidebar_lists_tasks_only_when_there_are_some():
    from textual.app import App
    from textual.widgets import Static

    from rlmagent.tui.sidebar import Sidebar

    class Host(App):
        def compose(self):
            yield Sidebar()

    async with Host().run_test() as pilot:
        bar = pilot.app.query_one(Sidebar)
        panel = bar.query_one("#task-panel", Static)
        bar.show_tasks([])
        assert not panel.display
        bar.show_tasks([{"text": "a", "done": True}, {"text": "b", "done": False}])
        await pilot.pause()
        assert panel.display and str(panel.render()).splitlines() == ["Tasks", "✓ a", "· b"]
