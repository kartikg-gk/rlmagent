import json

from _scripted import code_turn, text_turn
from test_improve import PROPOSAL, SKILL, _setup, _submit
from test_playbook_auto import YES, _auto

from rlmagent_app.playbook.digest import render
from rlmagent_app.playbook.store import Entry


async def test_compaction_only_marks_a_check_that_runs_at_the_next_boundary(tmp_path):
    session, provider, improver = await _setup(tmp_path, [text_turn(YES), text_turn(PROPOSAL)])
    _auto(improver, every=1000)
    try:
        improver.compacted()
        assert len(provider.calls) == 0            # nothing runs at compaction time
        await improver.at_boundary()
        assert improver.local.history[-1]["trigger"] == "auto:compaction"
        await improver.at_boundary()
        assert len(provider.calls) == 2            # runs once, not again
    finally:
        await session.shutdown()


async def test_aborting_a_turn_drops_pending_updates(tmp_path):
    session, provider, improver = await _setup(tmp_path, [])
    _auto(improver, every=1000)
    try:
        improver.request("keep x")
        improver.compacted()
        session.abort()
        await improver.at_boundary()
        assert len(provider.calls) == 0
    finally:
        await session.shutdown()


async def test_each_change_is_recorded_in_the_session(tmp_path):
    session, provider, improver = await _setup(tmp_path, [text_turn(PROPOSAL), text_turn(PROPOSAL)])
    try:
        await session.handle_command("/improve keep it")
        records = await session.playbook_changes()
        assert records[-1]["summary"] == "keep the test command"
        assert records[-1]["changes"] == [{"action": "create", "id": "tests", "applied": True}]
        await session.handle_command("/improve --shared keep it")
        log = (tmp_path / "playbook-changes.jsonl").read_text().splitlines()
        assert json.loads(log[-1])["scope"] == "global"
    finally:
        await session.shutdown()


async def test_a_sub_agent_has_its_own_local_playbook_and_sees_the_shared_one(tmp_path):
    session, provider, improver = await _setup(tmp_path, [
        code_turn("fn = await skill('lines')\nawait improve('keep the test command')\nFINAL(fn('a\\nb'))"),
        text_turn("ok"),
        text_turn(PROPOSAL),                      # the sub-agent's own planner call
    ])
    try:
        improver.apply([SKILL], trigger="t", scope="global")
        value = await session.agent_tree.run_child("root", "count", None)
        texts = [m.text for m in session.agent_tree.sessions[-1].transcript if getattr(m, "role", "") == "toolResult"]
        assert value == 2, texts
        child = session.agent_tree.sessions[-1].improver
        assert [e.id for e in child.local.entries.values()] == ["tests"]
        assert improver.local.entries == {}
    finally:
        await session.shutdown()


def _e(i, text, updated=0.0):
    return Entry(id=f"e{i}", kind="memory", title=f"t{i}", content=text, scope="local", updated=updated)


def test_ranking_weights_the_goal_above_recent_messages_and_rare_words_above_common():
    entries = [_e(1, "deploy staging"), _e(2, "database migration")]
    text = render(entries, goal="database work", recent=["deploy it now"], per_kind=1)
    assert "e2" in text and "e1" not in text
    common = [_e(i, "python tests " + w) for i, w in enumerate(["alpha", "beta", "gamma"])]
    text = render(common, goal="", recent=["python gamma"], per_kind=1)
    assert "e2" in text
