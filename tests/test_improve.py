import json

from _scripted import code_turn, text_turn
from rlmness import Allowance

from rlmagent_app.agents.tree import AgentTree
from rlmagent_app.conversation import CodingSession
from rlmagent_app.playbook.improver import Improver
from rlmagent_app.tools import build_tool_registry
from rlmagent_model.scripted import ReplayProvider

PROPOSAL = json.dumps({
    "summary": "keep the test command", "rationale": "user said so", "expected": "fewer retries",
    "edits": [{"action": "create", "kind": "memory", "id": "tests", "title": "Test command",
               "content": "Run pytest -q from the project root."}],
})
SKILL = {"action": "create", "kind": "skill", "id": "lines", "title": "Count lines",
         "content": "def count_lines(text):\n    return len(text.splitlines())\n",
         "callable": "count_lines"}


async def _setup(tmp_path, streams):
    provider = ReplayProvider(streams)
    tree = AgentTree(
        provider=provider, provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=Allowance(max_cost=10.0), sessions_dir=None,
        system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
    )
    await tree.start()
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name="replay", model="m", system="s",
        tools=build_tool_registry(tree.root_kernel()),
    )
    await session.set_name("t")
    session.agent_tree = tree
    tree.attach("root", session)
    improver = Improver(session, local_path=None, shared_path=tmp_path / "shared.json")
    session.improver = tree.improver = improver
    return session, provider, improver


async def _submit(session, text):
    async for _ in session.submit(text):
        pass


async def test_the_agent_asks_and_the_playbook_changes_after_the_turn(tmp_path):
    session, provider, improver = await _setup(tmp_path, [
        code_turn("print(await improve('keep the test command'))"),
        text_turn("noted"),
        text_turn(PROPOSAL),          # planner, after the turn
        text_turn("second answer"),
    ])
    try:
        await _submit(session, "tests run with pytest -q, remember that")
        assert [e.id for e in improver.entries()] == ["tests"]
        await _submit(session, "next")
        sent = json.dumps([str(m) for m in provider.calls[-1][2]])
        assert "[Playbook]" in sent and "Run pytest -q" in sent
    finally:
        await session.shutdown()


async def test_improve_command_undo_and_list(tmp_path):
    session, provider, improver = await _setup(tmp_path, [text_turn(PROPOSAL)])
    try:
        reply = await session.handle_command("/improve keep the test command")
        assert "tests" in reply
        event_id = improver.local.history[-1]["id"]
        assert "tests" in await session.handle_command("/improve list")
        assert "undone" in await session.handle_command(f"/improve undo {event_id}")
        assert improver.entries() == []
    finally:
        await session.shutdown()


async def test_shared_edits_go_to_the_shared_file(tmp_path):
    session, provider, improver = await _setup(tmp_path, [text_turn(PROPOSAL)])
    try:
        await session.handle_command("/improve --shared keep the test command")
        doc = json.loads((tmp_path / "shared.json").read_text())
        assert [e["id"] for e in doc["entries"]] == ["tests"]
        assert improver.local.entries == {}
    finally:
        await session.shutdown()


async def test_a_bad_planner_answer_changes_nothing(tmp_path):
    session, provider, improver = await _setup(tmp_path, [text_turn("sorry, no json")])
    try:
        reply = await session.handle_command("/improve anything")
        assert "JSON" in reply and improver.entries() == []
    finally:
        await session.shutdown()


async def test_kernel_reads_entries_and_loads_skills(tmp_path):
    session, provider, improver = await _setup(tmp_path, [])
    try:
        improver.apply([SKILL], trigger="test")
        kernel = session.agent_tree.root_kernel()
        out = await kernel.run("fn = await skill('lines')\nprint(fn('a\\nb\\nc'))\n"
                               "print((await playbook('lines'))['title'])")
        assert out.output.split() == ["3", "Count", "lines"], out.error
    finally:
        await session.shutdown()


async def test_sub_agents_cannot_ask_for_improvement(tmp_path):
    session, provider, improver = await _setup(tmp_path, [code_turn("improve"), code_turn("FINAL(1)")])
    try:
        await session.agent_tree.run_child("root", "x", None)
        texts = [m.text for m in session.agent_tree.sessions[-1].transcript
                 if getattr(m, "role", "") == "toolResult"]
        assert any("NameError" in t for t in texts)
    finally:
        await session.shutdown()


def test_root_prompt_mentions_improve_only_when_the_playbook_is_on():
    from rlmagent_app.agents.prompts import root_system

    assert "improve(" in root_system("B", can_delegate=True, playbook=True)
    assert "improve(" not in root_system("B", can_delegate=True)


def test_the_app_attaches_an_improver():
    import inspect

    from rlmagent_app import runtime

    source = inspect.getsource(runtime.build_session)
    assert "Improver(" in source and "show_digest()" in source and "playbook=True" in source
