import pytest
from rlmness import Allowance

from rlmagent_app.agents.tree import NEVER_FINAL, AgentTree
from rlmagent_app.tools import build_tool_registry
from rlmagent_model.scripted import ReplayProvider
from _scripted import code_turn, text_turn



def _tree(tmp_path, streams, **limits):
    return AgentTree(
        provider=ReplayProvider(streams),
        provider_name="replay",
        model="replay-model",
        cwd=str(tmp_path),
        allowance=Allowance(**{"max_depth": 2, "max_cost": 10.0, **limits}),
        sessions_dir=tmp_path / "sessions",
        system_for=lambda node, tools: f"agent depth {node.depth}",
        first_message_for=lambda node: node.task,
    )


async def test_child_final_value_reaches_the_parent_kernel(tmp_path):
    # Streams are consumed in call order: the child's two turns.
    tree = _tree(tmp_path, [
        code_turn("FINAL(sum(CONTEXT))"),
        text_turn("done"),
    ])
    await tree.start()
    try:
        root = tree.root_kernel()
        result = await root.run("answer = await rlm('add these', [1, 2, 3])\nprint(answer)")
        assert result.output.strip() == "6"
    finally:
        await tree.close()


async def test_gather_keeps_order(tmp_path):
    tree = _tree(tmp_path, [
        code_turn("FINAL(CONTEXT * 10)"), text_turn("ok"),
        code_turn("FINAL(CONTEXT * 10)"), text_turn("ok"),
    ], max_live=1)
    await tree.start()
    try:
        root = tree.root_kernel()
        out = await root.run("print(await gather_rlm([('a', 1), ('b', 2)]))")
        assert out.output.strip() == "[10, 20]"
    finally:
        await tree.close()


async def test_child_without_final_is_nudged_then_returns_its_text(tmp_path):
    tree = _tree(tmp_path, [text_turn("I think it is 4"), text_turn("still 4")])
    await tree.start()
    try:
        value = await tree.run_child(tree.root_id(), "guess", None)
        assert value.startswith("still 4") and NEVER_FINAL in value
    finally:
        await tree.close()


async def test_unserialisable_final_arrives_as_repr(tmp_path):
    tree = _tree(tmp_path, [code_turn("FINAL({1, 2})"), text_turn("ok")])
    await tree.start()
    try:
        value = await tree.run_child(tree.root_id(), "make a set", None)
        assert value.startswith("{1, 2}")
        assert "not JSON-serialisable" in value
    finally:
        await tree.close()


async def test_leaf_cannot_delegate(tmp_path):
    tree = _tree(tmp_path, [code_turn("rlm"), code_turn("FINAL('no')"), text_turn("ok")], max_depth=1)
    await tree.start()
    try:
        value = await tree.run_child(tree.root_id(), "try", None)
        assert value == "no"
        child_session = tree.sessions[-1]
        tool_texts = [m.text for m in child_session.transcript if getattr(m, "role", "") == "toolResult"]
        assert any("NameError" in t for t in tool_texts)
    finally:
        await tree.close()


async def test_close_shuts_every_kernel(tmp_path):
    tree = _tree(tmp_path, [])
    await tree.start()
    root = tree.root_kernel()
    await root.run("x = 1")
    await tree.close()
    assert not root.started


async def test_a_child_makes_no_model_calls_beyond_its_own_turns(tmp_path):
    tree = _tree(tmp_path, [code_turn("FINAL(1)"), text_turn("ok")])
    await tree.start()
    try:
        await tree.run_child(tree.root_id(), "one", None)
        assert len(tree.provider.calls) == 2  # no extra call to title the session
        assert tree.sessions[-1].title.startswith("sub-agent root.1")
    finally:
        await tree.close()


async def test_nested_delegation_does_not_wait_on_its_own_slot(tmp_path):
    # With one live slot, a child that delegates must still get its
    # grandchild run: waiting on children must not hold the slot.
    tree = AgentTree(
        provider=ReplayProvider([
            code_turn("x = await rlm('grandchild')\nFINAL(x + 1)"),  # child, turn 1
            code_turn("FINAL(1)"),                                   # grandchild, turn 1
            text_turn("ok"),                                         # grandchild, turn 2
            text_turn("ok"),                                         # child, turn 2
        ]),
        provider_name="replay", model="replay-model", cwd=str(tmp_path),
        allowance=Allowance(max_depth=2, max_live=1, max_cost=10.0),
        sessions_dir=None,
        system_for=lambda node, tools: "s", first_message_for=lambda node: node.task,
        cell_timeout=20,
    )
    await tree.start()
    try:
        assert await tree.run_child(tree.root_id(), "child", None) == 2
    finally:
        await tree.close()


async def test_a_finished_sub_agent_can_no_longer_call_the_host(tmp_path):
    tree = _tree(tmp_path, [code_turn("FINAL(1)"), text_turn("ok")])
    await tree.start()
    try:
        await tree.run_child(tree.root_id(), "one", None)
        token = tree.kernels["root.1"].env["RLM_AGENT_BRIDGE_TOKEN"]
        assert tree._bridge._caller(token) is None
    finally:
        await tree.close()


class _SlowProvider:
    """Replays scripted turns, each after a delay, like a slow model."""

    def __init__(self, streams, delay):
        self.inner = ReplayProvider(streams)
        self.delay = delay

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def stream_response(self, **kwargs):
        import asyncio

        await asyncio.sleep(self.delay)
        async for event in self.inner.stream_response(**kwargs):
            yield event


async def test_waiting_on_a_sub_agent_does_not_count_against_the_cell_timeout(tmp_path):
    tree = AgentTree(
        provider=_SlowProvider([code_turn("FINAL(7)"), text_turn("ok")], delay=2.5),
        provider_name="replay", model="replay-model", cwd=str(tmp_path),
        allowance=Allowance(max_depth=2, max_cost=10.0), sessions_dir=None,
        system_for=lambda node, tools: "s", first_message_for=lambda node: node.task,
        cell_timeout=3,
    )
    await tree.start()
    try:
        root = tree.root_kernel()
        await root.run("kept = 'still here'")
        result = await root.run("x = await rlm('slow child')\nprint(x, kept)")
        assert not result.timed_out and not result.restarted
        assert result.output.strip() == "7 still here"
    finally:
        await tree.close()


async def test_each_session_file_lists_the_sub_agents_it_started(tmp_path):
    from rlmagent_app.conversation import CodingSession
    from rlmagent_app.tools import build_tool_registry

    tree = _tree(tmp_path, [
        code_turn("x = await rlm('grandchild')\nFINAL(x)"),   # child
        code_turn("FINAL(1)"), text_turn("ok"),               # grandchild
        text_turn("ok"),                                      # child
    ])
    await tree.start()
    root = await CodingSession.create(
        provider=tree.provider, provider_name="replay", model="m", system="s",
        tools=build_tool_registry(tree.root_kernel()), sessions_dir=tmp_path / "sessions",
    )
    await root.set_name("the root")
    tree.attach(tree.root_id(), root)
    try:
        await tree.run_child(tree.root_id(), "child", None)
        child, grandchild = tree.sessions
        assert await root.sub_agent_sessions() == [child.session_id]
        assert await child.sub_agent_sessions() == [grandchild.session_id]
        assert root.title == "the root"
    finally:
        await root.shutdown()
        await tree.close()
