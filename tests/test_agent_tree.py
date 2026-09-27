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
        assert value == "{1, 2}"
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
