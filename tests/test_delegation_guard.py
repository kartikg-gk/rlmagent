import pytest
from _scripted import text_turn
from rlmness import Allowance

from rlmagent.agents.tree import AgentNode, AgentTree
from rlmagent.model.scripted import ReplayProvider


def _tree(tmp_path, turns=()):
    return AgentTree(provider=ReplayProvider(list(turns)), provider_name="replay", model="m",
                     cwd=str(tmp_path), allowance=Allowance(max_cost=10.0), sessions_dir=None,
                     system_for=lambda n, t: "s", first_message_for=lambda n: n.task)


def test_only_a_large_barely_reduced_context_is_flagged(tmp_path):
    tree = _tree(tmp_path)
    parent = AgentNode(id="root", depth=0, parent_id=None, context="x" * 10_000)
    assert tree._barely_compressed(parent, "x" * 9_000) == (9_000, 10_000)
    assert tree._barely_compressed(parent, "x" * 1_000) is None
    assert tree._barely_compressed(AgentNode(id="r", depth=0, parent_id=None, context="x" * 100),
                                   "x" * 100) is None
    tree.guard = False
    assert tree._barely_compressed(parent, "x" * 9_000) is None


async def test_an_explicit_no_rejects_and_anything_else_passes(tmp_path):
    node = AgentNode(id="root.1", depth=1, parent_id="root", task="count words")
    tree = _tree(tmp_path, [text_turn("NO - slice it first\n```python\nCONTEXT[:100]\n```")])
    with pytest.raises(RuntimeError, match="(?s)Sub-agent not started.*slice it first"):
        await tree._confirm(node, 9_000, 10_000)
    tree = _tree(tmp_path, [text_turn("Nothing wrong here: YES")])
    await tree._confirm(node, 9_000, 10_000)


async def test_a_batch_is_judged_once_and_only_when_flagged(tmp_path):
    parent = AgentNode(id="root", depth=0, parent_id=None, context="x" * 10_000)
    tree = _tree(tmp_path, [])
    await tree._confirm_batch(parent, [("a", "x" * 100), ("b", "x" * 100)])
    tree = _tree(tmp_path, [text_turn("NO, too big")])
    with pytest.raises(RuntimeError, match="gather_rlm not started"):
        await tree._confirm_batch(parent, [("a", "x" * 9_000), ("b", "x" * 100)])
