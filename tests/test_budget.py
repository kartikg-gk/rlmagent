import pytest
from rlmness import Allowance

from rlmagent_app.agents.budget import BudgetedProvider, to_spend
from rlmagent_app.agents.tree import AgentTree
from rlmagent_harness.contracts.transcript.diagnostics import CostBreakdown, UsageStats
from rlmagent_model.scripted import ReplayProvider
from _scripted import code_turn, text_turn



def test_to_spend_maps_tokens_and_cost():
    spend = to_spend(UsageStats(input=100, output=20, total_tokens=120, cost=CostBreakdown(total=0.5)))
    assert (spend.prompt_tokens, spend.completion_tokens, spend.cost) == (100, 20, 0.5)


def test_zero_cost_is_reported_as_unknown():
    assert to_spend(UsageStats(input=1, output=1)).cost is None


async def test_calls_beyond_the_limit_are_refused():
    allowance = Allowance(max_calls=1)
    provider = BudgetedProvider(ReplayProvider([text_turn("one"), text_turn("two")]), allowance)
    first = [e async for e in provider.stream_response(model="m", system="s", messages=[], tools=[])]
    assert first[-1].type == "done"
    second = [e async for e in provider.stream_response(model="m", system="s", messages=[], tools=[])]
    assert second[-1].type == "error"
    assert "calls" in provider.refusal


async def test_budget_is_shared_across_the_tree_and_stops_a_gather(tmp_path):
    # Root cell runs gather over 3 children; only 2 model calls are allowed in total.
    tree = AgentTree(
        provider=ReplayProvider([code_turn("FINAL(1)"), text_turn("ok")] * 3),
        provider_name="replay",
        model="replay-model",
        cwd=str(tmp_path),
        allowance=Allowance(max_calls=2, max_depth=2, max_live=1, max_cost=10.0),
        sessions_dir=None,
        system_for=lambda node, tools: "s",
        first_message_for=lambda node: node.task,
    )
    await tree.start()
    try:
        result = await tree.root_kernel().run("await gather_rlm(['a', 'b', 'c'])")
        assert result.error is not None
        assert "calls" in result.error
    finally:
        await tree.close()
