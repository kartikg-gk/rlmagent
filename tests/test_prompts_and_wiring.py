import argparse

import pytest

from rlmagent_app.agents.prompts import (
    LEAF_NOTE,
    RESUMED_NOTICE,
    child_first_message,
    child_system,
    context_preview,
    root_system,
)
from rlmagent_app.agents.tree import AgentNode


def test_root_prompt_explains_delegation():
    text = root_system("BASE", can_delegate=True)
    assert "BASE" in text and "gather_rlm" in text and "FINAL" not in text.split("BASE")[0]


def test_leaf_prompt_says_delegation_is_off():
    node = AgentNode(id="root.1", depth=2, parent_id="root", task="t")
    text = child_system("BASE", node, can_delegate=False)
    assert LEAF_NOTE in text and "FINAL" in text
    assert "gather_rlm(" not in text


def test_child_first_message_has_task_and_preview_not_whole_context():
    node = AgentNode(id="root.1", depth=1, parent_id="root", task="Count rows", context=list(range(10_000)))
    message = child_first_message(node)
    assert "Count rows" in message and "CONTEXT" in message
    assert len(message) < 2_000


def test_context_preview_describes_type_and_size():
    assert "dict" in context_preview({"a": 1})
    assert "None" in context_preview(None)
    assert "list of 3" in context_preview([1, 2, 3])


def test_resumed_notice_mentions_lost_variables():
    assert "variables" in RESUMED_NOTICE


def test_cli_accepts_budget_flags():
    from rlmagent_app.cli.main import _build_run_parser

    ns = _build_run_parser().parse_args(
        ["--max-depth", "1", "--max-cost", "0.5", "--max-calls", "9", "--cell-timeout", "30"]
    )
    assert (ns.max_depth, ns.max_cost, ns.max_calls, ns.cell_timeout) == (1, 0.5, 9, 30.0)


@pytest.mark.asyncio
async def test_session_shutdown_closes_the_tree(tmp_path, monkeypatch):
    from rlmness import Allowance

    from rlmagent_app.agents.tree import AgentTree
    from rlmagent_app.conversation import CodingSession
    from rlmagent_app.tools import build_tool_registry
    from rlmagent_model.scripted import ReplayProvider

    tree = AgentTree(
        provider=ReplayProvider([]), provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=Allowance(), sessions_dir=None,
        system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
    )
    await tree.start()
    kernel = tree.root_kernel()
    await kernel.run("x = 1")
    session = await CodingSession.create(
        provider=tree.provider, provider_name="replay", model="m", system="s",
        tools=build_tool_registry(kernel),
    )
    session.agent_tree = tree
    await session.shutdown()
    assert not kernel.started


@pytest.mark.asyncio
async def test_parent_session_records_its_sub_agent_sessions(tmp_path):
    import sys

    sys.path.insert(0, "tests")
    from _scripted import code_turn, text_turn
    from rlmness import Allowance

    from rlmagent_app.agents.tree import AgentTree
    from rlmagent_model.scripted import ReplayProvider

    tree = AgentTree(
        provider=ReplayProvider([code_turn("FINAL(1)"), text_turn("ok")]), provider_name="replay",
        model="m", cwd=str(tmp_path), allowance=Allowance(max_cost=10.0),
        sessions_dir=tmp_path / "sessions",
        system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
    )
    await tree.start()
    try:
        await tree.run_child(tree.root_id(), "one", None)
        assert tree.child_session_ids("root") == [tree.sessions[-1].session_id]
    finally:
        await tree.close()


@pytest.mark.asyncio
async def test_compaction_summary_says_the_kernel_is_still_running():
    from test_coding_session import _collect_events, _make_provider, _make_reply

    from rlmagent_app.agents.prompts import COMPACTION_NOTE
    from rlmagent_app.conversation import CodingSession
    from rlmagent_harness.provider.wire import StreamCloseEvent

    provider = _make_provider([_make_reply(f"Reply {i}") for i in range(6)])
    session = await CodingSession.create(
        provider=provider, provider_name="test", model="test-model", system="s",
    )
    for i in range(6):
        await _collect_events(session.submit(f"Message {i}"))
    provider._streams.append([StreamCloseEvent(reason="stop", message=_make_reply("summary"))])
    entry = await session.compact()
    assert entry.summary.rstrip().endswith(COMPACTION_NOTE)
    await session.shutdown()


@pytest.mark.asyncio
async def test_a_resumed_session_tells_the_model_its_kernel_is_new(tmp_path):
    from test_coding_session import _collect_events, _make_provider

    from rlmagent_app.conversation import CodingSession

    session = await CodingSession.create(
        provider=_make_provider(), provider_name="test", model="test-model", system="s",
        sessions_dir=tmp_path,
    )
    await _collect_events(session.submit("Hi"))
    sid = session.session_id
    await session.shutdown()

    provider = _make_provider()
    resumed = await CodingSession.resume(
        sid, provider=provider, provider_name="test", model="test-model", system="s",
        sessions_dir=tmp_path,
    )
    await _collect_events(resumed.submit("Next"))
    sent = " ".join(str(m) for m in provider.calls[0][2])
    assert RESUMED_NOTICE in sent
    await resumed.shutdown()


@pytest.mark.asyncio
async def test_agent_tree_follows_the_command_line_limits(tmp_path):
    from rlmagent_app.cli.main import _build_run_parser
    from rlmagent_app.runtime import make_agent_tree
    from rlmagent_model.scripted import ReplayProvider

    ns = _build_run_parser().parse_args(
        ["--max-depth", "1", "--max-calls", "9", "--max-cost", "0.5", "--max-live", "3",
         "--cell-timeout", "30"]
    )
    tree = make_agent_tree(ns, ReplayProvider([]), "replay", "m", tmp_path / "sessions", cwd=str(tmp_path))
    await tree.start()
    try:
        a = tree.allowance
        assert (a.max_depth, a.max_calls, a.max_cost, a.max_live) == (1, 9, 0.5, 3)
        assert tree.root_kernel().timeout == 30.0
        node = tree.nodes["root"]
        system = tree.system_for(node, [])
        assert "CONTEXT" in system  # child prompts carry the sub-agent role
    finally:
        await tree.close()


@pytest.mark.asyncio
async def test_root_kernel_from_make_agent_tree_reaches_the_bridge(tmp_path):
    import sys

    sys.path.insert(0, "tests")
    from _scripted import code_turn, text_turn

    from rlmagent_app.cli.main import _build_run_parser
    from rlmagent_app.runtime import make_agent_tree
    from rlmagent_model.scripted import ReplayProvider

    ns = _build_run_parser().parse_args(["--cell-timeout", "40"])
    provider = ReplayProvider([code_turn("FINAL(5)"), text_turn("ok")])
    tree = make_agent_tree(ns, provider, "replay", "m", None, cwd=str(tmp_path))
    await tree.start()
    try:
        out = await tree.root_kernel().run("print(await rlm('five'))")
        assert out.output.strip() == "5", out.error
        assert tree.kernels["root.1"].timeout == 40.0
    finally:
        await tree.close()


def test_child_first_message_says_only_final_returns_the_answer():
    node = AgentNode(id="root.1", depth=1, parent_id="root", task="Say what this is for.", context="x")
    message = child_first_message(node)
    assert "FINAL(" in message
    assert "not returned" in message


@pytest.mark.asyncio
async def test_switching_provider_or_model_keeps_the_budget_and_reaches_sub_agents(tmp_path):
    from rlmness import Allowance

    from rlmagent_app.agents.budget import BudgetedProvider
    from rlmagent_app.agents.tree import AgentTree
    from rlmagent_app.conversation import CodingSession
    from rlmagent_model.scripted import ReplayProvider

    allowance = Allowance(max_cost=10.0)
    tree = AgentTree(
        provider=ReplayProvider([]), provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=allowance, sessions_dir=None,
        system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
    )
    await tree.start()
    session = await CodingSession.create(
        provider=tree.provider, provider_name="replay", model="m", system="s",
    )
    session.agent_tree = tree
    try:
        await session.switch_provider(ReplayProvider([]), "other", model="m2")
        assert isinstance(session.harness.settings.provider, BudgetedProvider)
        assert session.harness.settings.provider.allowance is allowance
        assert tree.provider is session.harness.settings.provider
        assert (tree.provider_name, tree.model) == ("other", "m2")
        await session.switch_model("m3")
        assert tree.model == "m3"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_sub_agent_records_survive_resume(tmp_path):
    from test_coding_session import _collect_events, _make_provider

    from rlmagent_app.conversation import CodingSession

    session = await CodingSession.create(
        provider=_make_provider(), provider_name="test", model="test-model", system="s",
        sessions_dir=tmp_path,
    )
    await _collect_events(session.submit("Hi"))
    await session.note_sub_agent("child123")
    sid = session.session_id
    await session.shutdown()

    resumed = await CodingSession.resume(
        sid, provider=_make_provider(), provider_name="test", model="test-model", system="s",
        sessions_dir=tmp_path,
    )
    assert len(resumed.transcript) == 2
    assert await resumed.sub_agent_sessions() == ["child123"]
    await resumed.shutdown()
