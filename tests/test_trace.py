import json

from _scripted import code_turn
from rlmness import Allowance

from rlmagent_app.agents.trace import Tracer
from rlmagent_app.agents.tree import AgentTree
from rlmagent_app.conversation import CodingSession
from rlmagent_app.tools import build_tool_registry
from rlmagent_harness.contracts.transcript import ModelEntry, TextSegment
from rlmagent_harness.contracts.transcript.diagnostics import CostBreakdown, UsageStats
from rlmagent_harness.provider.wire import StreamCloseEvent
from rlmagent_model.scripted import ReplayProvider


def paid_text(text, cost=0.01):
    usage = UsageStats(input=10, output=5, total_tokens=15, cost=CostBreakdown(total=cost))
    return [StreamCloseEvent(reason="stop", message=ModelEntry(
        content=[TextSegment(text=text)], stop_reason="stop", usage=usage))]


async def _run_root(tmp_path, streams, prompt, **limits):
    tracer = Tracer(tmp_path / "trace.jsonl", contract={"format": "test"})
    tree = AgentTree(
        provider=ReplayProvider(streams), provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=Allowance(**{"max_depth": 2, "max_cost": 10.0, "max_live": 1, **limits}),
        sessions_dir=None, system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
        tracer=tracer,
    )
    await tree.start()
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name="replay", model="m", system="s",
        tools=build_tool_registry(tree.root_kernel()),
    )
    await session.set_name("trace test")  # no extra model call for a title
    session.agent_tree = tree
    tree.attach("root", session)
    async for _ in session.submit(prompt):
        pass
    await session.shutdown()
    return tracer


def _records(tracer, kind):
    return [r for r in tracer.records if r["kind"] == kind]


def _calls_of(tracer, agent):
    return [r["id"] for r in _records(tracer, "call") if r["agent"] == agent]


def _links(tracer, how):
    return {(r["from"], r["to"]) for r in _records(tracer, "link") if r["how"] == how}


async def test_gather_links_spawn_return_and_next(tmp_path):
    tracer = await _run_root(tmp_path, [
        code_turn("print(await gather_rlm(['a', 'b']))"),   # root call 1
        code_turn("FINAL(1)"), paid_text("ok"),             # root.1
        code_turn("FINAL(2)"), paid_text("ok"),             # root.2
        paid_text("done"),                                  # root call 2
    ], "go")
    root, c1, c2 = (_calls_of(tracer, a) for a in ("root", "root.1", "root.2"))
    assert len(root) == 2 and len(c1) == 2 and len(c2) == 2
    assert _links(tracer, "spawn") == {(root[0], c1[0]), (root[0], c2[0])}
    assert _links(tracer, "return") == {(c1[-1], root[1]), (c2[-1], root[1])}
    assert {(root[0], root[1]), (c1[0], c1[1]), (c2[0], c2[1])} <= _links(tracer, "next")


async def test_an_uncollected_background_sub_agent_has_no_return_link(tmp_path):
    tracer = await _run_root(tmp_path, [
        code_turn("h = await spawn('bg')\nimport asyncio\nawait asyncio.sleep(3)"),
        code_turn("FINAL(1)"), paid_text("ok"),
        paid_text("done"),
    ], "go")
    assert _links(tracer, "spawn") and not _links(tracer, "return")


async def test_summary_totals_match_the_calls_and_name_the_limit(tmp_path):
    tracer = await _run_root(tmp_path, [code_turn("x = 1"), paid_text("b")], "go", max_calls=1)
    summary = _records(tracer, "summary")[-1]
    calls = _records(tracer, "call")
    assert summary["calls"] == len(calls) == 1
    assert abs(summary["cost"] - sum(c["cost"] for c in calls)) < 1e-9
    assert summary["agents"]["root"]["calls"] == 1
    assert summary["stopped_by"] and "calls" in summary["stopped_by"]
    assert summary["complete"] is True


async def test_agent_records_and_file_order(tmp_path):
    tracer = await _run_root(tmp_path, [
        code_turn("print(await rlm('x'))"), code_turn("FINAL(1)"), paid_text("ok"), paid_text("d"),
    ], "go")
    agents = {r["agent"]: r for r in _records(tracer, "agent")}
    assert agents["root.1"]["parent"] == "root" and agents["root.1"]["status"] == "done"
    lines = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert lines[0]["kind"] == "run" and lines[0]["contract"] == {"format": "test"}
    assert lines[-1]["kind"] == "summary"


def test_a_call_after_compaction_links_as_compact():
    tracer = Tracer(None)
    first = tracer.begin("root", "m")
    tracer.end(first, None, "stop")
    tracer.compacted("root")
    second = tracer.begin("root", "m")
    tracer.end(second, None, "stop")
    assert _links(tracer, "compact") == {(first, second)}
    assert not _links(tracer, "next")


async def test_compaction_tells_the_tracer():
    from test_coding_session import _collect_events, _make_provider, _make_reply

    provider = _make_provider([_make_reply(f"Reply {i}") for i in range(6)])
    session = await CodingSession.create(provider=provider, provider_name="t", model="m", system="s")
    for i in range(6):
        await _collect_events(session.submit(f"Message {i}"))
    provider._streams.append([StreamCloseEvent(reason="stop", message=_make_reply("summary"))])
    seen = []
    session.on_compacted = lambda: seen.append(True)
    await session.compact()
    assert seen == [True]


async def test_switching_provider_keeps_the_root_traced(tmp_path):
    from rlmagent_app.agents.trace import TracedProvider

    tree = AgentTree(
        provider=ReplayProvider([]), provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=Allowance(max_cost=10.0), sessions_dir=None,
        system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
    )
    await tree.start()
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name="replay", model="m", system="s",
    )
    session.agent_tree = tree
    try:
        await session.switch_provider(ReplayProvider([paid_text("hi")]), "other")
        assert isinstance(session.harness.settings.provider, TracedProvider)
        await session.set_name("t")
        async for _ in session.submit("x"):
            pass
        assert _calls_of(tree.tracer, "root")
    finally:
        await session.shutdown()


def test_the_app_builds_the_root_session_on_a_traced_provider():
    import inspect

    from rlmagent_app import runtime

    source = inspect.getsource(runtime.build_session)
    assert 'tree.provider_for("root")' in source and "on_compacted" in source
