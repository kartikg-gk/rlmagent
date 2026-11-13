import copy
import json
from types import SimpleNamespace

import pytest
from _scripted import text_turn
from test_contract import VALID

from rlmagent.acp.server import Server
from rlmagent.contract import parse_contract
from rlmagent.contract_run import build_contract_session
from rlmagent.model.scripted import ReplayProvider


def document(tmp_path):
    value = copy.deepcopy(VALID)
    value["engine"] = "rlmness"
    value["model"]["name"] = "contract-model"
    value["tools"] = ["python"]
    value["workdir"] = str(tmp_path)
    value["trace"]["file"] = str(tmp_path / "trace.jsonl")
    value["kernel"] = {"kind": "wasm", "approve": "deny", "network": False}
    return value


def snapshot(session):
    return Server(lambda: None, lambda value: None)._snapshot(session)


async def test_recursive_runtime_inherits_fixed_model_and_delivers_steering_to_child(
    tmp_path, monkeypatch
):
    from rlmagent.kernel.wasm import available

    if not available():
        pytest.skip("Sandbox is not installed")
    model = ReplayProvider(
        [
            text_turn(
                "```python\nvalue = await rlm({'n': 7}, instruction='multiply n by six')\n```"
            ),
            text_turn("```python\nFINAL(PROMPT['n'] * 6)\n```"),
            text_turn("```python\nFINAL(value)\n```"),
        ]
    )
    settings = document(tmp_path)
    session = await build_contract_session(
        parse_contract(settings), provider=model, engine="rlmness"
    )
    session.steering.accept("Root instruction", "root-note")

    def started(event):
        if event["kind"] == "run_started" and event.get("depth", 0) > 0:
            session.steering.accept("Child instruction", "child-note", target=event["run_id"])

    session.backend.listeners.append(started)
    monkeypatch.setenv("RLM_AGENT_MODEL", "unrelated-model")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-credential")
    try:
        async for _ in session.submit("Compute the result using a child"):
            pass
        assert session.transcript[-1].text == "42"
        assert {call[0] for call in model.calls} == {"contract-model"}
        assert any("Child instruction" in call[2][-1].text for call in model.calls)
        assert {receipt["status"] for receipt in session.steering.receipts()} == {"consumed"}
        result = snapshot(session)
        calls = [record for record in session.agent_tree.tracer.records if record["kind"] == "call"]
        assert result["metrics"]["modelRequests"] == len(calls) == 3
        assert result["metrics"]["openModelRequests"] == 0
        assert result["metrics"]["agentsStarted"] == 1
        assert result["metrics"]["cells"] == sum(
            event["kind"] == "output_received" for event in session.backend.events
        )
        assert result["limits"] == {**settings["limits"], "request_timeout": 300.0, "strict_new_tokens": False,
                                    "max_new_tokens": None}
        assert result["runtime"]["kernel"]["network"] is False
        assert result["stop_reason"] == "done"
        assert "unrelated-credential" not in json.dumps(result)
    finally:
        await session.shutdown()


@pytest.mark.parametrize(
    "limit,value,expected",
    [
        ("max_calls", 0, "max_calls"),
        ("max_cost", 0, "max_cost"),
        ("max_seconds", 0, "max_seconds"),
    ],
)
async def test_exact_pre_admission_stop_reason_and_zero_provider_requests(
    tmp_path, limit, value, expected
):
    settings = document(tmp_path)
    settings["limits"][limit] = value
    settings["tools"] = []
    # The direct provider path exercises the same contract budget without starting a kernel.
    model = ReplayProvider([text_turn("should not run")])
    session = await build_contract_session(
        parse_contract(settings), provider=model, engine="session"
    )
    try:
        async for _ in session.submit("work"):
            pass
        result = snapshot(session)
        assert result["stop_reason"] == expected
        assert result["metrics"]["modelRequests"] == 0 and model.calls == []
    finally:
        await session.shutdown()


async def test_zero_agent_token_policy_is_not_treated_as_an_unset_limit(tmp_path):
    settings = document(tmp_path)
    settings["tools"] = []
    settings["policy"] = {
        "allow_git": False,
        "max_tool_output_bytes": 0,
        "max_compactions": 0,
        "agent_max_tokens": 0,
        "search_block": [],
    }
    model = ReplayProvider([text_turn("should not run")])
    session = await build_contract_session(
        parse_contract(settings), provider=model, engine="session"
    )
    try:
        async for _ in session.submit("work"):
            pass
        result = snapshot(session)
        assert result["stop_reason"] == "agent_tokens" and model.calls == []
        assert result["runtime"]["policy"]["max_tool_output_bytes"] == 0
        assert result["runtime"]["policy"]["max_compactions"] == 0
    finally:
        await session.shutdown()


async def test_post_request_budget_stop_retains_tree_usage(tmp_path):
    settings = document(tmp_path)
    settings["limits"]["max_calls"] = 1
    from rlmagent.harness.contracts.transcript import UsageStats

    turn = text_turn("```python\nintermediate = 42\n```")
    turn[-1].message.usage = UsageStats(input=5, output=7, total_tokens=12)
    model = ReplayProvider([turn])
    session = await build_contract_session(
        parse_contract(settings), provider=model, engine="rlmness"
    )
    try:
        async for _ in session.submit("Continue until done"):
            pass
        result = snapshot(session)
        assert result["stop_reason"] == "max_calls"
        assert result["metrics"]["modelRequests"] == 1
        assert result["metrics"]["openModelRequests"] == 0
        assert result["treeTotals"]["calls"] == 1
        assert result["usage"]["total_output_tokens"] > 0
    finally:
        await session.shutdown()


async def test_acp_close_cancels_request_and_returns_authoritative_snapshot(tmp_path, monkeypatch):
    import asyncio

    from test_acp_server import Pipe, _stop

    from rlmagent.acp.runtime import KEY
    from rlmagent.harness.contracts.transcript import ModelEntry
    from rlmagent.harness.provider.wire import StreamFaultEvent

    started = asyncio.Event()

    class Gated(ReplayProvider):
        async def stream_response(self, *, signal=None, **kwargs):
            started.set()
            while signal is None or not signal.is_cancelled():
                await asyncio.sleep(0.01)
            yield StreamFaultEvent(reason="aborted", error=ModelEntry(stop_reason="aborted"))

    monkeypatch.setattr("rlmagent.model.oai_compatible.OpenAIProvider", lambda profile: Gated([]))
    settings = document(tmp_path)
    settings["engine"], settings["tools"] = "session", []
    pipe = Pipe()
    server = Server(pipe.read_line, pipe.write)
    task = asyncio.create_task(server.serve())
    try:
        pipe.send(
            1,
            "session/new",
            {
                "cwd": str(tmp_path),
                "_meta": {
                    KEY: {
                        "contract": settings,
                        "credential": {"api_key": "fixture-private", "headers": {}},
                    }
                },
            },
        )
        sid = (await pipe.reply(1))["result"]["sessionId"]
        pipe.send(
            2, "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "work"}]}
        )
        await asyncio.wait_for(started.wait(), 5)
        pipe.send(3, "session/close", {"sessionId": sid})
        closed = (await pipe.reply(3))["result"]["_meta"]["rlmagent/session-v1"]
        prompt = (await pipe.reply(2))["result"]["_meta"]["rlmagent/session-v1"]
        for result in (closed, prompt):
            assert result["stop_reason"] == "cancelled"
            assert result["metrics"]["modelRequests"] == 1
            assert result["metrics"]["openModelRequests"] == 0
            assert result["runtime"]["model"]["name"] == "contract-model"
        assert "fixture-private" not in json.dumps(pipe.sent)
        assert sid not in server.sessions
    finally:
        await _stop(pipe, task)


async def test_global_gate_filters_captured_prompts_before_the_next_request(tmp_path):
    from rlmagent.agents.trace import TracedProvider, Tracer
    from rlmagent.conversation import CodingSession
    from rlmagent.refinement.controller import attach
    from rlmagent.refinement.store import Guidance

    model = ReplayProvider([text_turn("done")])
    session = await CodingSession.create(
        provider=model, provider_name="scripted", model="m", tools=[], system="Base policy"
    )
    controller = attach(session, tmp_path, persist=True)
    try:
        baseline = controller.store.read()
        updated = baseline.model_copy(deep=True)
        updated.revision += 1
        updated.entries["note"] = Guidance(
            id="note", kind="prompt", title="Note", content="Saved supplemental rule"
        )
        controller.store.commit(baseline, updated)
        controller.at_boundary()
        captured = session._system
        assert "Saved supplemental rule" in captured
        (tmp_path / "refinement.json").write_text('{"enabled": false}')
        tree = SimpleNamespace(provider=model, tracer=Tracer(), refinement=controller)
        events = [
            event
            async for event in TracedProvider(tree, "root").stream_response(
                model="m", system=captured, messages=[], tools=[]
            )
        ]
        assert events[-1].type == "done"
        assert model.calls[0][1] == "Base policy"
        assert controller.signal.is_cancelled()
    finally:
        await session.shutdown()
