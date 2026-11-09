import asyncio
from types import SimpleNamespace

import pytest
from _scripted import text_turn
from rlmness import Allowance

from rlmagent.agents.approval import Approver
from rlmagent.backend import BackendProvider
from rlmagent.harness.contracts.transcript import HumanEntry
from rlmagent.kernel.wasm import available
from rlmagent.model.scripted import ReplayProvider

pytestmark = pytest.mark.skipif(not available(), reason="Sandbox is not installed")


async def test_node_backend_recurses_with_reviewed_workspace_services_and_child_attribution(tmp_path):
    model = ReplayProvider([
        text_turn("```python\nvalue = await rlm({'n': 7}, instruction='multiply by six')\n```"),
        text_turn("```python\nawait write_file('child.txt', 'saved')\nFINAL(PROMPT['n']*6)\n```"),
        text_turn("```python\nFINAL(value)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("auto"), network=False,
        kernel_kind="wasm", recursive_engine="node", cell_timeout=30,
        allowance=Allowance(max_calls=8, max_cost=2))
    backend = BackendProvider(model, model, tree)
    try:
        events = [event async for event in backend.stream_response(model="fixture", system="base",
            messages=[HumanEntry(content="compute through a child")],
            tools=[SimpleNamespace(name="python")])]
        assert events[-1].type == "done", events[-1]
        assert events[-1].message.text == "42"
        assert (tmp_path / "child.txt").read_text() == "saved"
        result = next(event for event in backend.events
                      if event["kind"] == "host_result" and event["tool"] == "write_file")
        assert result["run_id"] != "root"
        assert tree.allowance.calls == 3
    finally:
        await backend.close()


async def test_node_backend_denied_write_is_recoverable_and_has_no_side_effect(tmp_path):
    model = ReplayProvider([text_turn("```python\nresult = await write_file('blocked.txt', 'bad')\n"
        "assert result.startswith('Error:')\nFINAL('refused safely')\n```")])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
        kernel_kind="wasm", recursive_engine="node", cell_timeout=30,
        allowance=Allowance(max_calls=3, max_cost=2))
    backend = BackendProvider(model, model, tree)
    try:
        events = [event async for event in backend.stream_response(model="fixture", system="base",
            messages=[HumanEntry(content="try a refused write")],
            tools=[SimpleNamespace(name="python")])]
        assert events[-1].type == "done", events[-1]
        assert events[-1].message.text == "refused safely"
        assert not (tmp_path / "blocked.txt").exists()
    finally:
        await backend.close()


async def test_contract_allowlist_applies_to_node_children(tmp_path):
    import copy

    from test_contract import POLICY, VALID

    from rlmagent.contract import parse_contract
    from rlmagent.contract_run import build_contract_session

    doc = copy.deepcopy(VALID)
    doc.update(engine="rlmness", workdir=str(tmp_path), tools=["python", "Read"],
               kernel={"kind": "wasm", "approve": "auto", "network": False},
               policy={**POLICY, "allowed_tools": ["python", "read_file"]})
    doc["trace"]["file"] = str(tmp_path / "trace.jsonl")
    (tmp_path / "input.txt").write_text("allowed")
    model = ReplayProvider([
        text_turn("```python\n"
                  "try:\n"
                  "    child = await spawn('change provider', model='cli:claude-code')\n"
                  "    await child.result()\n"
                  "except Exception as exc:\n"
                  "    assert 'model specified by the runtime contract' in str(exc)\n"
                  "else:\n"
                  "    raise AssertionError('child changed the contract model')\n"
                  "value = await rlm('input.txt', instruction='read it')\nFINAL(value)\n```"),
        text_turn("```python\nassert 'write_file' not in (await capabilities())['functions']\n"
                  "assert 'write_file' not in globals()\nFINAL(await read_file('input.txt'))\n```"),
    ])
    session = await build_contract_session(parse_contract(doc), provider=model, engine="rlmness")
    try:
        assert [tool.name for tool in session.tools] == ["python"]
        assert "write_file" not in session.backend.function_names()
        events = [event async for event in session.submit("read through a child")]
        assert events[-1].messages[-1].text == "allowed"
        assert [requested_model for requested_model, *_ in model.calls] == [session.model, session.model]
    finally:
        await session.shutdown()


async def test_node_child_cancellation_stops_the_actual_provider_request(tmp_path):
    entered, stopped = asyncio.Event(), asyncio.Event()
    root = ReplayProvider([text_turn("```python\nh = await spawn('wait', model='waiting')\n"
        "await wait_for_child()\nawait h.cancel()\nFINAL('cancelled')\n```")])

    class Provider:
        async def stream_response(self, **kwargs):
            if kwargs["model"] == "waiting":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            else:
                async for event in root.stream_response(**kwargs):
                    yield event

    async def wait_for_child():
        await asyncio.wait_for(entered.wait(), 10)

    model = Provider()
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
        kernel_kind="wasm", cell_timeout=15, allowance=Allowance(max_calls=3, max_cost=2),
        host_functions={"wait_for_child": wait_for_child})
    backend = BackendProvider(model, model, tree)
    try:
        events = [event async for event in backend.stream_response(model="fixture", system="base",
            messages=[HumanEntry(content="cancel the waiting child")],
            tools=[SimpleNamespace(name="python")])]
        assert events[-1].type == "done", events[-1]
        assert events[-1].message.text == "cancelled"
        await asyncio.wait_for(stopped.wait(), 3)
        assert tree.allowance.calls == 2
    finally:
        await backend.close()


async def test_coordinator_loss_cancels_inference_and_reopens_checkpoint_without_losing_job(tmp_path):
    import sys

    from rlmness.node_engine import _shared_host
    from rlmness.session import Session

    from rlmagent.jobs import Jobs

    entered, cancelled = asyncio.Event(), asyncio.Event()
    phase = "seed"
    replay = ReplayProvider([text_turn("```python\nnumber = 41\nFINAL(number)\n```"),
                             text_turn("```python\nFINAL(number + 1)\n```")])

    class Provider:
        async def stream_response(self, **kwargs):
            if phase == "wait":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            else:
                async for event in replay.stream_response(**kwargs):
                    yield event

    provider = Provider()
    jobs = Jobs(str(tmp_path), tmp_path / "jobs")
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
                           kernel_kind="wasm", cell_timeout=20,
                           allowance=Allowance(max_calls=8, max_cost=2))
    backend = BackendProvider(provider, provider, tree)
    backend.state = Session.load(tmp_path / "state.json")

    async def turn():
        return [event async for event in backend.stream_response(
            model="fixture", system="base", messages=[HumanEntry(content=phase)],
            tools=[SimpleNamespace(name="python")],
        )]

    try:
        assert (await turn())[-1].message.text == "41"
        job = await jobs.job_start(argv=[sys.executable, "-u", "-c",
                                        "import time; print('started'); time.sleep(3); print('finished')"])
        phase = "wait"
        in_flight = asyncio.create_task(turn())
        await asyncio.wait_for(entered.wait(), 15)
        _shared_host().process.kill()
        assert (await asyncio.wait_for(in_flight, 15))[-1].type == "error"
        await asyncio.wait_for(cancelled.wait(), 5)
        phase = "recover"
        assert (await turn())[-1].message.text == "42"
        assert (await jobs.job_wait(job["id"], 15))["exit"] == 0
        assert (await jobs.job_read(job["id"]))["text"].splitlines() == ["started", "finished"]
    finally:
        await backend.close()
        await jobs.close()


async def test_node_shapes_multibyte_cell_output_and_recognised_errors(tmp_path):
    from rlmagent.policy import Policy

    model = ReplayProvider([
        text_turn("```python\nprint('界'*1000)\nimport missing_fixture_module\n```"),
        text_turn("```python\nFINAL(42)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
        kernel_kind="wasm", cell_timeout=15, allowance=Allowance(max_calls=3, max_cost=2),
        policy=Policy(max_tool_output_bytes=80))
    backend = BackendProvider(model, model, tree)
    try:
        events = [event async for event in backend.stream_response(model="fixture", system="base",
            messages=[HumanEntry(content="inspect output")],
            tools=[SimpleNamespace(name="python")])]
        assert events[-1].message.text == "42"
        observed = model.calls[1][2][-1].text
        assert "output truncated" in observed and observed.count("界") < 27
        assert "�" not in observed and "Hint: The module is not installed" in observed
    finally:
        await backend.close()
