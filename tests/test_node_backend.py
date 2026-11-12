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


@pytest.mark.parametrize("engine", ["node", "python"])
@pytest.mark.parametrize("has_next_request", [True, False])
async def test_active_child_steering_reaches_one_request_without_queuing_a_turn(tmp_path, engine, has_next_request):
    from rlmness.session import Session

    from rlmagent.agents.trace import Tracer
    from rlmagent.steering import Steering

    entered, release = asyncio.Event(), asyncio.Event()
    text = "targeted live instruction"
    observed = []
    status = "consumed" if has_next_request else "expired"
    root = ReplayProvider([text_turn("```python\n"
        "h = await spawn('begin work', keep=True, model='child-model')\n"
        "await wait_for_request()\n"
        "receipt = await steer_child(h, 'targeted live instruction', 'focus')\n"
        "assert receipt['status'] == 'accepted' and receipt['target'] == h.id\n"
        "await release_request()\n"
        "assert await h.result() == 2\n"
        f"assert await steering_status() == {status!r}\n"
        "again = await steer_child(h, 'targeted live instruction', 'focus')\n"
        f"assert again['status'] == {status!r}\n"
        "assert (await child_info(h))['steering'] == [again]\n"
        "try:\n"
        "    await steer_child(h, 'idle message')\n"
        "except ValueError as exc:\n"
        "    assert 'use send' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('steering started an idle turn')\n"
        "await h.send('follow-up')\n"
        "FINAL(await h.result())\n```")])
    replies = iter(["n = 1", "n += 1", "FINAL(n)", "FINAL(n + 1)"] if has_next_request
                   else ["n = 2\nFINAL(n)", "FINAL(n + 1)"])

    class Provider:
        async def stream_response(self, **kwargs):
            if kwargs["model"] != "child-model":
                assert all(text not in message.text for message in kwargs["messages"])
                async for event in root.stream_response(**kwargs):
                    yield event
                return
            observed.append(any(text in message.text for message in kwargs["messages"]))
            if len(observed) == 1:
                entered.set()
                await release.wait()
            for event in text_turn(f"```python\n{next(replies)}\n```"):
                yield event

    async def wait_for_request():
        await asyncio.wait_for(entered.wait(), 10)

    async def release_request():
        release.set()

    model = Provider()
    ledger = Steering(tmp_path / "steering.json")

    async def steering_status():
        return ledger.receipts()[0]["status"]

    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
        kernel_kind="wasm", recursive_engine=engine, cell_timeout=20,
        allowance=Allowance(max_calls=10, max_cost=2), tracer=Tracer(),
        session=SimpleNamespace(steering=ledger),
        host_functions={"wait_for_request": wait_for_request, "release_request": release_request,
                        "steering_status": steering_status})
    backend = BackendProvider(model, model, tree)
    backend.state = Session.load(tmp_path / "parent.json")
    try:
        events = [event async for event in backend.stream_response(model="root-model", system="base",
            messages=[HumanEntry(content="guide an active child")],
            tools=[SimpleNamespace(name="python")])]
        assert events[-1].type == "done", events[-1]
        assert events[-1].message.text == "3"
        assert observed == ([False, True, False, False] if has_next_request else [False, False])
        [receipt] = Steering(ledger.path).receipts()
        assert receipt["status"] == status
        if has_next_request:
            assert receipt["deliveries"][receipt["target"]]["requestId"]
        else:
            assert not receipt["deliveries"]
    finally:
        release.set()
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


async def test_node_waits_between_cells_before_next_provider_request(tmp_path, monkeypatch):
    from rlmagent.jobs import Jobs

    monkeypatch.setattr("rlmagent.subscriptions.POLL_SECONDS", 0.01)
    jobs = Jobs(str(tmp_path), tmp_path / "jobs")
    model = ReplayProvider([
        text_turn("```python\nwatcher = await watch_start(path='wake.txt')\n"
                  "await wait_for_activity(watcher['id'], timeout=10)\n```"),
        text_turn("```python\nFINAL('resumed')\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("auto"), network=False,
        kernel_kind="wasm", recursive_engine="node", cell_timeout=30,
        allowance=Allowance(max_calls=4, max_cost=2), session=SimpleNamespace(jobs=jobs),
        host_functions=jobs.functions())
    backend = BackendProvider(model, model, tree)

    async def run():
        return [event async for event in backend.stream_response(model="fixture", system="base",
            messages=[HumanEntry(content="wait for a workspace change")],
            tools=[SimpleNamespace(name="python")])]

    turn = asyncio.create_task(run())
    try:
        async with asyncio.timeout(20):
            while not any(row.get("tool") == "wait_for_activity" for row in backend.events):
                if turn.done():
                    pytest.fail(f"backend completed before requesting wait: {turn.result()}")
                await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)
        assert len(model.calls) == 1
        (tmp_path / "wake.txt").write_text("wake", encoding="utf-8")
        events = await asyncio.wait_for(turn, 15)
        assert events[-1].message.text == "resumed"
        assert len(model.calls) == 2
        watch, = await jobs.subscriptions.watch_list()
        assert watch["unread"] == 1
    finally:
        if not turn.done():
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
        await backend.close()
        await jobs.close()


async def test_node_reads_kept_child_receipt_after_reopen(tmp_path):
    from rlmness.session import Session
    from rlmagent.jobs import Jobs

    state = tmp_path / "backend.json"

    async def solve_steps(steps):
        model = ReplayProvider([text_turn("```python\n" + step + "\n```") for step in steps])
        jobs = Jobs(str(tmp_path), tmp_path / "jobs")
        tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("auto"), network=False,
            kernel_kind="wasm", recursive_engine="node", cell_timeout=30,
            allowance=Allowance(max_calls=6, max_cost=2), session=SimpleNamespace(jobs=jobs),
            host_functions=jobs.functions())
        backend = BackendProvider(model, model, tree)
        backend.state = Session.load(state)
        try:
            events = [event async for event in backend.stream_response(model="fixture", system="base",
                messages=[HumanEntry(content="get a child note")],
                tools=[SimpleNamespace(name="python")])]
            assert events[-1].type == "done", events[-1]
            return events[-1].message.text
        finally:
            await backend.close()
            await jobs.close()

    ids = await solve_steps([
        "h = await spawn('report', keep=True)\nawait name_child(h, 'reporter')\n"
        "await h.result(10)\nw = await watch_child(h)\n"
        "sent = await h.send('follow-up')\nawait h.result(10)\n"
        "assert (await child_delivery_receipt(h, sent))['status'] == 'done'\n"
        "rows = await child_messages(h)\nevents = await watch_events(w['id'])\n"
        "FINAL('|'.join([rows[0]['id'], w['id'], events[-1]['id']]))",
        "await tell_parent('ready')\nFINAL('done')",
        "assert not await watch_list()\nawait tell_parent('follow-up finished')\nFINAL('done')",
    ])
    identity, watch_id, event_id = ids.split('|')
    assert len(identity) == 32
    assert await solve_steps([
        "h = await find_child('reporter')\n"
        f"note = await read_child_message(h, {identity!r})\n"
        "assert note['status'] == 'read' and len(await child_messages(h)) == 1\n"
        f"event = await watch_read({watch_id!r}, {event_id!r})\n"
        "assert event['status'] == 'read' and event['change']['notes'] == 2\n"
        "FINAL(note['text'])",
    ]) == "ready"


async def test_parked_batch_child_releases_the_only_fanout_slot(tmp_path, monkeypatch):
    from rlmagent import backend as backend_module
    from rlmagent.jobs import Jobs

    config = backend_module.Config
    monkeypatch.setattr(backend_module, "Config", lambda **kwargs: config(max_concurrent=1, **kwargs))
    monkeypatch.setattr("rlmagent.subscriptions.POLL_SECONDS", 0.01)
    jobs = Jobs(str(tmp_path), tmp_path / "jobs")
    model = ReplayProvider([
        text_turn("```python\nresults = await gather_rlm(['waiter', 'writer'])\n"
                  "FINAL('|'.join(results))\n```"),
        text_turn("```python\nassert PROMPT == 'waiter'\n"
                  "w = await watch_start(path='signal.txt')\n"
                  "assert await wait_for_activity(w['id'], timeout=60)\n```"),
        text_turn("```python\nassert PROMPT == 'writer'\n"
                  "await write_file('signal.txt', 'ready')\nFINAL(PROMPT)\n```"),
        text_turn("```python\nassert PROMPT == 'waiter'\n"
                  "assert await read_file('signal.txt') == 'ready'\nFINAL(PROMPT)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("auto"), network=False,
        kernel_kind="wasm", recursive_engine="node", cell_timeout=30,
        allowance=Allowance(max_calls=8, max_live=1, max_cost=2),
        session=SimpleNamespace(jobs=jobs), host_functions=jobs.functions())
    backend = BackendProvider(model, model, tree)
    try:
        async with asyncio.timeout(30):
            events = [event async for event in backend.stream_response(model="fixture", system="base",
                messages=[HumanEntry(content="run a waiting child and a writer")],
                tools=[SimpleNamespace(name="python")])]
        assert events[-1].type == "done", "\n".join(str(row) for row in backend.events
            if row.get("error") or row["kind"] in ("child_status", "host_result", "run_failed"))
        assert events[-1].message.text == "waiter|writer"
        rows = backend.events
        parked = next(i for i, row in enumerate(rows) if row.get("status") == "waiting")
        wrote = next(i for i, row in enumerate(rows) if row.get("tool") == "write_file")
        resumed = next(i for i, row in enumerate(rows) if row.get("kind") == "child_status"
                       and row.get("status") == "running" and row["run_id"] == rows[parked]["run_id"])
        assert parked < wrote < resumed
        assert len(model.calls) == 4
    finally:
        await backend.close()
        await jobs.close()


async def test_failed_batch_cancels_parked_and_queued_children(tmp_path, monkeypatch):
    from rlmagent import backend as backend_module
    from rlmagent.harness.contracts.transcript import ModelEntry
    from rlmagent.harness.provider.wire import StreamFaultEvent
    from rlmagent.jobs import Jobs

    config = backend_module.Config
    monkeypatch.setattr(backend_module, "Config", lambda **kwargs: config(max_concurrent=1, **kwargs))
    jobs = Jobs(str(tmp_path), tmp_path / "jobs")
    model = ReplayProvider([
        text_turn("```python\ntry:\n    await gather_rlm(['waiter', 'failure', 'queued'])\n"
                  "except RuntimeError as error:\n    FINAL(str(error))\n```"),
        text_turn("```python\nawait wait_for_activity(timeout=60)\n```"),
        [StreamFaultEvent(reason="error", error=ModelEntry(
            stop_reason="error", error_message="fixture sibling failed"))],
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("auto"), network=False,
        kernel_kind="wasm", recursive_engine="node", cell_timeout=30,
        allowance=Allowance(max_calls=8, max_live=1, max_cost=2),
        session=SimpleNamespace(jobs=jobs), host_functions=jobs.functions())
    backend = BackendProvider(model, model, tree)
    try:
        async with asyncio.timeout(30):
            events = [event async for event in backend.stream_response(model="fixture", system="base",
                messages=[HumanEntry(content="verify that the batch can cancel")],
                tools=[SimpleNamespace(name="python")])]
        assert events[-1].type == "done", events[-1]
        assert "fixture sibling failed" in events[-1].message.text
        parked, = [row["run_id"] for row in backend.events if row.get("status") == "waiting"]
        assert any(row["kind"] == "run_failed" and row["run_id"] == parked
                   and "abandoned" in row["error"] for row in backend.events)
        assert sum(row["kind"] == "run_started" and row.get("depth") == 1
                   for row in backend.events) == 2
        assert len(model.calls) == 3
    finally:
        await backend.close()
        await jobs.close()
