from threading import Event

import pytest

from rlmagent.kept_agents import KeptAgents


def test_busy_kept_child_queues_in_order_and_recovers_durable_receipts(tmp_path):
    entered, release = Event(), Event()
    questions = []
    agents = KeptAgents(4)
    state = tmp_path / "session.json"
    agents.open(state)

    def child(_data, question, _token, **_):
        questions.append(question)
        if question == "start":
            entered.set()
            assert release.wait(5)
        return question

    try:
        bindings = agents.bindings("root", child)
        key = bindings["_spawn"]("start", keep=True)
        assert entered.wait(5)
        first = bindings["_send"](key, "first")
        second = bindings["_send"](key, "second")
        assert bindings["_delivery"](key, first) == "queued"
        release.set()
        assert bindings["_result"](key, 5) == "second"
        assert questions == ["start", "first", "second"]
        assert bindings["_delivery"](key, second) == "done"
        agents.saved[key]["deliveries"].update(
            crashed={"status": "running"}, waiting={"status": "queued", "text": "after restart"},
        )
        agents._write(key)
    finally:
        release.set()
        agents.close()

    reopened = KeptAgents(4)
    try:
        reopened.open(state)
        bindings = reopened.bindings("root", child)
        assert bindings["_result"](key, 5) == "after restart"
        assert bindings["_delivery"](key, "crashed") == "interrupted"
        assert bindings["_delivery"](key, "waiting") == "done"
        assert bindings["_delivery"](key, first) == "done"
        assert questions == ["start", "first", "second", "after restart"]
    finally:
        reopened.close()


def test_node_kept_child_queues_busy_messages_and_restores_pending_delivery(tmp_path):
    from rlmness import Config, Spend, solve, wasm_available

    if not wasm_available():
        pytest.skip("Node and Pyodide are unavailable")
    entered, release = Event(), Event()
    children = iter((
        "await block(); FINAL('initial')",
        "FINAL('first')",
        "FINAL('second')",
        "FINAL('recovered')",
    ))
    root_steps = iter((
        "h = await spawn('start', keep=True, model='child')\n"
        "await wait_started()\n"
        "one = await h.send('first')\n"
        "two = await h.send('second')\n"
        "assert await h.delivery(one) == 'queued'\n"
        "await unblock()\n"
        "value = await h.result(10)\n"
        "FINAL([h.id, one, two, value])",
        "h = (await children())[0]\n"
        "value = await h.result(10)\n"
        "FINAL([value, await h.delivery('running-before-crash'), "
        "await h.delivery('pending-before-crash')])",
    ))

    class Provider:
        def complete(self, messages, *, model):
            code = next(children if model == "child" else root_steps)
            return f"```python\n{code}\n```", Spend(cost=0)

    def block():
        entered.set()
        assert release.wait(10)

    state = tmp_path / "session.json"
    cfg = Config(primary_agent="fixture", engine="node", runtime="wasm", timeout=20)
    agents = KeptAgents(4)
    agents.open(state)
    try:
        result = solve("root", Provider(), config=cfg, run_id="root", background=agents,
                       host_functions={"block": block, "wait_started": lambda: entered.wait(10),
                                       "unblock": release.set})
        key, first, second, value = result.output
        assert value == "second"
        assert agents.saved[key]["deliveries"][first]["status"] == "done"
        assert agents.saved[key]["deliveries"][second]["status"] == "done"
        agents.saved[key]["deliveries"].update({
            "running-before-crash": {"status": "running"},
            "pending-before-crash": {"status": "queued", "text": "recover queued work"},
        })
        agents._write(key)
    finally:
        release.set()
        agents.close()

    reopened = KeptAgents(4)
    try:
        reopened.open(state)
        result = solve("root", Provider(), config=cfg, run_id="root", background=reopened)
        assert result.output == ["recovered", "interrupted", "done"]
    finally:
        reopened.close()
