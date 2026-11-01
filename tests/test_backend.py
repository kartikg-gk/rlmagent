from types import SimpleNamespace

import pytest
from rlmness import Allowance

from _scripted import text_turn
from rlmagent.agents.approval import Approver
from rlmagent.backend import BackendProvider
from rlmagent.harness.contracts.transcript import HumanEntry
from rlmagent.harness.provider.wire import StreamCloseEvent
from rlmagent.kernel.wasm import available
from rlmagent.model.scripted import ReplayProvider

pytestmark = pytest.mark.skipif(not available(), reason="sandbox unavailable")


async def test_backend_owns_recursion_budget_and_persistent_state(tmp_path):
    model = ReplayProvider([
        text_turn("```python\nvalue = await rlm({'n': 21}, instruction='double n')\n```"),
        text_turn("```python\nFINAL(PROMPT['n'] * 2)\n```"),
        text_turn("```python\nFINAL(value)\n```"),
        text_turn("```python\nFINAL(value + 1)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    async def run(question):
        events = [e async for e in backend.stream_response(
            model='m', system='', messages=[HumanEntry(content=question)],
            tools=[SimpleNamespace(name='python')])]
        assert isinstance(events[-1], StreamCloseEvent), events
        return events[-1].message.text
    assert await run('double the input using a child') == '42'
    assert tree.allowance.calls == 3
    assert await run('add one to the previous value') == '43'
    assert tree.allowance.calls == 4
    assert sum(e['kind'] == 'run_started' for e in backend.events) == 3


async def test_backend_workspace_bridges_use_the_approver(tmp_path):
    model = ReplayProvider([
        text_turn("```python\nawait write_file('answer.txt', 'saved')\n"
                  "value = await read_file('answer.txt')\nFINAL(value)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('auto'), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=5, max_cost=2))
    backend = BackendProvider(model, model, tree)
    events = [e async for e in backend.stream_response(
        model='m', system='', messages=[HumanEntry(content='write and read a file')],
        tools=[SimpleNamespace(name='python')])]
    assert isinstance(events[-1], StreamCloseEvent), events
    assert events[-1].message.text == 'saved'
    assert (tmp_path / 'answer.txt').read_text() == 'saved'


async def _run_backend(backend, question="work"):
    events = [e async for e in backend.stream_response(
        model='m', system='', messages=[HumanEntry(content=question)],
        tools=[SimpleNamespace(name='python')])]
    assert isinstance(events[-1], StreamCloseEvent), events
    return events[-1].message.text, events


async def test_background_job_returns_result_and_mail(tmp_path):
    model = ReplayProvider([
        text_turn("```python\nh = await spawn('double', 21)\nvalue = await h.result()\nmail = await h.messages()\nFINAL({'value': value, 'mail': mail})\n```"),
        text_turn("```python\nawait tell_parent('finished')\nFINAL(PROMPT * 2)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    try:
        answer, events = await _run_backend(backend)
        assert "42" in answer and "finished" in answer
        assert any(e.type == 'text_delta' for e in events)
    finally:
        await backend.close()


async def test_backend_restores_saved_state_in_new_adapter(tmp_path):
    from rlmness.session import Session
    model = ReplayProvider([text_turn("```python\nnumber = 41\nFINAL(number)\n```"),
                            text_turn("```python\nFINAL(number + 1)\n```")])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    first = BackendProvider(model, model, tree)
    first.state = Session.load(tmp_path / 'state.json')
    assert (await _run_backend(first))[0] == '41'
    await first.close()
    second = BackendProvider(model, model, tree)
    second.state = Session.load(tmp_path / 'state.json')
    try:
        assert (await _run_backend(second))[0] == '42'
    finally:
        await second.close()


async def test_backend_playbook_helpers_and_denied_write(tmp_path):
    from rlmagent.playbook.improver import Improver
    improver = Improver(SimpleNamespace(), None, None)
    model = ReplayProvider([text_turn("```python\nawait improve('remember checks')\n"
        "message = await write_file('blocked', 'x')\nFINAL(message)\n```")])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           improver=improver, cell_timeout=30,
                           allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    try:
        answer, _ = await _run_backend(backend)
        assert answer.startswith('Error:') and 'approval' in answer
        assert not (tmp_path / 'blocked').exists()
        assert improver._pending[0][0] == 'remember checks'
    finally:
        await backend.close()


async def test_persistent_background_agent_accepts_followup(tmp_path):
    model = ReplayProvider([
        text_turn("```python\nh = await spawn('start', {}, keep=True)\nvalue = await h.result()\nawait h.send('increment')\nvalue = await h.result()\nFINAL(value)\n```"),
        text_turn("```python\ncount = 1\nFINAL(count)\n```"),
        text_turn("```python\ncount += 1\nFINAL(count)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    try:
        answer, _ = await _run_backend(backend)
        assert answer == '2'
        assert list(backend.background.jobs.values())[0].status == 'idle'
    finally:
        await backend.close()


async def test_backend_provider_switch_preserves_budget(tmp_path):
    from rlmagent.agents.budget import BudgetedProvider
    first = ReplayProvider([text_turn("```python\nFINAL(len(PROMPT))\n```")])
    second = ReplayProvider([text_turn("```python\nFINAL(len(PROMPT) + 1)\n```")])
    allowance = Allowance(max_calls=10, max_cost=2)
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           cell_timeout=30, allowance=allowance,
                           provider=BudgetedProvider(first, allowance))
    backend = BackendProvider(first, first, tree)
    try:
        assert (await _run_backend(backend))[0] == '1'
        tree.provider = BudgetedProvider(second, allowance)
        assert (await _run_backend(backend))[0] == '2'
        assert allowance.calls == 2
    finally:
        await backend.close()


async def test_backend_cancels_inflight_model_request(tmp_path):
    import asyncio
    import threading
    entered, stopped = asyncio.Event(), asyncio.Event()
    class WaitingProvider:
        async def stream_response(self, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
                yield None
            finally:
                stopped.set()
    signal = threading.Event()
    token = SimpleNamespace(is_cancelled=signal.is_set)
    model = WaitingProvider()
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    async def collect():
        return [e async for e in backend.stream_response(model='m', system='',
                messages=[HumanEntry(content='wait')], tools=[SimpleNamespace(name='python')],
                signal=token)]
    task = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(entered.wait(), 20)
        signal.set()
        events = await asyncio.wait_for(task, 15)
        assert events[-1].reason == 'aborted'
        await asyncio.wait_for(stopped.wait(), 5)
    finally:
        await backend.close()


async def test_contract_backend_records_actual_calls(tmp_path):
    import json
    from pathlib import Path
    from test_contract_run import _contract
    from rlmagent.contract_run import build_contract_session
    contract = _contract(tmp_path, tools=['python', 'Read'],
        kernel={'kind': 'wasm', 'approve': 'deny', 'network': False})
    provider = ReplayProvider([text_turn("```python\nFINAL(len(PROMPT))\n```")])
    session = await build_contract_session(contract, provider=provider, engine='rlmness')
    try:
        async for _ in session.submit('count keys'):
            pass
        assert session.agent_tree.allowance.calls == 1
    finally:
        await session.shutdown()
    records = [json.loads(line) for line in Path(contract.trace.file).read_text().splitlines()]
    assert records[-1]['kind'] == 'summary'
    assert records[-1]['calls'] == 1


async def test_contract_playbook_is_explicit_and_readonly(tmp_path):
    from test_contract_run import _contract
    from rlmagent.contract_run import build_contract_session
    from rlmagent.playbook.store import Playbook, Entry, save
    path = tmp_path / 'playbook.json'
    book = Playbook()
    book.entries['checks'] = Entry(id='checks', kind='rule', title='Checks',
        content='Use verify.py', scope='local')
    save(path, book)
    before = path.read_bytes()
    contract = _contract(tmp_path, tools=['python'],
        playbook={'file': 'playbook.json', 'update': False},
        kernel={'kind': 'wasm', 'approve': 'deny', 'network': False})
    model = ReplayProvider([text_turn("```python\nentry = await playbook('checks')\n"
        "try:\n    await improve('change')\nexcept Exception as exc:\n    message = str(exc)\nFINAL({'entry': entry['content'], 'error': message})\n```")])
    session = await build_contract_session(contract, provider=model, engine='rlmness')
    try:
        async for _ in session.submit('inspect playbook'):
            pass
        assert not session.improver._pending
        assert path.read_bytes() == before
        assert 'read-only' in session.transcript[-1].text
    finally:
        await session.shutdown()


async def test_backend_cancel_stops_workspace_command(tmp_path, monkeypatch):
    import asyncio
    import sys
    import threading
    entered = asyncio.Event()
    processes = []
    start = asyncio.create_subprocess_exec
    async def observe(*args, **kwargs):
        process = await start(*args, **kwargs)
        processes.append(process)
        entered.set()
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', observe)
    command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    model = ReplayProvider([text_turn(f"```python\nawait run({command!r})\n```")])
    signal = threading.Event()
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('auto'), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    async def collect():
        return [e async for e in backend.stream_response(model='m', system='',
                messages=[HumanEntry(content='run command')], tools=[SimpleNamespace(name='python')],
                signal=SimpleNamespace(is_cancelled=signal.is_set))]
    task = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(entered.wait(), 20)
        signal.set()
        events = await asyncio.wait_for(task, 15)
        assert events[-1].reason == 'aborted'
        await asyncio.wait_for(processes[0].wait(), 5)
    finally:
        await backend.close()


async def test_backend_timeout_keeps_checkpoint_for_next_turn(tmp_path):
    from rlmagent.harness.provider.wire import StreamFaultEvent
    model = ReplayProvider([
        text_turn("```python\nnumber = 42\nFINAL(number)\n```"),
        text_turn("```python\nwhile True: pass\n```"),
        text_turn("```python\nFINAL(number)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver('deny'), network=False,
                           cell_timeout=3, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    try:
        assert (await _run_backend(backend))[0] == '42'
        events = [e async for e in backend.stream_response(model='m', system='',
            messages=[HumanEntry(content='timeout')], tools=[SimpleNamespace(name='python')])]
        assert isinstance(events[-1], StreamFaultEvent)
        assert (await _run_backend(backend))[0] == '42'
    finally:
        await backend.close()

