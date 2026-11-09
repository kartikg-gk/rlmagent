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


@pytest.mark.parametrize("kind", ["local", "wasm", "node"])
async def test_default_loop_keeps_live_objects_modules_and_async_tasks_across_turns(tmp_path, kind):
    if kind in ("wasm", "node") and not available():
        pytest.skip("Sandbox is not installed")
    provider = ReplayProvider([
        text_turn("```python\nimport asyncio, math\nmarker = object()\nidentity = id(marker)\n"
                  "async def later():\n    await asyncio.sleep(0.05)\n    return 42\n"
                  "pending = asyncio.create_task(later())\nFINAL('ready')\n```"),
        text_turn("```python\nassert id(marker) == identity\nassert math.sqrt(49) == 7\n"
                  "assert pending.done()\nFINAL(pending.result())\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
        kernel_kind="wasm" if kind == "node" else kind,
        recursive_engine="node" if kind == "node" else "python",
        cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(provider, provider, tree)
    try:
        async def run(prompt):
            events = [event async for event in backend.stream_response(model="fixture", system="base",
                messages=[HumanEntry(content=prompt)], tools=[SimpleNamespace(name="python")])]
            assert events[-1].type == "done", events[-1]
            return events[-1].message.text

        assert await run("create live state") == "ready"
        runtime = backend.state._runtime
        await asyncio.sleep(0.15)
        assert await run("reuse live state") == "42"
        assert backend.state._runtime is runtime
        assert not runtime._closed
    finally:
        await backend.close()
    assert runtime._closed


@pytest.mark.parametrize("kind", ["local", "wasm", "node"])
async def test_real_compaction_keeps_live_kernel_and_background_task(tmp_path, kind):
    from rlmagent.conversation import CodingSession
    from rlmagent.harness.contracts.tooling import ToolSpec
    if kind in ("wasm", "node") and not available():
        pytest.skip("Sandbox is not installed")
    model = ReplayProvider([
        text_turn("```python\nimport asyncio, math\nmarker = object()\nidentity = id(marker)\n"
                  "async def later():\n    await asyncio.sleep(0.05)\n    return 42\n"
                  "pending = asyncio.create_task(later())\nFINAL('ready')\n```"),
        text_turn("```python\nFINAL('second')\n```"),
        text_turn("```python\nFINAL('third')\n```"),
        text_turn("Created a live marker, imported math, and started an async task."),
        text_turn("```python\nassert id(marker) == identity\nassert math.sqrt(49) == 7\n"
                  "assert pending.done()\nFINAL(pending.result())\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
        kernel_kind="wasm" if kind == "node" else kind,
        recursive_engine="node" if kind == "node" else "python",
        cell_timeout=30, allowance=Allowance(max_calls=12, max_cost=2))
    backend = BackendProvider(model, model, tree)
    tool = ToolSpec(name="python", label="Python", description="Execute Python",
                    parameters={"type": "object"}, run=None)
    session = await CodingSession.create(provider=backend, provider_name="fixture",
        model="fixture", system="base", tools=[tool])
    try:
        for prompt in ("create state", "continue", "continue again"):
            async for _ in session.submit(prompt):
                pass
        runtime = backend.state._runtime
        assert len(session.transcript) >= 6
        summary = await session.compact()
        assert "marker" in summary.summary
        await asyncio.sleep(0.1)
        async for _ in session.submit("check the original objects"):
            pass
        assert session.transcript[-1].text == "42"
        assert backend.state._runtime is runtime
    finally:
        await session.shutdown()
        await backend.close()


@pytest.mark.parametrize("kind", ["local", "wasm", "node"])
async def test_kept_child_reuses_its_live_interpreter_on_followup(tmp_path, kind):
    if kind in ("wasm", "node") and not available():
        pytest.skip("Sandbox is not installed")
    model = ReplayProvider([
        text_turn("```python\nchild = await spawn('remember', keep=True)\n"
                  "value = await child.result()\nFINAL(value)\n```"),
        text_turn("```python\nimport math\nmarker=object()\nidentity=id(marker)\nFINAL('ready')\n```"),
    ])
    replacement = ReplayProvider([
        text_turn("```python\nawait child.send('check live state')\n"
                  "value = await child.result()\nFINAL(value)\n```"),
        text_turn("```python\nassert id(marker)==identity\nFINAL(int(math.sqrt(1764)))\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
        kernel_kind="wasm" if kind == "node" else kind,
        recursive_engine="node" if kind == "node" else "python",
        cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(model, model, tree)
    try:
        async def run(prompt):
            events = [event async for event in backend.stream_response(model="fixture", system="base",
                messages=[HumanEntry(content=prompt)], tools=[SimpleNamespace(name="python")])]
            assert events[-1].type == "done", events[-1]
            return events[-1].message.text
        assert await run("start child") == "ready"
        child_state = (next(state for state in backend.state._runtime.sessions.values()
                            if state is not backend.state) if kind == "node"
                       else next(iter(backend.background.jobs.values())).state)
        runtime = child_state._runtime
        from rlmagent.agents.budget import BudgetedProvider
        tree.provider = BudgetedProvider(replacement, tree.allowance)
        assert await run("follow up") == "42"
        assert child_state._runtime is runtime
        assert len(model.calls) == len(replacement.calls) == 2
        assert tree.allowance.calls == 4
    finally:
        await backend.close()
    assert runtime._closed
