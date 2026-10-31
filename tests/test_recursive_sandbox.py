import asyncio

import pytest
from rlmness import Allowance

from _scripted import code_turn, text_turn
from rlmagent.agents.tree import AgentTree
from rlmagent.kernel.wasm import WasmKernel, available
from rlmagent.model.scripted import ReplayProvider


@pytest.mark.skipif(not available(), reason="WebAssembly sandbox not set up")
async def test_recursive_agents_get_separate_interpreters_in_one_host(tmp_path, monkeypatch):
    started = []
    original_start = WasmKernel._start

    async def observe_start(kernel):
        failure = await original_start(kernel)
        if failure is None:
            channel = kernel._runtime.channel
            started.append((channel.sid, channel.process.pid))
        return failure

    monkeypatch.setattr(WasmKernel, "_start", observe_start)
    tree = AgentTree(
        provider=ReplayProvider([
            code_turn("assert 'marker' not in globals()\n"
                      "marker = 'child'\n"
                      "value = await rlm('grandchild', {'level': 2})\n"
                      "FINAL({'marker': marker, 'nested': value, 'context': CONTEXT})"),
            code_turn("assert 'marker' not in globals()\n"
                      "marker = 'grandchild'\n"
                      "FINAL({'marker': marker, 'context': CONTEXT})"),
            text_turn("done"),
            text_turn("done"),
        ]),
        provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=Allowance(max_depth=2, max_live=1, max_calls=10, max_cost=1.0),
        sessions_dir=None, system_for=lambda node, tools: "s",
        first_message_for=lambda node: node.task, kernel="wasm", cell_timeout=30,
    )
    await tree.start()
    try:
        root = tree.root_kernel()
        outcome = await asyncio.wait_for(root.run(
            "marker = 'root'\n"
            "answer = await rlm('child', {'level': 1})\n"
            "assert marker == 'root'\n"
            "assert answer == {'marker': 'child', 'context': {'level': 1}, "
            "'nested': {'marker': 'grandchild', 'context': {'level': 2}}}\n"
            "print('isolated recursion complete')"
        ), timeout=90)
        assert outcome.error is None, outcome.error
        assert "isolated recursion complete" in outcome.output
        assert len(started) == 3
        assert len({sid for sid, pid in started}) == 3
        assert len({pid for sid, pid in started}) == 1
    finally:
        await tree.close()
