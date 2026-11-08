import asyncio

import pytest

from rlmagent.kernel.wasm import WasmKernel, available


@pytest.mark.skipif(not available(), reason="Sandbox is not installed")
async def test_runaway_worker_does_not_stall_or_reset_its_sibling(tmp_path):
    bad = WasmKernel(str(tmp_path), {}, timeout=1, restore=True)
    sibling = WasmKernel(str(tmp_path), {}, timeout=5)
    try:
        assert not (await bad.run("saved = 7")).error
        assert not (await sibling.run("marker = object()\nidentity = id(marker)\nnumber = 41")).error
        broker = sibling._runtime.process.pid
        running = asyncio.create_task(bad.run("while True: pass"))
        await asyncio.sleep(0.1)
        outcome = await asyncio.wait_for(sibling.run(
            "assert id(marker) == identity\nnumber += 1\nprint(number)"), 5)
        assert outcome.output.strip() == "42" and not outcome.error
        timed = await asyncio.wait_for(running, 5)
        assert timed.timed_out and timed.restarted
        assert sibling._runtime.process.pid == broker
        assert not (await sibling.run("assert id(marker) == identity\nassert number == 42")).error
        assert (await bad.run("print(saved)")).output.strip().endswith("7")
    finally:
        await bad.shutdown()
        await sibling.shutdown()
