"""The local kernel can get its variables back after it crashes."""

import pytest

from rlmagent.kernel.session import KernelSession


@pytest.mark.asyncio
async def test_local_kernel_restores_after_a_crash(tmp_path):
    kernel = KernelSession(str(tmp_path), startup_code="def helper(): return 9", restore=True,
                           timeout=60)
    try:
        r = await kernel.run("import math as m\nrows = [1, 2, 3]\ndef twice(x): return 2 * x\n"
                             "gen = (i for i in range(3))")
        assert r.error is None
        await kernel.kill()
        r = await kernel.run("print(rows, twice(4), m.pi > 3, helper())")
        assert r.restarted and r.error is None
        assert "[1, 2, 3] 8 True 9" in r.output
        assert "Kernel checkpoint restored" in r.output and "gen" in r.output.split("unavailable:")[1]
    finally:
        await kernel.shutdown()


@pytest.mark.asyncio
async def test_without_restore_a_crash_loses_variables(tmp_path):
    kernel = KernelSession(str(tmp_path), timeout=60)
    try:
        await kernel.run("rows = [1]")
        await kernel.kill()
        r = await kernel.run("print('rows' in dir())")
        assert "False" in r.output and "checkpoint" not in r.output
    finally:
        await kernel.shutdown()
