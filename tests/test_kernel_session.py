import pytest

from rlmagent_app.kernel.session import KernelSession, clip



@pytest.fixture
async def kernel(tmp_path):
    k = KernelSession(cwd=str(tmp_path), timeout=20)
    yield k
    await k.shutdown()


async def test_state_persists_between_cells(kernel):
    await kernel.run("x = 21")
    result = await kernel.run("print(x * 2)")
    assert result.output.strip() == "42"
    assert result.error is None


async def test_expression_value_is_shown(kernel):
    result = await kernel.run("1 + 1")
    assert "2" in result.output


async def test_error_is_a_plain_traceback(kernel):
    result = await kernel.run("1 / 0")
    assert "ZeroDivisionError" in result.error
    assert "\x1b[" not in result.error


async def test_timeout_interrupts_and_keeps_state(kernel):
    await kernel.run("kept = 7")
    result = await kernel.run("i = 0\nwhile True:\n    i += 1", timeout=2)
    assert result.timed_out
    after = await kernel.run("print(kept)")
    assert after.output.strip() == "7"


async def test_cell_that_ignores_the_interrupt_gets_a_fresh_kernel(tmp_path):
    # A blocking call such as a long sleep cannot always be interrupted
    # (on Windows it never is); the kernel is restarted instead of wedging.
    k = KernelSession(cwd=str(tmp_path), timeout=20, interrupt_grace=2)
    try:
        await k.run("kept = 7")
        result = await k.run("import time\ntime.sleep(120)", timeout=1)
        assert result.timed_out and result.restarted
        after = await k.run("print('alive')")
        assert after.output.strip() == "alive"
        assert "NameError" in (await k.run("kept")).error
    finally:
        await k.shutdown()


async def test_runs_in_cwd(kernel, tmp_path):
    result = await kernel.run("import os; print(os.getcwd())")
    assert result.output.strip().lower() == str(tmp_path).lower()


async def test_startup_code_runs_before_first_cell(tmp_path):
    k = KernelSession(cwd=str(tmp_path), startup_code="GREETING = 'hi'", timeout=20)
    try:
        assert (await k.run("print(GREETING)")).output.strip() == "hi"
    finally:
        await k.shutdown()


async def test_dead_kernel_restarts_once_and_says_so(kernel):
    await kernel.run("y = 1")
    await kernel.run("import os; os._exit(1)")
    result = await kernel.run("print('back')")
    assert result.restarted
    assert "back" in result.output


async def test_huge_output_is_clipped(kernel):
    result = await kernel.run("print('a' * 100_000)")
    assert len(result.output) < 25_000
    assert "characters omitted" in result.output


def test_clip_keeps_head_and_tail():
    text = "H" * 50 + "M" * 1000 + "T" * 50
    out = clip(text, 200)
    assert out.startswith("H" * 50) and out.endswith("T" * 50)
    assert "characters omitted" in out
    assert clip("short", 200) == "short"


async def test_setup_code_errors_are_reported(tmp_path):
    k = KernelSession(cwd=str(tmp_path), startup_code="raise ValueError('bad setup')", timeout=20)
    try:
        result = await k.run("print('after')")
        assert "bad setup" in (result.error or "")
    finally:
        await k.shutdown()


async def test_a_kernel_that_cannot_start_is_shut_down_and_reported(tmp_path, monkeypatch):
    import rlmagent_app.kernel.session as module

    shut = []

    class _Client:
        def start_channels(self):
            pass

        def stop_channels(self):
            pass

        async def wait_for_ready(self, timeout=None):
            raise RuntimeError("kernel never answered")

    class _Manager:
        def __init__(self, **kwargs):
            pass

        async def start_kernel(self, **kwargs):
            pass

        def client(self):
            return _Client()

        async def shutdown_kernel(self, now=False):
            shut.append(now)

    monkeypatch.setattr(module, "AsyncKernelManager", _Manager)
    k = KernelSession(cwd=str(tmp_path))
    result = await k.run("1")
    assert shut == [True]
    assert "could not start" in result.error and "kernel never answered" in result.error
    assert not k.started


async def test_kernel_start_writes_nothing_to_the_terminal(tmp_path, capfd):
    k = KernelSession(cwd=str(tmp_path), timeout=20)
    try:
        await k.run("1")
    finally:
        await k.shutdown()
    assert "without encryption" not in capfd.readouterr().err
