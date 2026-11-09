import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from rlmagent.agents.approval import Approver
from rlmagent.agents.workspace import Workspace
from rlmagent.jobs import Jobs
from rlmagent.policy import Policy
from rlmagent.shell_sandbox import SandboxUnavailable, plan


@pytest.fixture
async def workspace(tmp_path):
    try:
        await asyncio.to_thread(plan, str(tmp_path), "true")
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    work = Workspace(tmp_path, Approver("auto"))
    work.policy = Policy(shell_access="workspace")
    return work


async def test_real_shell_cannot_read_parent_symlink_target_or_write_system_files(workspace):
    outside = workspace.root.parent / "outside-secret.txt"
    outside.write_text("synthetic-private-content")
    command = (
        "test ! -r ../outside-secret.txt && "
        "ln -s ../outside-secret.txt escape && test ! -r escape && "
        "test ! -e /mnt/c && test ! -e /mnt/e && "
        "test ! -e /proc/1/root/mnt && "
        "! (printf denied > /usr/forbidden-write) 2>/dev/null && "
        "printf allowed > result.txt && cat result.txt"
    )
    result = await workspace.run(command)
    assert result["exit"] == 0, result
    assert result["output"].strip() == "allowed"
    assert (workspace.root / "result.txt").read_text() == "allowed"
    assert outside.read_text() == "synthetic-private-content"


async def test_isolated_shell_has_no_host_environment_or_network_routes(workspace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-host-secret")
    code = ("import os,pathlib; assert 'OPENAI_API_KEY' not in os.environ; "
            "assert not pathlib.Path('/home').exists(); "
            "assert len(pathlib.Path('/proc/net/route').read_text().splitlines()) == 1; "
            "print('isolated')")
    command = "python3 -c " + shlex.quote(code)
    result = await workspace.run(command)
    assert result["exit"] == 0 and result["output"].strip() == "isolated", result


async def test_unavailable_isolation_never_falls_back_to_host(workspace, monkeypatch):
    from rlmagent import shell_sandbox
    def unavailable(*args, **kwargs):
        raise SandboxUnavailable("namespace runner unavailable")
    monkeypatch.setattr(shell_sandbox, "plan", unavailable)
    with pytest.raises(SandboxUnavailable):
        await workspace.run("printf bad > fallback-side-effect")
    assert not (workspace.root / "fallback-side-effect").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
async def test_windows_junction_cannot_reach_outside_the_workspace(workspace):
    target = workspace.root.parent / "junction-target"
    target.mkdir()
    (target / "private.txt").write_text("synthetic junction marker")
    junction = workspace.root / "junction"
    env = {**os.environ, "RLM_TEST_JUNCTION": str(junction), "RLM_TEST_TARGET": str(target)}
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
        "New-Item -ItemType Junction -Path $env:RLM_TEST_JUNCTION -Target $env:RLM_TEST_TARGET"],
        env=env, capture_output=True, check=True, timeout=15)
    try:
        result = await workspace.run("test ! -r junction/private.txt && printf protected")
        assert result["exit"] == 0 and result["output"] == "protected", result
    finally:
        junction.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="WSL interoperability")
@pytest.mark.parametrize("network", [False, True])
async def test_wsl_sandbox_cannot_launch_a_windows_program(workspace, network):
    source = Path(os.environ["SystemRoot"]) / "System32/cmd.exe"
    shutil.copyfile(source, workspace.root / "windows-probe.exe")
    workspace.policy.shell_network = network
    result = await workspace.run("./windows-probe.exe /c echo interop-sentinel", timeout=10)
    assert result["exit"] != 0 and "interop-sentinel" not in result["output"], result


async def test_cancellation_stops_linux_descendants_across_wsl_boundary(workspace):
    command = """python3 -c "import time,pathlib; time.sleep(3); pathlib.Path('late-mutation').write_text('bad')" & printf ready > ready; wait"""
    task = asyncio.create_task(workspace.run(command, timeout=30))
    try:
        for _ in range(300):
            if (workspace.root / "ready").exists():
                break
            if task.done():
                pytest.fail(str(task.result()))
            await asyncio.sleep(0.05)
        assert (workspace.root / "ready").exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(3.5)
        assert not (workspace.root / "late-mutation").exists()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_background_jobs_use_the_same_containment_and_stop_descendants(workspace):
    jobs = Jobs(str(workspace.root), workspace.root / "logs", shell_access="workspace")
    command = """python3 -c "import time,pathlib; time.sleep(3); pathlib.Path('late-job-mutation').write_text('bad')" & printf ready > job-ready; wait"""
    try:
        started = await jobs.job_start(command)
        for _ in range(300):
            if (workspace.root / "job-ready").exists():
                break
            await asyncio.sleep(0.05)
        assert (workspace.root / "job-ready").exists()
        stopped = await jobs.job_stop(started["id"])
        assert stopped["done"] and stopped["stopped"]
        await asyncio.sleep(3.5)
        assert not (workspace.root / "late-job-mutation").exists()
    finally:
        await jobs.close()


@pytest.mark.parametrize("engine", ["session", "rlmness"])
async def test_interactive_kernel_routes_use_the_default_shell_boundary(workspace, monkeypatch, engine):
    import importlib

    from _scripted import code_turn, text_turn

    from rlmagent.kernel.wasm import available
    from rlmagent.model.scripted import ReplayProvider
    from rlmagent.runtime import build_session

    if not available():
        pytest.skip("WebAssembly runtime is unavailable")
    cli = importlib.import_module("rlmagent.cli.main")
    command = "test ! -e /mnt/e && test ! -e /home && printf isolated > shell-result"
    code = f"result = await run({command!r})\nassert result['exit'] == 0, result\nFINAL('verified')"
    turns = [code_turn(code), text_turn("verified")] if engine == "session" else [
        text_turn(f"```python\n{code}\n```")]
    model = ReplayProvider(turns)
    monkeypatch.chdir(workspace.root)
    monkeypatch.setenv("RLM_AGENT_HOME", str(workspace.root / "agent-home"))
    monkeypatch.setattr(cli, "_resolve_provider", lambda _: model)
    ns = cli._build_run_parser().parse_args(["-p", "--no-session", "--no-approve", "--auto-approve",
        "--provider", "openai", "--model", "fixture", "--engine", engine, "--kernel", "wasm"])
    session = await build_session(ns)
    try:
        async for _ in session.submit("Use the shell in this project"):
            pass
        assert (workspace.root / "shell-result").read_text() == "isolated"
    finally:
        await session.shutdown()
