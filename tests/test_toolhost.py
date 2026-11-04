import asyncio
import os
import sys
from pathlib import Path

import pytest

from rlmagent.agents.approval import Approver, Refused
from rlmagent.policy import Policy
from rlmagent.toolhost import RemoteTools, ToolHost

SRC = str(Path(__file__).resolve().parents[1] / "src")


async def _stdio(root, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return await RemoteTools.over_stdio(
        [sys.executable, "-m", "rlmagent.toolhost", "--root", str(root)])


async def test_tools_run_in_the_host_over_stdio(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    remote = await _stdio(tmp_path, monkeypatch)
    try:
        assert await remote.call("read_file", "a.py") == "x = 1\n"
        assert await remote.call("list_files", "*.py") == ["a.py"]
        hits = await remote.call("search_files", "x =")
        assert hits[0]["path"] == "a.py"
        with pytest.raises(ValueError, match="outside the workspace"):
            await remote.call("read_file", "../secret")
        with pytest.raises(ValueError, match="does not serve"):
            await remote.call("os_system", "x")
        done = await remote.call("run", "echo from-host")
        assert done["exit"] == 0 and "from-host" in done["output"]
    finally:
        await remote.close()


async def test_the_harness_approves_before_anything_is_sent(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    seen = []

    async def ask(request):
        seen.append((request.kind, request.target, request.preview))
        return "deny" if request.kind == "run" else "once"

    remote = await _stdio(tmp_path, monkeypatch)
    try:
        functions = remote.functions(Approver("ask", ask), Policy(allow_git=False))
        await functions["write_file"]("a.py", "x = 2\n")
        assert (tmp_path / "a.py").read_text() == "x = 2\n"
        assert seen[0][0] == "write" and "-x = 1" in seen[0][2] and "+x = 2" in seen[0][2]
        await functions["edit_file"](path="a.py", old="x = 2", new="x = 3")
        assert (tmp_path / "a.py").read_text() == "x = 3\n"
        with pytest.raises(Refused):
            await functions["run"]("echo no")
        with pytest.raises(ValueError, match="git push reaches beyond"):
            await functions["run"]("git push")
        assert [kind for kind, *_ in seen] == ["write", "edit", "run"]
    finally:
        await remote.close()


async def test_tcp_needs_the_token(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    host = ToolHost(str(tmp_path))
    server = await asyncio.start_server(lambda r, w: host.serve(r, w, "s3cret"), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(ConnectionError, match="bad token"):
            await RemoteTools.over_tcp("127.0.0.1", port, "wrong")
        remote = await RemoteTools.over_tcp("127.0.0.1", port, "s3cret")
        assert await remote.call("read_file", "a.txt") == "hello"
        started = await remote.call("job_start", f'"{sys.executable}" -c "print(7)"')
        waited = await remote.call("job_wait", started["id"], 30)
        assert waited["exit"] == 0
        assert (await remote.call("job_read", started["id"]))["text"].strip() == "7"
        await remote.close()
    finally:
        server.close()
        await host.close()


def test_the_flag_is_on_the_command_line():
    from rlmagent.cli.main import _build_run_parser

    ns = _build_run_parser().parse_args(["--tools-at", "tcp://127.0.0.1:7700"])
    assert ns.tools_at == "tcp://127.0.0.1:7700"
