import json

import pytest

from rlmagent_app.agents.bridge import BridgeServer, kernel_env
from rlmagent_app.agents.kernel_api import KERNEL_API
from rlmagent_app.kernel import KernelSession



async def _ask(port, payload):
    import asyncio

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    line = await reader.readline()
    writer.close()
    return json.loads(line)


@pytest.fixture
async def server():
    calls = []

    async def handler(agent, op, args):
        calls.append((agent, op, args))
        if op == "boom":
            raise RuntimeError("it broke")
        if op == "context":
            return {"rows": [1, 2, 3]}
        if op == "rlm":
            return f"done: {args['task']}"
        if op == "gather":
            return [f"done: {task}" for task, _ in args["jobs"]]
        return None

    s = BridgeServer(handler)
    await s.start()
    s.calls = calls
    yield s
    await s.close()


async def test_wrong_token_is_refused(server):
    reply = await _ask(server.port, {"token": "nope", "agent": "a", "op": "rlm", "args": {}})
    assert "error" in reply
    assert server.calls == []


async def test_handler_value_and_error(server):
    ok = await _ask(server.port, {"token": server.token, "agent": "a", "op": "rlm", "args": {"task": "t"}})
    assert ok == {"ok": "done: t"}
    bad = await _ask(server.port, {"token": server.token, "agent": "a", "op": "boom", "args": {}})
    assert bad == {"error": "it broke"}


async def test_kernel_api_round_trip(server, tmp_path):
    env = kernel_env(server.port, server.token, "child-1", can_delegate=True, is_child=True)
    kernel = KernelSession(cwd=str(tmp_path), env=env, startup_code=KERNEL_API, timeout=30)
    try:
        assert (await kernel.run("print(CONTEXT['rows'])")).output.strip() == "[1, 2, 3]"
        assert (await kernel.run("print(await rlm('x'))")).output.strip() == "done: x"
        out = await kernel.run("print(await gather_rlm([('a', None), ('b', 1)]))")
        assert out.output.strip() == "['done: a', 'done: b']"
        await kernel.run("FINAL({1, 2})")
        agent, op, args = server.calls[-1]
        assert (agent, op) == ("child-1", "final")
        assert args["value"] == "{1, 2}" and "repr" in args["note"]
    finally:
        await kernel.shutdown()


async def test_no_rlm_when_delegation_is_off(server, tmp_path):
    env = kernel_env(server.port, server.token, "leaf", can_delegate=False, is_child=True)
    kernel = KernelSession(cwd=str(tmp_path), env=env, startup_code=KERNEL_API, timeout=30)
    try:
        result = await kernel.run("rlm")
        assert "NameError" in result.error
    finally:
        await kernel.shutdown()


async def test_root_has_no_final_and_no_context(server, tmp_path):
    env = kernel_env(server.port, server.token, "root", can_delegate=True, is_child=False)
    kernel = KernelSession(cwd=str(tmp_path), env=env, startup_code=KERNEL_API, timeout=30)
    try:
        assert "NameError" in (await kernel.run("FINAL")).error
        assert "NameError" in (await kernel.run("CONTEXT")).error
    finally:
        await kernel.shutdown()
