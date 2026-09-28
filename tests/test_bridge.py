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
    token = server.issue("a")
    ok = await _ask(server.port, {"token": token, "op": "rlm", "args": {"task": "t"}})
    assert ok == {"ok": "done: t"}
    bad = await _ask(server.port, {"token": token, "op": "boom", "args": {}})
    assert bad["error"] == "it broke"


async def test_the_caller_is_whoever_its_token_was_issued_to(server):
    token = server.issue("leaf")
    await _ask(server.port, {"token": token, "agent": "root", "op": "rlm", "args": {"task": "t"}})
    assert server.calls[-1][0] == "leaf"


async def test_a_revoked_token_is_refused(server):
    token = server.issue("gone")
    server.revoke(token)
    reply = await _ask(server.port, {"token": token, "op": "rlm", "args": {}})
    assert "error" in reply and server.calls == []


async def test_kernel_api_round_trip(server, tmp_path):
    env = kernel_env(server.port, server.issue("child-1"), "child-1", can_delegate=True, is_child=True)
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
    env = kernel_env(server.port, server.issue("leaf"), "leaf", can_delegate=False, is_child=True)
    kernel = KernelSession(cwd=str(tmp_path), env=env, startup_code=KERNEL_API, timeout=30)
    try:
        result = await kernel.run("rlm")
        assert "NameError" in result.error
    finally:
        await kernel.shutdown()


async def test_root_has_no_final_and_no_context(server, tmp_path):
    env = kernel_env(server.port, server.issue("root"), "root", can_delegate=True, is_child=False)
    kernel = KernelSession(cwd=str(tmp_path), env=env, startup_code=KERNEL_API, timeout=30)
    try:
        assert "NameError" in (await kernel.run("FINAL")).error
        assert "NameError" in (await kernel.run("CONTEXT")).error
    finally:
        await kernel.shutdown()


async def test_a_request_is_cancelled_when_its_caller_hangs_up():
    import asyncio

    stopped = asyncio.Event()

    async def handler(agent, op, args):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stopped.set()
            raise

    server = BridgeServer(handler)
    await server.start()
    try:
        token = server.issue("a")
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        writer.write((json.dumps({"token": token, "op": "rlm", "args": {}}) + "\n").encode())
        await writer.drain()
        await asyncio.sleep(0.3)
        writer.close()
        await asyncio.wait_for(stopped.wait(), timeout=5)
    finally:
        await server.close()


async def _raw(port, data: bytes):
    import asyncio

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(data)
    await writer.drain()
    line = await reader.readline()
    writer.close()
    return json.loads(line) if line else None


async def test_malformed_requests_get_an_error_reply(server):
    assert "error" in await _raw(server.port, b"not json\n")
    assert "error" in await _raw(server.port, b"[1, 2]\n")
    assert server.calls == []


async def test_an_oversized_request_gets_an_error_reply(monkeypatch):
    import rlmagent_app.agents.bridge as module

    monkeypatch.setattr(module, "LINE_LIMIT", 1024)

    async def handler(agent, op, args):
        return None

    s = BridgeServer(handler)
    await s.start()
    try:
        reply = await _raw(s.port, b"x" * 5000 + b"\n")
        assert reply and "too large" in reply["error"]
    finally:
        await s.close()
