"""Code run in every kernel before the model's first cell."""

KERNEL_API = r'''
import asyncio as _rlmagent_asyncio
import json as _rlmagent_json
import os as _rlmagent_os
import socket as _rlmagent_socket

_RLMAGENT = {
    "port": int(_rlmagent_os.environ["RLM_AGENT_BRIDGE_PORT"]),
    "token": _rlmagent_os.environ["RLM_AGENT_BRIDGE_TOKEN"],
}


def _rlmagent_payload(op, args):
    return (_rlmagent_json.dumps({"token": _RLMAGENT["token"], "op": op, "args": args})
            + "\n").encode()


def _rlmagent_reply(line):
    reply = _rlmagent_json.loads(line)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["ok"]


def _rlmagent_call_sync(op, args):
    with _rlmagent_socket.create_connection(("127.0.0.1", _RLMAGENT["port"])) as sock:
        sock.sendall(_rlmagent_payload(op, args))
        chunks = []
        while True:
            chunk = sock.recv(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
    return _rlmagent_reply(b"".join(chunks))


async def _rlmagent_call(op, args):
    reader, writer = await _rlmagent_asyncio.open_connection(
        "127.0.0.1", _RLMAGENT["port"], limit=2**26)
    writer.write(_rlmagent_payload(op, args))
    await writer.drain()
    line = await reader.readline()
    writer.close()
    return _rlmagent_reply(line)


if _rlmagent_os.environ.get("RLM_AGENT_CAN_DELEGATE") == "1":
    async def rlm(task, context=None):
        """Hand `task` to a sub-agent; `context` arrives in its kernel as CONTEXT. Returns its FINAL value."""
        return await _rlmagent_call("rlm", {"task": str(task), "context": context})

    async def gather_rlm(jobs):
        """Run several sub-agents at once. `jobs` is a list of (task, context) pairs or tasks. Results keep order."""
        pairs = [list(j) if isinstance(j, (tuple, list)) else [j, None] for j in jobs]
        return await _rlmagent_call("gather", {"jobs": [[str(t), c] for t, c in pairs]})

if _rlmagent_os.environ.get("RLM_AGENT_IS_CHILD") == "1":
    def FINAL(value):
        """Return `value` to the agent that started you. Call once, when done."""
        try:
            _rlmagent_json.dumps(value)
            args = {"value": value, "note": ""}
        except (TypeError, ValueError):
            args = {"value": repr(value), "note": "value was not JSON-serialisable; sent as repr"}
        _rlmagent_call_sync("final", args)
        print("FINAL recorded.")

    CONTEXT = _rlmagent_call_sync("context", {})
'''
