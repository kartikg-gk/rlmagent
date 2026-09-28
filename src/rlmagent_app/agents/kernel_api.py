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
        kind = TimeoutError if reply.get("type") == "TimeoutError" else RuntimeError
        raise kind(reply["error"])
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

class _SubAgent:
    """A sub-agent started with spawn(). Losing this object does not stop it."""

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"<sub-agent {self.id}>"

    async def status(self):
        """'starting', 'running', 'idle', 'done', 'failed' or 'cancelled'."""
        return await _rlmagent_call("status", {"id": self.id})

    async def result(self, timeout=None):
        """Its FINAL value; waits for it. TimeoutError if still running after `timeout` seconds."""
        return await _rlmagent_call("result", {"id": self.id, "timeout": timeout})

    async def cancel(self):
        """Stop it and every sub-agent it started."""
        return await _rlmagent_call("cancel", {"id": self.id})

    async def send(self, text):
        """Give it another instruction; wakes it if it is idle."""
        return await _rlmagent_call("send", {"id": self.id, "text": str(text)})

    async def messages(self):
        """Notes it sent with tell_parent, oldest first; each is returned once."""
        return await _rlmagent_call("messages", {"id": self.id})


if _rlmagent_os.environ.get("RLM_AGENT_CAN_DELEGATE") == "1":
    async def spawn(task, data=None, keep=False):
        """Start a sub-agent and return at once. keep=True: it stays idle after FINAL, ready for send()."""
        return _SubAgent(await _rlmagent_call(
            "spawn", {"task": str(task), "context": data, "keep": bool(keep)}))

    async def children():
        """Every sub-agent you started, in order."""
        return [_SubAgent(i) for i in await _rlmagent_call("children", {})]

if _rlmagent_os.environ.get("RLM_AGENT_IS_CHILD") == "1":
    async def tell_parent(text):
        """Leave a note the agent that started you can read with messages()."""
        await _rlmagent_call("tell_parent", {"text": str(text)})

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
