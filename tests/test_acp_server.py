import asyncio
import json

from _scripted import code_turn, text_turn

from rlmagent.acp import updates
from rlmagent.acp.server import Server, flatten
from rlmagent.agents.approval import ONCE, Approver, Request
from rlmagent.conversation import CodingSession
from rlmagent.model.scripted import ReplayProvider


class Pipe:
    """The editor side: lines in, lines out."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.new = asyncio.Event()

    async def read_line(self) -> bytes:
        return await self.inbox.get()

    def write(self, data: bytes) -> None:
        self.sent.append(json.loads(data))
        self.new.set()

    def send(self, msg_id, method, params) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        if msg_id is not None:
            msg["id"] = msg_id
        self.inbox.put_nowait((json.dumps(msg) + "\n").encode())

    async def reply(self, msg_id) -> dict:
        while True:
            for m in self.sent:
                if m.get("id") == msg_id and "method" not in m:
                    return m
            self.new.clear()
            await asyncio.wait_for(self.new.wait(), 10)


def _builder(tmp_path, turns):
    async def build(ns):
        session = await CodingSession.create(
            provider=ReplayProvider(list(turns)), provider_name="replay", model="m", system="s",
            tools=[], cwd=str(tmp_path))
        await session.set_name("t")
        return session
    return build


async def _start(tmp_path, turns):
    pipe = Pipe()
    server = Server(pipe.read_line, pipe.write, build=_builder(tmp_path, turns))
    task = asyncio.create_task(server.serve())
    pipe.send(1, "initialize", {"protocolVersion": 1})
    assert (await pipe.reply(1))["result"]["agentCapabilities"]["loadSession"]
    pipe.send(2, "session/new", {"cwd": str(tmp_path), "mcpServers": []})
    sid = (await pipe.reply(2))["result"]["sessionId"]
    return pipe, server, task, sid


async def _stop(pipe, task):
    pipe.inbox.put_nowait(b"")
    await asyncio.wait_for(task, 10)


async def test_a_prompt_streams_text_and_ends_the_turn(tmp_path):
    pipe, _, task, sid = await _start(tmp_path, [text_turn("hello there")])
    try:
        pipe.send(3, "session/prompt", {"sessionId": sid,
                                        "prompt": [{"type": "text", "text": "hi"}]})
        assert (await pipe.reply(3))["result"] == {"stopReason": "end_turn"}
        text = "".join(m["params"]["update"]["content"]["text"] for m in pipe.sent
                       if m.get("method") == "session/update"
                       and m["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
        assert text == "hello there"
    finally:
        await _stop(pipe, task)


async def test_unknown_methods_and_sessions_are_errors(tmp_path):
    pipe, _, task, _ = await _start(tmp_path, [])
    try:
        pipe.send(4, "nope", {})
        assert (await pipe.reply(4))["error"]["code"] == -32601
        pipe.send(5, "session/prompt", {"sessionId": "missing", "prompt": []})
        assert (await pipe.reply(5))["error"]["code"] == -32602
    finally:
        await _stop(pipe, task)


async def test_approval_goes_to_the_editor(tmp_path):
    pipe, server, task, sid = await _start(tmp_path, [])
    try:
        asking = asyncio.create_task(server._ask(sid, Request("run", "make test")))
        while not any(m.get("method") == "session/request_permission" for m in pipe.sent):
            pipe.new.clear()
            await asyncio.wait_for(pipe.new.wait(), 10)
        ask = next(m for m in pipe.sent if m.get("method") == "session/request_permission")
        assert "make test" in ask["params"]["toolCall"]["title"]
        pipe.inbox.put_nowait((json.dumps({"jsonrpc": "2.0", "id": ask["id"], "result": {
            "outcome": {"outcome": "selected", "optionId": ONCE}}}) + "\n").encode())
        assert await asyncio.wait_for(asking, 10) == ONCE
    finally:
        await _stop(pipe, task)


def test_tool_events_and_prompt_blocks_translate():
    from rlmagent.harness.contracts.stream import ToolRunStartEvent

    [update] = updates.Translator().updates(ToolRunStartEvent(
        tool_call_id="c1", tool_name="python", args={"code": "x = 1\nprint(x)"}))
    assert update["kind"] == "execute" and update["title"] == "x = 1"
    assert flatten([{"type": "text", "text": "fix"},
                    {"type": "resource", "resource": {"uri": "a.py", "text": "x"}}]) \
        == "fix\n\na.py\n```\nx\n```"


__all__ = ["Approver", "code_turn"]
