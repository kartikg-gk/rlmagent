import asyncio
import json

import pytest

# The WebSocket transport is part of the optional subscription extra.
pytest.importorskip("websockets")

from rlmagent.model.transport.responses_socket import ResponsesSockets  # noqa: E402


class Socket:
    def __init__(self, *, miss=False):
        self.sent = []
        self.closed = False
        self.queue = asyncio.Queue()
        self.miss = miss

    async def send(self, text):
        body = json.loads(text)
        self.sent.append(body)
        if self.miss and body.get("previous_response_id"):
            await self.queue.put(
                {"type": "error", "error": {"code": "previous_response_not_found"}}
            )
        else:
            await self.queue.put(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "response-" + str(len(self.sent)),
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "answer", "annotations": []}
                                ],
                            }
                        ],
                    },
                }
            )

    async def recv(self):
        return json.dumps(await self.queue.get())

    async def close(self):
        self.closed = True


async def collect(pool, body, **kw):
    return [
        event
        async for event in pool.stream(
            "https://api.openai.com/v1/responses", body, {"Authorization": "Bearer fixture"}, **kw
        )
    ]


async def test_reuses_connection_and_sends_incremental_input():
    sockets = []

    async def connect(url, **kw):
        assert url == "wss://api.openai.com/v1/responses"
        assert kw["additional_headers"] == {"Authorization": "Bearer fixture"}
        socket = Socket()
        sockets.append(socket)
        return socket

    pool = ResponsesSockets(connect=connect)
    user = {"role": "user", "content": "question"}
    body = {"model": "m", "store": False, "stream": True, "input": [user]}
    await collect(pool, body)
    followup = {"role": "user", "content": [{"type": "input_text", "text": "next"}]}
    body["input"] += [
        {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        followup,
    ]
    await collect(pool, body)
    assert len(sockets) == 1
    assert sockets[0].sent[1]["previous_response_id"] == "response-1"
    assert sockets[0].sent[1]["input"] == [followup]
    assert "stream" not in sockets[0].sent[1]
    await pool.close()
    assert sockets[0].closed


async def test_cache_miss_resends_full_context_once_and_agents_use_separate_sockets():
    sockets = []

    async def connect(*args, **kw):
        socket = Socket(miss=True)
        sockets.append(socket)
        return socket

    pool = ResponsesSockets(connect=connect)
    body = {"model": "m", "input": [{"role": "user", "content": []}]}
    await asyncio.gather(collect(pool, body, affinity="a"), collect(pool, body, affinity="b"))
    assert len(sockets) == 2
    body["input"] += [{"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}]
    await collect(pool, body, affinity="a")
    assert len(sockets[0].sent) == 3
    assert "previous_response_id" in sockets[0].sent[1]
    assert "previous_response_id" not in sockets[0].sent[2]
    assert sockets[0].sent[2]["input"] == body["input"]
    await pool.close()


async def test_visible_disconnect_is_not_replayed(monkeypatch):
    from rlmagent.model import oai_compatible
    from rlmagent.model.settings import Credential, OpenAIProfile

    class Broken(Socket):
        async def send(self, text):
            self.sent.append(json.loads(text))

        async def recv(self):
            if not getattr(self, "visible", False):
                self.visible = True
                return json.dumps({"type": "response.output_text.delta", "delta": "partial"})
            raise OSError("lost connection")

    socket = Broken()

    async def connect(*args, **kw):
        return socket

    provider = oai_compatible.OpenAIProvider(
        OpenAIProfile(
            name="codex",
            credential=Credential("fixture", "https://api.openai.com/v1"),
            endpoint="responses",
            transport="websocket",
        )
    )
    provider._sockets = ResponsesSockets(connect=connect)
    try:
        events = [
            event
            async for event in provider.stream_response(
                model="m", system="s", messages=[], tools=[]
            )
        ]
        assert events[-1].type == "error" and len(socket.sent) == 1
    finally:
        await provider.close()


async def test_cancellation_closes_own_socket():
    from rlmagent.harness.driver import ManualCancelFlag

    class Slow(Socket):
        async def send(self, text):
            pass

    socket = Slow()

    async def connect(*args, **kw):
        return socket

    pool = ResponsesSockets(connect=connect)
    signal = ManualCancelFlag()
    task = asyncio.create_task(collect(pool, {"input": []}, signal=signal))
    await asyncio.sleep(0.05)
    signal.cancel()
    assert await asyncio.wait_for(task, 2) == []
    assert socket.closed
    await pool.close()
