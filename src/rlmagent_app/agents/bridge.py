"""A localhost channel that only this run's kernels can use."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable

Handler = Callable[[str, str, dict], Awaitable[object]]
LINE_LIMIT = 2**26  # 64 MB: a sub-agent may hand back a large value


def kernel_env(port: int, token: str, agent_id: str, *, can_delegate: bool, is_child: bool) -> dict[str, str]:
    return {
        "RLM_AGENT_BRIDGE_PORT": str(port),
        "RLM_AGENT_BRIDGE_TOKEN": token,
        "RLM_AGENT_ID": agent_id,
        "RLM_AGENT_CAN_DELEGATE": "1" if can_delegate else "0",
        "RLM_AGENT_IS_CHILD": "1" if is_child else "0",
    }


class BridgeServer:
    """Serves requests from kernels. Each agent gets its own token, and the
    token alone says who is calling: a kernel can rewrite anything it sends,
    so nothing else in a request is trusted for identity."""

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self._agents: dict[str, str] = {}
        self._running: set[asyncio.Task] = set()
        self.port = 0
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._serve, "127.0.0.1", 0, limit=LINE_LIMIT
        )
        self.port = self._server.sockets[0].getsockname()[1]

    def issue(self, agent_id: str) -> str:
        token = secrets.token_hex(16)
        self._agents[token] = agent_id
        return token

    def revoke(self, token: str) -> None:
        self._agents.pop(token, None)

    def _caller(self, token: object) -> str | None:
        for known, agent_id in self._agents.items():
            if secrets.compare_digest(str(token), known):
                return agent_id
        return None

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            request = json.loads(line)
            caller = self._caller(request.get("token", "")) if isinstance(request, dict) else None
            if caller is None:
                reply: dict = {"error": "refused: bad token"}
            else:
                reply = await self._answer(caller, request, reader)
                if reply is None:
                    return  # the caller hung up; nobody is left to answer
            writer.write((json.dumps(reply, default=repr) + "\n").encode())
            await writer.drain()
        except (json.JSONDecodeError, ConnectionError):
            pass
        finally:
            writer.close()

    async def _answer(self, caller: str, request: dict, reader: asyncio.StreamReader) -> dict | None:
        """Run the request, stopping it if the kernel hangs up first.

        A kernel that was restarted or killed mid-call closes its end. The
        sub-agent it was waiting on would otherwise run on, spending budget
        on an answer nobody can receive.
        """
        work = asyncio.create_task(
            self._handler(caller, str(request["op"]), dict(request.get("args") or {}))
        )
        self._running.add(work)
        hangup = asyncio.create_task(reader.read(1))
        try:
            await asyncio.wait({work, hangup}, return_when=asyncio.FIRST_COMPLETED)
            if not work.done():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                return None
            try:
                return {"ok": work.result()}
            except Exception as exc:  # the kernel sees it as an exception in its cell
                return {"error": str(exc) or type(exc).__name__}
        finally:
            hangup.cancel()
            self._running.discard(work)

    async def close(self) -> None:
        for work in list(self._running):
            work.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
