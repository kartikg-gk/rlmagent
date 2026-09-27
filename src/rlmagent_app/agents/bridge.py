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
    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self.token = secrets.token_hex(16)
        self.port = 0
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._serve, "127.0.0.1", 0, limit=LINE_LIMIT
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            request = json.loads(line)
            if not secrets.compare_digest(str(request.get("token", "")), self.token):
                reply: dict = {"error": "refused: bad token"}
            else:
                try:
                    value = await self._handler(
                        str(request["agent"]), str(request["op"]), dict(request.get("args") or {})
                    )
                    reply = {"ok": value}
                except Exception as exc:  # the kernel sees it as an exception in its cell
                    reply = {"error": str(exc) or type(exc).__name__}
            writer.write((json.dumps(reply, default=repr) + "\n").encode())
            await writer.drain()
        except (json.JSONDecodeError, ConnectionError):
            pass
        finally:
            writer.close()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
