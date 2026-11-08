"""Persistent, independently locked Responses connections per agent affinity."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from rlmagent.model._oai.helpers import ServerSentEvent
from rlmagent.model._oai.transport import HttpStreamError
from rlmagent.model.transport.client import certificate_context, patched_proxy_environment


def canonical(items):
    result = []
    for item in items:
        kind = item.get("type")
        if kind == "reasoning":
            continue
        if item.get("role"):
            content = item.get("content", [])
            if isinstance(content, str):
                content = [
                    {
                        "type": "output_text" if item["role"] == "assistant" else "input_text",
                        "text": content,
                    }
                ]
            result.append(
                {
                    "role": item["role"],
                    "content": [
                        {
                            key: value
                            for key, value in part.items()
                            if key in ("type", "text", "image_url", "detail")
                        }
                        for part in content
                    ],
                }
            )
        elif kind == "function_call":
            arguments = item.get("arguments", "{}")
            try:
                arguments = json.dumps(json.loads(arguments), sort_keys=True)
            except (ValueError, TypeError):
                pass
            result.append(
                {
                    "type": kind,
                    "call_id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": arguments,
                }
            )
        else:
            result.append(item)
    return result


@dataclass
class Lane:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    socket: object = None
    auth: str | None = None
    created: float = 0
    previous: str | None = None
    prefix: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)


class ResponsesSockets:
    def __init__(self, *, timeout=120, connect=None):
        self.timeout = timeout
        self.connect = connect
        self.lanes = {}
        self.last_error = None
        self.last_error_location = []

    async def _discard(self, lane):
        if lane.socket is not None:
            with suppress(Exception):
                await lane.socket.close()
        lane.socket, lane.previous, lane.prefix = None, None, []

    async def close(self):
        for lane in self.lanes.values():
            await self._discard(lane)
        self.lanes.clear()

    async def stream(self, url, payload, headers, *, signal=None, affinity=None):
        try:
            from websockets.asyncio.client import connect
            from websockets.exceptions import InvalidStatus
        except ImportError:
            raise httpx.ConnectError(
                "Install rlmagent[subscription] for WebSocket transport"
            ) from None
        key = affinity or "root"
        temporary = key not in self.lanes and len(self.lanes) >= 32
        lane = Lane() if temporary else self.lanes.setdefault(key, Lane())
        async with lane.lock:
            socket_url = str(httpx.URL(url).copy_with(scheme="wss"))
            auth = {
                name: value
                for name, value in headers.items()
                if name.lower() in ("authorization", "openai-organization", "openai-project")
            }
            fingerprint = hashlib.sha256(
                (socket_url + json.dumps(auth, sort_keys=True)).encode()
            ).hexdigest()
            try:
                if lane.auth != fingerprint or time.monotonic() - lane.created > 2700:
                    await self._discard(lane)
                if lane.socket is None:
                    with patched_proxy_environment():
                        lane.socket = await (self.connect or connect)(
                            socket_url,
                            additional_headers=auth,
                            ssl=certificate_context(),
                            open_timeout=20,
                            max_size=16 * 1024 * 1024,
                        )
                    lane.auth, lane.created = fingerprint, time.monotonic()
                request = {
                    key: value
                    for key, value in payload.items()
                    if key not in ("stream", "background")
                }
                original = request.get("input", [])
                settings = {key: value for key, value in request.items() if key != "input"}
                same_prefix = canonical(original[: len(lane.prefix)]) == lane.prefix
                if lane.previous and lane.prefix and same_prefix and settings == lane.settings:
                    request["previous_response_id"] = lane.previous
                    request["input"] = original[len(lane.prefix) :]
                for recovery in range(2):
                    await lane.socket.send(json.dumps({"type": "response.create", **request}))
                    visible = False
                    deadline = time.monotonic() + self.timeout
                    while True:
                        receiver = asyncio.create_task(lane.socket.recv())
                        try:
                            while not receiver.done():
                                if signal is not None and signal.is_cancelled():
                                    await self._discard(lane)
                                    return
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("Responses socket timed out")
                                await asyncio.wait({receiver}, timeout=0.05)
                            raw = receiver.result()
                        finally:
                            if not receiver.done():
                                receiver.cancel()
                                with suppress(asyncio.CancelledError):
                                    await receiver
                        event = json.loads(raw)
                        kind = event.get("type", "")
                        failure = event.get("error") or event.get("response", {}).get("error") or {}
                        if (
                            failure.get("code") == "previous_response_not_found"
                            and recovery == 0
                            and not visible
                            and "previous_response_id" in request
                        ):
                            request.pop("previous_response_id")
                            request["input"] = original
                            lane.previous, lane.prefix = None, []
                            from rlmagent.agents.trace import model_attempt

                            model_attempt(
                                attempt=recovery + 1, transport="websocket", phase="cache_recovery"
                            )
                            break
                        visible = visible or kind in (
                            "response.output_text.delta",
                            "response.output_item.added",
                        )
                        if kind == "response.incomplete":
                            event = {
                                "type": "error",
                                "error": {"message": "Response was incomplete"},
                            }
                            kind = "error"
                        yield ServerSentEvent(event=kind, data=json.dumps(event))
                        if kind == "response.completed":
                            response = event["response"]
                            lane.previous = response.get("id")
                            lane.prefix = canonical([*original, *response.get("output", [])])
                            lane.settings = settings
                            return
                        if kind in ("response.failed", "error"):
                            lane.previous, lane.prefix = None, []
                            return
            except InvalidStatus as error:
                await self._discard(lane)
                raise HttpStreamError(
                    error.response.status_code, "WebSocket upgrade rejected"
                ) from None
            except asyncio.CancelledError:
                await self._discard(lane)
                raise
            except HttpStreamError:
                raise
            except Exception as error:
                self.last_error = type(error).__name__
                self.last_error_location = [
                    f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                    for frame in traceback.extract_tb(error.__traceback__)
                ]
                await self._discard(lane)
                raise httpx.ReadError(
                    f"Responses WebSocket disconnected ({self.last_error})"
                ) from None
            finally:
                if temporary:
                    await self._discard(lane)
