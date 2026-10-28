"""The ACP methods over JSON-RPC on stdio, one CodingSession per ACP session."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import uuid
from collections.abc import Awaitable, Callable

from rlmagent.acp import updates
from rlmagent.agents.approval import ALWAYS, DENY, ONCE

PROTOCOL_VERSION = 1
OPTIONS = [
    {"optionId": ONCE, "name": "Allow once", "kind": "allow_once"},
    {"optionId": ALWAYS, "name": "Always this session", "kind": "allow_always"},
    {"optionId": DENY, "name": "Deny", "kind": "reject_once"},
]
MODES = [
    {"id": "default", "name": "Default", "description": "Reads, writes and runs (writes and commands ask first)"},
    {"id": "plan", "name": "Plan", "description": "Reads and plans; changes nothing"},
]

ReadLine = Callable[[], Awaitable[bytes]]
Write = Callable[[bytes], None]


def _ids_file():
    from rlmagent.plugins import rlm_agent_home

    return rlm_agent_home() / "acp_sessions.json"


def _recall(sid: str) -> str:
    """Our saved-session id for an editor's session id (they differ for sessions started here)."""
    with contextlib.suppress(OSError, ValueError):
        return json.loads(_ids_file().read_text(encoding="utf-8")).get(sid, sid)
    return sid


def _remember(sid: str, session_id: str) -> None:
    path = _ids_file()
    known: dict = {}
    with contextlib.suppress(OSError, ValueError):
        known = json.loads(path.read_text(encoding="utf-8"))
    known[sid] = session_id
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(known), encoding="utf-8")


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def flatten(prompt: list[dict]) -> str:
    """Prompt blocks as one text: resources become fenced file text."""
    parts = []
    for block in prompt:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "resource":
            res = block.get("resource", {})
            parts.append(f"{res.get('uri', '')}\n```\n{res.get('text', '')}\n```")
        elif kind == "resource_link":
            parts.append(block.get("uri", ""))
    return "\n\n".join(p for p in parts if p)


class Server:
    def __init__(self, read_line: ReadLine, write: Write, argv: list[str] | None = None,
                 build=None) -> None:
        self.read_line = read_line
        self.write = write
        self.argv = argv or []
        self._build = build
        self.sessions: dict[str, object] = {}
        self.pending: dict[str, str] = {}
        self.modes: dict[str, str] = {}
        self._starting = asyncio.Lock()
        self.running: dict[str, asyncio.Task] = {}
        self._next_id = 0
        self._waiting: dict[int, asyncio.Future] = {}
        self._tasks: set[asyncio.Task] = set()

    # -- wire ---------------------------------------------------------------

    def _send(self, message: dict) -> None:
        self.write((json.dumps({"jsonrpc": "2.0", **message}) + "\n").encode())

    def notify(self, method: str, params: dict) -> None:
        self._send({"method": method, "params": params})

    async def request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._waiting[self._next_id] = future
        self._send({"id": self._next_id, "method": method, "params": params})
        return await future

    async def serve(self) -> None:
        while True:
            line = await self.read_line()
            if not line:
                break
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._send({"id": None, "error": {"code": -32700, "message": "parse error"}})
                continue
            if "method" not in message:
                future = self._waiting.pop(message.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(message.get("result") or {})
                continue
            task = asyncio.create_task(self._dispatch(message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        await self.close()

    async def _dispatch(self, message: dict) -> None:
        handler = getattr(self, "on_" + message["method"].replace("/", "_"), None)
        msg_id = message.get("id")
        try:
            if handler is None:
                raise RpcError(-32601, f"method not found: {message['method']}")
            result = await handler(message.get("params") or {})
        except RpcError as exc:
            if msg_id is not None:
                self._send({"id": msg_id, "error": {"code": exc.code, "message": str(exc)}})
            return
        except Exception as exc:  # noqa: BLE001 - reported to the editor, not raised
            if msg_id is not None:
                self._send({"id": msg_id, "error": {"code": -32603, "message": repr(exc)}})
            return
        if msg_id is not None:
            self._send({"id": msg_id, "result": result if result is not None else {}})

    # -- methods ------------------------------------------------------------

    async def on_initialize(self, params: dict) -> dict:
        return {"protocolVersion": PROTOCOL_VERSION,
                "agentCapabilities": {"loadSession": True,
                                      "promptCapabilities": {"embeddedContext": True}},
                "authMethods": []}

    async def _open(self, sid: str, cwd: str, resume: str | None):
        from rlmagent.cli.main import _build_run_parser

        ns = _build_run_parser().parse_args(self.argv)
        ns.resume = resume
        ns.trust_ask = lambda request: self._ask_trust(sid, request)
        build = self._build
        if build is None:
            from rlmagent.runtime import build_session as build
        # Start-up reads the process's working folder, so sessions start one at a time.
        async with self._starting:
            if cwd:
                os.chdir(cwd)
            try:
                session = await build(ns)
            except SystemExit as exc:
                # Start-up failures exit the terminal app; here they answer this request only.
                raise RpcError(-32603, f"could not start a session (exit {exc.code}); "
                                       "see the agent's stderr") from None
        tree = getattr(session, "agent_tree", None)
        if tree is not None and tree.approver.policy == "ask":
            tree.approver.ask = lambda request: self._ask(sid, request)
        if self.modes.get(sid) == "plan":
            session.set_plan_mode(True)
        self.sessions[sid] = session
        if session.session_id != sid:
            _remember(sid, session.session_id)
        return session

    async def _ask_trust(self, sid: str, request):
        from rlmagent.runtime import trust_choices, trust_question

        menu = dict((choice.value, choice) for choice, _ in trust_choices(request))
        options = [{"optionId": choice.value, "name": label,
                    "kind": ("reject_" if choice.value.startswith("distrust") else "allow_")
                    + ("once" if choice.value.endswith("once") else "always")}
                   for choice, label in trust_choices(request)]
        reply = await self.request("session/request_permission", {
            "sessionId": sid,
            "toolCall": {"toolCallId": "project-trust", "title": trust_question(request),
                         "kind": "other", "status": "pending"},
            "options": options})
        outcome = reply.get("outcome", {})
        return menu.get(outcome.get("optionId")) if outcome.get("outcome") == "selected" else None

    async def _ask(self, session_id: str, request) -> str:
        reply = await self.request("session/request_permission", {
            "sessionId": session_id,
            "toolCall": {"toolCallId": f"approval-{request.digest[:12]}",
                         "title": request.describe(), "kind": "edit" if request.kind == "write"
                         else "execute", "status": "pending"},
            "options": OPTIONS})
        outcome = reply.get("outcome", {})
        return outcome.get("optionId", DENY) if outcome.get("outcome") == "selected" else DENY

    def _modes(self, sid: str) -> dict:
        return {"currentModeId": self.modes.get(sid, "default"), "availableModes": MODES}

    async def on_session_new(self, params: dict) -> dict:
        # The session starts with its first prompt, so the folder-trust question (if any)
        # reaches the editor while a prompt is running, where editors show permission requests.
        sid = uuid.uuid4().hex[:12]
        self.pending[sid] = params.get("cwd", "")
        return {"sessionId": sid, "modes": self._modes(sid)}

    async def on_session_load(self, params: dict) -> dict:
        sid = params["sessionId"]
        session = await self._open(sid, params.get("cwd", ""), _recall(sid))
        for update in updates.replay(session.transcript):
            self.notify("session/update", {"sessionId": sid, "update": update})
        return {"modes": self._modes(sid)}

    async def on_session_set_mode(self, params: dict) -> dict:
        sid, mode = params.get("sessionId", ""), params.get("modeId", "")
        if mode not in {m["id"] for m in MODES}:
            raise RpcError(-32602, f"unknown mode: {mode}")
        if sid not in self.sessions and sid not in self.pending:
            raise RpcError(-32602, f"unknown session: {sid}")
        self.modes[sid] = mode
        if sid in self.sessions:
            self.sessions[sid].set_plan_mode(mode == "plan")
        self.notify("session/update", {"sessionId": sid, "update": {
            "sessionUpdate": "current_mode_update", "currentModeId": mode}})
        return {}

    async def on_session_prompt(self, params: dict) -> dict:
        from rlmagent.tools._shared import set_workspace

        sid = params["sessionId"]
        if sid in self.running:
            raise RpcError(-32602, "a prompt is already running in this session")
        self.running[sid] = asyncio.current_task()
        try:
            if sid in self.pending:
                await self._open(sid, self.pending.pop(sid), None)
            session = self.sessions.get(sid)
            if session is None:
                raise RpcError(-32602, f"unknown session: {sid}")
            set_workspace(session.cwd)  # this task's file tools stay in this session's folder
            return await self._prompt(sid, session, params)
        finally:
            self.running.pop(sid, None)

    async def _prompt(self, sid: str, session, params: dict) -> dict:
        translator = updates.Translator()
        watcher = asyncio.create_task(self._watch_plan(sid, session))
        messages: list = []
        try:
            async for event in session.submit(flatten(params.get("prompt", []))):
                for update in translator.updates(event):
                    self.notify("session/update", {"sessionId": sid, "update": update})
                if getattr(event, "type", "") == "agent_end":
                    messages = event.messages
        finally:
            watcher.cancel()
        return {"stopReason": updates.stop_reason(messages)}

    async def _watch_plan(self, sid: str, session) -> None:
        tree = getattr(session, "agent_tree", None)
        if tree is None:
            return
        last: list = []
        while True:
            entries = updates.plan(tree)
            if entries != last:
                self.notify("session/update", {"sessionId": sid,
                                               "update": {"sessionUpdate": "plan", "entries": entries}})
                last = entries
            await asyncio.sleep(0.5)

    async def on_session_cancel(self, params: dict) -> None:
        session = self.sessions.get(params.get("sessionId", ""))
        if session is not None:
            session.abort()
        for future in self._waiting.values():
            if not future.done():
                future.set_result({"outcome": {"outcome": "cancelled"}})

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for session in self.sessions.values():
            with contextlib.suppress(Exception):
                await session.shutdown()
        self.sessions.clear()


def run(argv: list[str]) -> int:
    """Serve on this process's stdin/stdout; everything else printed goes to stderr."""
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    stdin = sys.stdin.buffer

    def write(data: bytes) -> None:
        out.write(data)
        out.flush()

    async def main() -> None:
        await Server(lambda: asyncio.to_thread(stdin.readline), write, argv).serve()

    asyncio.run(main())
    return 0
