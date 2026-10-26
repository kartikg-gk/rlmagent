"""Any agent that speaks the Agent Client Protocol, as a model."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import shutil
from pathlib import Path

from rlmagent.harness.contracts.tooling import CancelToken
from rlmagent.harness.contracts.transcript import UsageStats

_DENY_CLAUDE = ["Bash(*)", "Read(*)", "Write(*)", "Edit(*)", "Glob(*)", "Grep(*)",
                "WebFetch(*)", "WebSearch(*)", "Task(*)", "NotebookEdit(*)"]

PRESETS: dict[str, dict] = {
    "claude-code": {
        "command": "npx", "args": ["-y", "@agentclientprotocol/claude-agent-acp@0.89.1"],
        "files": {".claude/settings.json": {"permissions": {"deny": _DENY_CLAUDE}}},
    },
    "codex": {
        "command": "npx", "args": ["-y", "@agentclientprotocol/codex-acp@2.2.2"],
        "readonly_mode": "read-only",
        "approve": True,
    },
    "opencode": {
        "command": "opencode", "args": ["acp"], "readonly_mode": "plan",
        "files": {"opencode.json": {"permission": {k: "deny" for k in (
            "bash", "edit", "read", "glob", "grep", "list", "webfetch", "task")}}},
    },
}

START_TIMEOUT = 180.0


def agents() -> dict[str, dict]:
    """The presets, plus any agents registered in ~/.rlm-agent/acp_agents.json."""
    from rlmagent.plugins import rlm_agent_home

    found = dict(PRESETS)
    path = rlm_agent_home() / "acp_agents.json"
    try:
        found.update(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        pass
    return found


class Cancelled(Exception):
    pass


class _Connection:
    """JSON-RPC 2.0, one message per line, over the agent's stdin and stdout."""

    def __init__(self, proc, approve: bool = False) -> None:
        self.proc = proc
        self.approve = approve
        self.ids = itertools.count(1)
        self.waiting: dict[int, asyncio.Future] = {}
        self.text: list[str] = []
        self.closed = asyncio.get_running_loop().create_future()

    async def _send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", **message}) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()

    async def request(self, method: str, params: dict) -> dict:
        rid = next(self.ids)
        future = asyncio.get_running_loop().create_future()
        self.waiting[rid] = future
        await self._send({"id": rid, "method": method, "params": params})
        done, _ = await asyncio.wait({future, self.closed}, return_when=asyncio.FIRST_COMPLETED)
        if future not in done:
            raise ValueError(f"the agent exited during {method}")
        return future.result()

    async def notify(self, method: str, params: dict) -> None:
        await self._send({"method": method, "params": params})

    async def pump(self) -> None:
        try:
            while line := await self.proc.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trace := os.environ.get("RLM_AGENT_ACP_TRACE"):
                    with open(trace, "a", encoding="utf-8") as f:
                        f.write(line.decode("utf-8", errors="replace"))
                await self._dispatch(message)
        finally:
            if not self.closed.done():
                self.closed.set_result(None)

    async def _dispatch(self, message: dict) -> None:
        method = message.get("method")
        if method is None:
            future = self.waiting.pop(message.get("id"), None)
            if future is not None and not future.done():
                if "error" in message:
                    future.set_exception(ValueError(message["error"].get("message", "agent error")))
                else:
                    future.set_result(message.get("result") or {})
            return
        params = message.get("params") or {}
        if method == "session/update":
            update = params.get("update") or {}
            content = update.get("content") or {}
            if update.get("sessionUpdate") == "agent_message_chunk" and content.get("type") == "text":
                self.text.append(content.get("text", ""))
            return
        if "id" not in message:
            return
        if method == "session/request_permission":
            wanted = "allow_once" if self.approve else "reject"
            picks = [o for o in params.get("options", []) if o.get("kind", "").startswith(wanted)]
            outcome = ({"outcome": "selected", "optionId": picks[0]["optionId"]} if picks
                       else {"outcome": "cancelled"})
            await self._send({"id": message["id"], "result": {"outcome": outcome}})
        else:
            await self._send({"id": message["id"],
                              "error": {"code": -32601, "message": f"{method} is not offered"}})


def _usage(result: dict) -> UsageStats:
    u = result.get("usage") or {}
    kw = {"input": u.get("inputTokens", 0), "output": u.get("outputTokens", 0),
          "cache_read": u.get("cachedReadTokens", 0), "cache_write": u.get("cachedWriteTokens", 0)}
    return UsageStats(total_tokens=u.get("totalTokens") or sum(kw.values()), **kw)


async def answer(name: str, sub: str | None, prompt: str, cwd: Path,
                 signal: CancelToken | None) -> tuple[str, UsageStats]:
    spec = agents()[name]
    for rel, content in (spec.get("files") or {}).items():
        target = cwd / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(content), encoding="utf-8")
    exe = shutil.which(spec["command"])
    if exe is None:
        raise ValueError(f"{spec['command']} not found on PATH")
    with open(cwd / "agent-stderr.log", "wb") as errors:
        proc = await asyncio.create_subprocess_exec(
            exe, *spec.get("args", []), cwd=str(cwd), limit=2**24,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=errors,
            env={**os.environ, **(spec.get("env") or {})},
        )
    conn = _Connection(proc, approve=bool(spec.get("approve")))
    pump = asyncio.create_task(conn.pump())
    try:
        await asyncio.wait_for(conn.request("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False},
                                   "terminal": False},
            "clientInfo": {"name": "rlm-agent", "version": "0"},
        }), START_TIMEOUT)
        session = await conn.request("session/new", {"cwd": str(cwd), "mcpServers": []})
        sid = session["sessionId"]
        modes = {m.get("id") for m in (session.get("modes") or {}).get("availableModes", [])}
        if spec.get("readonly_mode") in modes:
            await conn.request("session/set_mode", {"sessionId": sid, "modeId": spec["readonly_mode"]})
        model = sub or spec.get("model")
        if model and session.get("models"):
            await conn.request("session/set_model", {"sessionId": sid, "modelId": model})
        turn = asyncio.ensure_future(conn.request("session/prompt", {
            "sessionId": sid, "prompt": [{"type": "text", "text": prompt}]}))
        while not turn.done():
            if signal is not None and signal.is_cancelled():
                await conn.notify("session/cancel", {"sessionId": sid})
                turn.cancel()
                raise Cancelled
            await asyncio.wait({turn}, timeout=0.2)
        result = turn.result()
    finally:
        from rlmagent.model.external.cli import kill_tree

        kill_tree(proc)
        await proc.wait()
        pump.cancel()
    if result.get("stopReason") == "refusal":
        raise ValueError(f"{name} refused")
    return "".join(conn.text), _usage(result)
