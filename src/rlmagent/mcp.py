"""Model Context Protocol servers: their tools, reachable from cells and the session engine.

Servers are listed in mcp.json under the rlm-agent home, and in a trusted project's
.rlm-agent/mcp.json:

    {"servers": {
        "files": {"command": "npx", "args": ["-y", "some-mcp-server"], "env": {"TOKEN": "${MY_TOKEN}"}},
        "docs":  {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${DOCS_KEY}"}}
    }}

``${NAME}`` is filled from the environment when the server starts, so keys stay out of
the file. Servers start on first use. Every tool call goes through approval unless the
server marks the tool read-only.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL = "2025-06-18"
START_SECONDS = 30.0
CALL_SECONDS = 300.0
MAX_RESULT_CHARS = 100_000
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CONNECTORS = {
    "slack": ("https://mcp.slack.com/mcp", "https://mcp.slack.com"),
    "notion": ("https://mcp.notion.com/mcp", "https://mcp.notion.com"),
}


class McpError(RuntimeError):
    pass


def _fill(value):
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_fill(v) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v) for k, v in value.items()}
    return value


@dataclass
class ServerConfig:
    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    source: str = ""
    oauth: dict | None = None
    expand_env: bool = True

    def expand(self, value):
        return _fill(value) if self.expand_env else value

    @classmethod
    def parse(cls, name: str, raw: dict, source: str) -> ServerConfig:
        if isinstance(raw, dict) and "connector" in raw:
            connector = raw["connector"]
            if not isinstance(connector, str) or connector not in _CONNECTORS:
                raise McpError(f"{source}: unknown MCP connector {connector!r}")
            oauth = raw.get("oauth")
            if not isinstance(oauth, dict):
                raise McpError(f"{source}: connector {name!r} needs OAuth client configuration")
            url, issuer = _CONNECTORS[connector]
            raw = {**raw, "url": url, "oauth": {**oauth, "issuer": issuer}}
        if not isinstance(raw, dict) or bool(raw.get("command")) == bool(raw.get("url")):
            raise McpError(f"{source}: server {name!r} needs either command or url")
        oauth = raw.get("oauth")
        if oauth is not None and (
            not raw.get("url") or not isinstance(oauth, dict)
            or not all(isinstance(oauth.get(k), str) and oauth[k] for k in ("issuer", "client_id"))
            or type(oauth.get("redirect_port")) is not int
            or not 1 <= oauth["redirect_port"] <= 65535
            or not isinstance(oauth.get("scope", ""), str)
            or type(oauth.get("persist", False)) is not bool
            or not isinstance(oauth.get("client_secret_env", ""), str)
        ):
            raise McpError(f"{source}: server {name!r} needs OAuth issuer, client_id and redirect_port")
        return cls(name, raw.get("command"), [str(a) for a in raw.get("args", [])],
                   {str(k): str(v) for k, v in raw.get("env", {}).items()},
                   raw.get("url"), {str(k): str(v) for k, v in raw.get("headers", {}).items()},
                   source, oauth)


def load_configs(home: Path, project: Path | None) -> tuple[dict[str, ServerConfig], list[str]]:
    """Servers from the home file, then the project's (which wins on a name), and problems."""
    servers: dict[str, ServerConfig] = {}
    problems: list[str] = []
    files = [home / "mcp.json"]
    if project is not None:
        files.append(project / ".rlm-agent" / "mcp.json")
    for path in files:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for name, raw in (data.get("servers") or {}).items():
                servers[name] = ServerConfig.parse(name, raw, str(path))
        except (OSError, json.JSONDecodeError, McpError, AttributeError) as exc:
            problems.append(f"{path}: {exc}")
    return servers, problems


class _Stdio:
    """Newline-delimited JSON-RPC over a child process's stdin and stdout."""

    def __init__(self, config: ServerConfig, cwd: str) -> None:
        self.config, self.cwd = config, cwd
        self.proc: asyncio.subprocess.Process | None = None
        self._waiting: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None

    async def open(self) -> None:
        import shutil

        from rlmagent.model.external.cli import own_group

        command = shutil.which(self.config.command) or self.config.command
        self.proc = await asyncio.create_subprocess_exec(
            command, *self.config.expand(self.config.args), cwd=self.cwd,
            env={**os.environ, **self.config.expand(self.config.env)},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=16 * 1024 * 1024, **own_group())
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        while line := await self.proc.stdout.readline():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            waiter = self._waiting.pop(message.get("id"), None) if isinstance(message, dict) else None
            if waiter is not None and not waiter.done():
                waiter.set_result(message)
        for waiter in self._waiting.values():
            if not waiter.done():
                waiter.set_exception(McpError(f"{self.config.name} stopped"))
        self._waiting.clear()

    async def send(self, message: dict) -> dict | None:
        assert self.proc is not None and self.proc.stdin is not None
        waiter = None
        if "id" in message:
            waiter = asyncio.get_running_loop().create_future()
            self._waiting[message["id"]] = waiter
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        await self.proc.stdin.drain()
        return await waiter if waiter is not None else None

    async def close(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            from rlmagent.model.external.cli import kill_tree

            kill_tree(self.proc)
            await self.proc.wait()
        if self._reader is not None:
            self._reader.cancel()


class _Http:
    """JSON-RPC over HTTP POST; the reply is JSON or a stream of server-sent events."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.session_id: str | None = None
        self.client = None
        self.credential = None
        self.credential_key = None
        self._auth_lock = asyncio.Lock()

    async def open(self) -> None:
        import httpx

        self.client = httpx.AsyncClient(timeout=CALL_SECONDS)

    async def send(self, message: dict) -> dict | None:
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL, **self.config.expand(self.config.headers)}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        url = self.config.expand(self.config.url)
        token = await self._token(url) if self.config.oauth else None
        if token:
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            headers["Authorization"] = f"Bearer {token}"
        response = await self.client.post(url, json=message, headers=headers)
        if response.status_code == 401 and self.config.oauth:
            raise McpError(f"{self.config.name} requires login. Run /mcp login {self.config.name}.")
        if response.status_code >= 400:
            raise McpError(f"{self.config.name} answered HTTP {response.status_code}")
        self.session_id = response.headers.get("mcp-session-id", self.session_id)
        if "id" not in message:
            return None
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict) and data.get("id") == message["id"]:
                        return data
            raise McpError(f"{self.config.name} sent no answer")
        return response.json()

    async def _token(self, url: str) -> str | None:
        from rlmagent.mcp_oauth import keyring_io, refresh

        config = _fill(self.config.oauth)
        key = (url, config)
        async with self._auth_lock:
            if key != self.credential_key:
                self.credential = await asyncio.to_thread(keyring_io, url, config, "load")
                self.credential_key = key
            value = self.credential
            if value is not None and value.expires is not None and time.time() >= value.expires:
                self.credential = value = await refresh(url, config, value)
                await asyncio.to_thread(keyring_io, url, config, "save", value)
            return value.access if value is not None else None

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()


class Server:
    def __init__(self, config: ServerConfig, cwd: str) -> None:
        self.config = config
        self.transport = _Http(config) if config.url else _Stdio(config, cwd)
        self.tools: list[dict] = []
        self.state = "not started"
        self.error = ""
        self._next = 0
        self._lock = asyncio.Lock()

    async def _request(self, method: str, params: dict | None = None,
                       seconds: float = CALL_SECONDS) -> dict:
        self._next += 1
        message = {"jsonrpc": "2.0", "id": self._next, "method": method, "params": params or {}}
        reply = await asyncio.wait_for(self.transport.send(message), seconds)
        if reply is None:
            raise McpError(f"{self.config.name} gave no reply to {method}")
        if reply.get("error"):
            raise McpError(f"{self.config.name}: {reply['error'].get('message', reply['error'])}")
        return reply.get("result") or {}

    async def start(self) -> None:
        async with self._lock:
            if self.state == "ready":
                return
            try:
                await self.transport.open()
                await self._request("initialize", {
                    "protocolVersion": PROTOCOL, "capabilities": {},
                    "clientInfo": {"name": "rlm-agent", "version": "1"}}, START_SECONDS)
                await self.transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
                self.tools = (await self._request("tools/list", seconds=START_SECONDS)).get("tools", [])
                self.state, self.error = "ready", ""
            except Exception as exc:
                self.state, self.error = "failed", str(exc) or type(exc).__name__
                await self.transport.close()
                raise McpError(f"{self.config.name} did not start: {self.error}") from exc

    def tool(self, name: str) -> dict:
        for tool in self.tools:
            if tool.get("name") == name:
                return tool
        raise McpError(f"{self.config.name} has no tool {name!r}; "
                       f"it has {', '.join(t.get('name', '?') for t in self.tools) or 'none'}")

    async def call(self, name: str, arguments: dict) -> str:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(f"[{item.get('type', 'content')} not shown]")
        if not parts and "structuredContent" in result:
            parts.append(json.dumps(result["structuredContent"]))
        text = "\n".join(parts)[:MAX_RESULT_CHARS]
        if result.get("isError"):
            raise McpError(text or f"{name} failed")
        return text

    async def close(self) -> None:
        await self.transport.close()
        self.state = "stopped"


class McpHub:
    """Every configured server, started when first used."""

    def __init__(self, configs: dict[str, ServerConfig], cwd: str, approver=None,
                 problems: list[str] | None = None) -> None:
        self.servers = {name: Server(config, cwd) for name, config in configs.items()}
        self.approver = approver
        self.problems = problems or []

    def _server(self, name: str) -> Server:
        server = self.servers.get(name)
        if server is None:
            known = ", ".join(self.servers) or "none are configured"
            raise McpError(f"no MCP server named {name!r} ({known})")
        return server

    async def mcp_tools(self, server: str | None = None) -> list[dict]:
        """The tools MCP servers offer: [{"server", "tool", "description", "input"}].
        Pass a server name to list only that one."""
        rows = []
        for name in [server] if server else list(self.servers):
            target = self._server(name)
            try:
                await target.start()
            except McpError as exc:
                rows.append({"server": name, "error": str(exc)})
                continue
            rows += [{"server": name, "tool": t.get("name"),
                      "description": (t.get("description") or "")[:300],
                      "input": t.get("inputSchema", {})} for t in target.tools]
        return rows

    async def login(self, name: str) -> str:
        from rlmagent.mcp_oauth import keyring_io, login

        target = self._server(name)
        transport = target.transport
        if not isinstance(transport, _Http) or target.config.oauth is None:
            raise McpError(f"{name} has no HTTP OAuth configuration")
        async with target._lock:
            url = _fill(target.config.url)
            config = _fill(target.config.oauth)
            if config.get("persist"):
                await asyncio.to_thread(keyring_io, url, config, "load")
            credential = await login(url, config)
            await asyncio.to_thread(keyring_io, url, config, "save", credential)
            await target.close()
            transport.session_id = None
            transport.credential = credential
            transport.credential_key = (url, config)
        await target.start()
        return f"Signed in to {name}."

    async def logout(self, name: str) -> str:
        from rlmagent.mcp_oauth import keyring_io

        target = self._server(name)
        transport = target.transport
        if not isinstance(transport, _Http) or target.config.oauth is None:
            raise McpError(f"{name} has no HTTP OAuth configuration")
        async with target._lock:
            url, config = _fill(target.config.url), _fill(target.config.oauth)
            await asyncio.to_thread(keyring_io, url, config, "delete")
            await target.close()
            transport.session_id = None
            transport.credential = None
            transport.credential_key = (url, config)
            target.tools.clear()
        return f"Signed out of {name}."

    async def mcp(self, server: str, tool: str, arguments: dict | None = None) -> str:
        """Call one tool on an MCP server; see mcp_tools() for what exists."""
        from rlmagent.agents.approval import Request, check

        target = self._server(server)
        await target.start()
        spec = target.tool(tool)
        arguments = dict(arguments or {})
        read_only = bool((spec.get("annotations") or {}).get("readOnlyHint"))
        if self.approver is not None and not read_only:
            request = Request("mcp", f"{server}.{tool}", (json.dumps(arguments, sort_keys=True),))
            check(await self.approver.approve(request), request)
        return await target.call(tool, arguments)

    def report(self) -> str:
        if not self.servers and not self.problems:
            return ("No MCP servers. Add them to mcp.json in the rlm-agent home, or to "
                    ".rlm-agent/mcp.json in a trusted project.")
        lines = []
        for name, server in self.servers.items():
            where = server.config.url or server.config.command
            line = f"{name} ({where}): {server.state}"
            if server.state == "ready":
                line += f", {len(server.tools)} tools: " + ", ".join(
                    t.get("name", "?") for t in server.tools)
            if server.error:
                line += f" — {server.error}"
            lines.append(line)
        lines += [f"problem: {p}" for p in self.problems]
        return "\n".join(lines)

    async def close(self) -> None:
        for server in self.servers.values():
            if server.state == "ready":
                await server.close()


__all__ = ["McpError", "McpHub", "ServerConfig", "load_configs"]
