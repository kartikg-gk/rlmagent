"""Split execution: the agent loop and its keys stay here, the tools run in another workspace.

The tool host is a small process next to the files it works on:

    python -m rlmagent.toolhost --root /work            # over stdin/stdout
    python -m rlmagent.toolhost --root /work --tcp 0.0.0.0:7700   # over TCP

Over TCP the first line a client sends must carry the token from RLM_AGENT_TOOLS_TOKEN.
The host only runs what it is asked; approvals, the git guard and every decision about
whether something may run are made by the harness before a call is sent.

Each request is one JSON line: {"id": 1, "fn": "read_file", "args": [...], "kwargs": {...}}.
Each reply is one JSON line: {"id": 1, "ok": true, "value": ...} or {"id": 1, "ok": false,
"error": "..."}.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import sys
from pathlib import Path

TOKEN_ENV = "RLM_AGENT_TOOLS_TOKEN"
MAX_LINE = 64 * 1024 * 1024

# What a tool host serves. Changes and commands are approved by the harness first.
SERVED = ("read_file", "list_files", "search_files", "write_file", "edit_file", "apply_patch",
          "run", "instruction_docs", "job_start", "job_read", "job_wait", "job_stop", "jobs",
          "service_start", "watch")
CHANGES = {"write_file", "edit_file", "apply_patch", "run", "job_start", "service_start"}
# Positional parameter names of the calls the harness checks before sending.
_POSITIONAL = {"run": ("cmd",), "write_file": ("path", "content"),
               "edit_file": ("path", "old", "new"), "apply_patch": ("diff",),
               "job_start": ("command",), "service_start": ("name", "command")}


class ToolHost:
    def __init__(self, root: str) -> None:
        from rlmagent.agents.approval import Approver
        from rlmagent.agents.workspace import Workspace
        from rlmagent.jobs import Jobs

        self.root = str(Path(root).resolve())
        workspace = Workspace(self.root, Approver("auto"))
        self.jobs = Jobs(self.root, Path(self.root) / ".rlm-agent" / "jobs")
        self.functions = {**workspace.functions(), **self.jobs.functions()}

    async def answer(self, request: dict) -> dict:
        number = request.get("id")
        name = request.get("fn")
        if name not in SERVED or name not in self.functions:
            return {"id": number, "ok": False, "error": f"the tool host does not serve {name!r}"}
        try:
            value = await self.functions[name](*request.get("args", []), **request.get("kwargs", {}))
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the caller
            return {"id": number, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"id": number, "ok": True, "value": value}

    async def serve(self, reader, writer, token: str | None = None) -> None:
        if token is not None:
            first = await reader.readline()
            try:
                given = str(json.loads(first).get("token", ""))
            except (json.JSONDecodeError, AttributeError):
                given = ""
            if not hmac.compare_digest(given, token):
                writer.write(b'{"ok": false, "error": "bad token"}\n')
                await writer.drain()
                writer.close()
                return
            writer.write(b'{"ok": true}\n')
            await writer.drain()
        pending = set()
        lock = asyncio.Lock()

        async def handle(line: bytes) -> None:
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                return
            reply = await self.answer(request)
            data = (json.dumps(reply, default=str) + "\n").encode()
            async with lock:
                writer.write(data)
                await writer.drain()

        while line := await reader.readline():
            task = asyncio.create_task(handle(line))
            pending.add(task)
            task.add_done_callback(pending.discard)
        await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        await self.jobs.close()


class RemoteTools:
    """The harness's side: calls tools in a tool host over a stream."""

    def __init__(self, reader, writer, process=None) -> None:
        self.reader, self.writer, self.process = reader, writer, process
        self._next = 0
        self._waiting: dict[int, asyncio.Future] = {}
        self._reading = asyncio.create_task(self._read())

    @classmethod
    async def over_stdio(cls, command: list[str]) -> RemoteTools:
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=MAX_LINE)
        return cls(process.stdout, process.stdin, process)

    @classmethod
    async def over_tcp(cls, host: str, port: int, token: str) -> RemoteTools:
        reader, writer = await asyncio.open_connection(host, port, limit=MAX_LINE)
        writer.write((json.dumps({"token": token}) + "\n").encode())
        await writer.drain()
        hello = json.loads(await reader.readline() or b"{}")
        if not hello.get("ok"):
            writer.close()
            raise ConnectionError(f"the tool host refused the connection: {hello.get('error')}")
        return cls(reader, writer)

    async def _read(self) -> None:
        try:
            while line := await self.reader.readline():
                reply = json.loads(line)
                waiter = self._waiting.pop(reply.get("id"), None)
                if waiter is not None and not waiter.done():
                    waiter.set_result(reply)
        finally:
            for waiter in self._waiting.values():
                if not waiter.done():
                    waiter.set_exception(ConnectionError("the tool host went away"))

    async def call(self, name: str, *args, **kwargs):
        self._next += 1
        number = self._next
        waiter = asyncio.get_running_loop().create_future()
        self._waiting[number] = waiter
        line = json.dumps({"id": number, "fn": name, "args": list(args), "kwargs": kwargs})
        self.writer.write((line + "\n").encode())
        await self.writer.drain()
        reply = await waiter
        if not reply.get("ok"):
            raise ValueError(reply.get("error", "the tool host failed"))
        return reply.get("value")

    def functions(self, approver, policy=None) -> dict:
        """The served tools as cell functions, approved here before they are sent."""
        import difflib

        from rlmagent.agents.approval import Request, check
        from rlmagent.policy import git_refusal

        async def approve(request: Request) -> None:
            check(await approver.approve(request), Request(request.kind, request.target,
                                                           request.payload))

        def make(name):
            async def call(*args, **kwargs):
                given = {**dict(zip(_POSITIONAL.get(name, ()), args)), **kwargs}
                if name in ("run", "job_start", "service_start"):
                    command = str(given.get("cmd", given.get("command", "")))
                    if policy is not None and (reason := git_refusal(command, policy)):
                        raise ValueError(reason)
                    await approve(Request("run", command))
                elif name == "write_file":
                    path, content = str(given.get("path")), str(given.get("content", ""))
                    try:
                        before = await self.call("read_file", path)
                    except ValueError:
                        before = ""
                    preview = "\n".join(difflib.unified_diff(
                        before.splitlines(), content.splitlines(), f"a/{path}", f"b/{path}",
                        lineterm=""))
                    await approve(Request("write", path, (content,), preview=preview))
                elif name == "edit_file":
                    await approve(Request("edit", str(given.get("path")),
                                          (given.get("old"), given.get("new"))))
                elif name == "apply_patch":
                    diff = str(given.get("diff", ""))
                    await approve(Request("patch", "remote workspace", (diff,), preview=diff))
                return await self.call(name, *args, **kwargs)

            call.__name__ = name
            return call

        return {name: make(name) for name in SERVED}

    async def close(self) -> None:
        self.writer.close()
        self._reading.cancel()
        if self.process is not None and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()


async def connect(spec: str) -> RemoteTools:
    """`tcp://host:port` (token from RLM_AGENT_TOOLS_TOKEN), or a command run over stdio."""
    import shlex

    if spec.startswith("tcp://"):
        host, _, port = spec[len("tcp://"):].rpartition(":")
        token = os.environ.get(TOKEN_ENV)
        if not token:
            raise ValueError(f"set {TOKEN_ENV} to the tool host's token")
        return await RemoteTools.over_tcp(host or "127.0.0.1", int(port), token)
    words = shlex.split(spec, posix=os.name != "nt")
    if os.name == "nt":
        # Windows splitting keeps the quotes that protect paths with spaces.
        words = [w[1:-1] if len(w) > 1 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]
    return await RemoteTools.over_stdio(words)


async def _stdio(host: ToolHost) -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=MAX_LINE)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout)
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    await host.serve(reader, writer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rlmagent.toolhost",
                                     description="Serve workspace tools to an rlm-agent harness.")
    parser.add_argument("--root", default=".", help="The workspace the tools work in.")
    parser.add_argument("--tcp", default=None, metavar="HOST:PORT",
                        help=f"Listen on TCP; clients must send the token in {TOKEN_ENV}.")
    ns = parser.parse_args(argv)
    host = ToolHost(ns.root)

    async def run() -> None:
        try:
            if ns.tcp is None:
                if os.name == "nt":
                    # Pipes on Windows do not work with the selector loop's pipe readers.
                    await _stdio_threads(host)
                else:
                    await _stdio(host)
                return
            token = os.environ.get(TOKEN_ENV)
            if not token:
                raise SystemExit(f"set {TOKEN_ENV} before listening on TCP")
            address, _, port = ns.tcp.rpartition(":")
            server = await asyncio.start_server(
                lambda r, w: host.serve(r, w, token), address or "127.0.0.1", int(port),
                limit=MAX_LINE)
            print(f"tool host listening on {ns.tcp} for {host.root}", file=sys.stderr, flush=True)
            async with server:
                await server.serve_forever()
        finally:
            await host.close()

    asyncio.run(run())
    return 0


async def _stdio_threads(host: ToolHost) -> None:
    """stdin/stdout through threads, for platforms whose event loop cannot watch pipes."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=MAX_LINE)

    def pump():
        stream = sys.stdin.buffer
        while line := stream.readline():
            loop.call_soon_threadsafe(reader.feed_data, line)
        loop.call_soon_threadsafe(reader.feed_eof)

    loop.run_in_executor(None, pump)

    class Writer:
        def write(self, data: bytes) -> None:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

    await host.serve(reader, Writer())


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CHANGES", "SERVED", "TOKEN_ENV", "RemoteTools", "ToolHost", "connect", "main"]
