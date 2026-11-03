import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rlmagent.agents.approval import Approver, Refused
from rlmagent.mcp import McpError, McpHub, ServerConfig, load_configs

SERVER = str(Path(__file__).with_name("_mcp_server.py"))


def _stdio_hub(tmp_path, approver=None):
    config = ServerConfig("t", command=sys.executable, args=[SERVER],
                          env={"TEST_TOKEN": "${MCP_TEST_SECRET}"})
    return McpHub({"t": config}, str(tmp_path), approver)


def test_configs_come_from_home_and_project(tmp_path):
    home, project = tmp_path / "home", tmp_path / "proj"
    (home).mkdir()
    (project / ".rlm-agent").mkdir(parents=True)
    (home / "mcp.json").write_text(json.dumps({"servers": {
        "a": {"command": "x"}, "b": {"url": "http://h/mcp"}}}))
    (project / ".rlm-agent" / "mcp.json").write_text(json.dumps({"servers": {
        "a": {"url": "http://p/mcp"}, "bad": {}}}))
    servers, problems = load_configs(home, project)
    assert servers["a"].url == "http://p/mcp" and servers["b"].url == "http://h/mcp"
    assert len(problems) == 1 and "needs either command or url" in problems[0]
    servers, _ = load_configs(home, None)
    assert servers["a"].command == "x"


async def test_stdio_server_lists_and_calls_tools_with_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TEST_SECRET", "s3")
    asked = []

    async def ask(request):
        asked.append(request.target)
        return "deny"

    hub = _stdio_hub(tmp_path, Approver("ask", ask))
    try:
        rows = await hub.mcp_tools()
        assert [r["tool"] for r in rows] == ["echo", "store", "fail"]
        assert await hub.mcp("t", "echo", {"x": 1}) == 'echo: {"x": 1} token=s3'
        assert asked == []  # read-only tools need no approval
        with pytest.raises(Refused):
            await hub.mcp("t", "store", {"v": 2})
        assert asked == ["t.store"]
        hub.approver = Approver("auto")
        assert (await hub.mcp("t", "store", {"v": 2})).startswith("store:")
        with pytest.raises(McpError, match="it broke"):
            await hub.mcp("t", "fail")
        with pytest.raises(McpError, match="has no tool 'nope'"):
            await hub.mcp("t", "nope")
        with pytest.raises(McpError, match="no MCP server named 'z'"):
            await hub.mcp("z", "echo")
        assert "t (" in hub.report() and "3 tools: echo, store, fail" in hub.report()
    finally:
        await hub.close()


async def test_a_server_that_does_not_start_is_reported(tmp_path):
    hub = McpHub({"gone": ServerConfig("gone", command="no-such-mcp-binary-xyz")}, str(tmp_path))
    rows = await hub.mcp_tools()
    assert rows[0]["server"] == "gone" and "did not start" in rows[0]["error"]
    assert "gone (no-such-mcp-binary-xyz): failed" in hub.report()


async def test_http_server_with_event_stream_replies(tmp_path):
    from _mcp_server import answer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = answer(message)
            if reply is None:
                self.send_response(202)
                self.end_headers()
                return
            body = f"event: message\ndata: {json.dumps(reply)}\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Mcp-Session-Id", "abc")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    hub = McpHub({"h": ServerConfig("h", url=f"http://127.0.0.1:{server.server_port}/mcp")},
                 str(tmp_path), Approver("auto"))
    try:
        assert await hub.mcp("h", "store", {"k": "v"}) == 'store: {"k": "v"} token='
        assert hub.servers["h"].transport.session_id == "abc"
    finally:
        await hub.close()
        server.shutdown()


async def test_mcp_command_reports_servers(tmp_path):
    from _scripted import text_turn  # noqa: F401

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([]), provider_name="replay", model="m", system="s",
        tools=[], cwd=str(tmp_path))
    assert await session._cmd_mcp("") == "MCP is not available in this session."
    session.mcp = _stdio_hub(tmp_path)
    assert "t (" in await session._cmd_mcp("") and "not started" in await session._cmd_mcp("")
    assert "3 tools" in await session._cmd_mcp("t")
    assert await session._cmd_mcp("zz") == "No MCP server named 'zz'."
    await session.shutdown()
    assert session.mcp.servers["t"].state == "stopped"
