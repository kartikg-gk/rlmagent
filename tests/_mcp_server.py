"""A tiny MCP server for tests: stdio when run as a script, or HTTP through `answer`."""

import json
import os
import sys

TOOLS = [
    {"name": "echo", "description": "Say it back", "inputSchema": {"type": "object"},
     "annotations": {"readOnlyHint": True}},
    {"name": "store", "description": "Keep a value", "inputSchema": {"type": "object"}},
    {"name": "fail", "description": "Always fails", "inputSchema": {"type": "object"}},
]


def answer(message):
    method = message.get("method")
    if "id" not in message:
        return None
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "test", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name, args = message["params"]["name"], message["params"].get("arguments", {})
        if name == "fail":
            result = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
        else:
            text = f"{name}: {json.dumps(args, sort_keys=True)} token={os.environ.get('TEST_TOKEN', '')}"
            result = {"content": [{"type": "text", "text": text}]}
    else:
        return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "no"}}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


if __name__ == "__main__":
    for line in sys.stdin:
        reply = answer(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
