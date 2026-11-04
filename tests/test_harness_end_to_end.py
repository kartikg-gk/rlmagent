"""The default engine, run from the command line against a local model server."""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import rlmagent
from rlmagent.kernel.wasm import available


def _server(replies):
    class Model(http.server.BaseHTTPRequestHandler):
        log_message = lambda *_: None  # noqa: E731

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            chunks = [
                {"id": "x", "model": "m", "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": replies.pop(0)}}]},
                {"id": "x", "model": "m", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


CELL = """```python
info = await capabilities()
FINAL([await word_count('a b c'), 'word_count' in info['functions'],
       'read_file' in info['functions'], 'made_up' in info['functions'], info['approval']])
```"""


@pytest.mark.parametrize("kernel", ["local", "wasm"])
def test_a_project_extension_and_capabilities_reach_a_real_cell(tmp_path, kernel):
    if kernel == "wasm" and not available():
        pytest.skip("needs the sandbox")
    server = _server([CELL])
    work = tmp_path / "work"
    (work / ".rlm-agent" / "extensions").mkdir(parents=True)
    (work / ".rlm-agent" / "extensions" / "words.py").write_text(
        "def word_count(text):\n    return len(text.split())\n"
        "def register():\n    return {'word_count': word_count}\n",
        encoding="utf-8",
    )
    env = {**os.environ, "RLM_AGENT_HOME": str(tmp_path / "home"),
           "PYTHONPATH": str(Path(rlmagent.__file__).parents[1]),
           "OPENAI_API_KEY": "test-not-real",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}/v1"}
    try:
        done = subprocess.run(
            [sys.executable, "-m", "rlmagent.cli.main", "--provider", "openai", "--model", "m",
             "--kernel", kernel, "--approve", "--print", "go"],
            capture_output=True, text=True, env=env, cwd=work, timeout=180,
        )
    finally:
        server.shutdown()
    assert done.returncode == 0, done.stderr
    assert "[3, True, True, False, 'deny']" in done.stdout


def test_an_untrusted_folder_gets_no_project_extension(tmp_path):
    server = _server(["```python\nFINAL('word_count' in (await capabilities())['functions'])\n```"])
    work = tmp_path / "work"
    (work / ".rlm-agent" / "extensions").mkdir(parents=True)
    (work / ".rlm-agent" / "extensions" / "words.py").write_text(
        "def register():\n    return {'word_count': len}\n", encoding="utf-8")
    env = {**os.environ, "RLM_AGENT_HOME": str(tmp_path / "home"),
           "PYTHONPATH": str(Path(rlmagent.__file__).parents[1]),
           "OPENAI_API_KEY": "test-not-real",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_address[1]}/v1"}
    try:
        done = subprocess.run(
            [sys.executable, "-m", "rlmagent.cli.main", "--provider", "openai", "--model", "m",
             "--kernel", "local", "--no-approve", "--print", "go"],
            capture_output=True, text=True, env=env, cwd=work, timeout=180,
        )
    finally:
        server.shutdown()
    assert done.returncode == 0, done.stderr
    assert "False" in done.stdout
