import json
import os
import sys
from types import SimpleNamespace

import pytest
from _scripted import code_turn, text_turn
from test_improve import _setup, _submit

from rlmagent.harness.contracts.transcript import HumanEntry
from rlmagent.model.external import ExternalAgentProvider, parse_model
from rlmagent.model.external.cli import parse_claude, parse_codex
from rlmagent.model.external.textcalls import parse_reply, render

CLAUDE_OK = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "hi",
                        "total_cost_usd": 0.002, "usage": {"input_tokens": 600, "output_tokens": 5,
                        "cache_read_input_tokens": 10, "cache_creation_input_tokens": 0}})
CODEX_OK = "\n".join(json.dumps(e) for e in [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "hi"}},
    {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 5}},
])

FAKE = r'''
import json, os, sys
from pathlib import Path
log = Path(os.environ["FAKE_AGENT_LOG"])
calls = json.loads(log.read_text()) if log.exists() else []
prompt = sys.stdin.read()
calls.append({"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": prompt})
log.write_text(json.dumps(calls))
replies = json.loads(os.environ["FAKE_AGENT_REPLIES"])
text = replies[min(len(calls), len(replies)) - 1]
if os.path.basename(sys.argv[0]).startswith("codex"):
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}}))
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 5}}))
else:
    print(json.dumps({"subtype": "success", "is_error": False, "result": text,
                      "total_cost_usd": 0.001, "usage": {"input_tokens": 50, "output_tokens": 5}}))
'''


@pytest.fixture
def fake_agents(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("claude", "codex"):
        script = bin_dir / f"{name}_fake.py"
        script.write_text(FAKE)
        if os.name == "nt":
            (bin_dir / f"{name}.bat").write_text(f'@"{sys.executable}" "{script}" %*\n')
        else:
            launcher = bin_dir / name
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
            launcher.chmod(0o755)
    log = tmp_path / "calls.json"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_AGENT_LOG", str(log))

    def replies(*texts):
        monkeypatch.setenv("FAKE_AGENT_REPLIES", json.dumps(list(texts)))
        return lambda: json.loads(log.read_text()) if log.exists() else []
    return replies


def test_a_reply_splits_into_text_and_the_first_python_block():
    assert parse_reply("Let me look.\n```python\nprint(1)\n```\n```python\nprint(2)\n```") == (
        "Let me look.", "print(1)\n")
    assert parse_reply("The answer is 4.") == ("The answer is 4.", None)
    assert parse_reply("```repl\nx = 1\n```")[1] == "x = 1\n"


def test_the_conversation_renders_as_one_prompt():
    text = render([HumanEntry(content="fix calc.py")])
    assert text == "[user]\nfix calc.py"


def test_cli_output_parses_with_usage_and_cost():
    text, usage = parse_claude(CLAUDE_OK)
    assert text == "hi" and usage.input == 600 and usage.cost.total == 0.002 and usage.total_tokens == 615
    text, usage = parse_codex(CODEX_OK)
    assert text == "hi" and usage.input == 60 and usage.cache_read == 40 and usage.output == 5
    with pytest.raises(ValueError, match="claude failed"):
        parse_claude(json.dumps({"subtype": "error_max_turns", "is_error": True}))
    with pytest.raises(ValueError, match="codex failed"):
        parse_codex(json.dumps({"type": "turn.failed", "error": {"message": "quota"}}))


def test_model_names():
    assert parse_model("cli:codex?model=gpt-5.5") == ("cli", "codex", "gpt-5.5")
    assert parse_model("acp:claude-code") == ("acp", "claude-code", None)
    with pytest.raises(ValueError, match="unknown agent"):
        parse_model("cli:cursor")


async def test_an_external_agent_runs_code_through_our_kernel(tmp_path, fake_agents):
    calls = fake_agents("Checking.\n```python\nprint(6 * 7)\n```", "It is 42.")
    session, _, _ = await _setup(tmp_path, [])
    session.agent_tree.use_provider(ExternalAgentProvider(), "cli")
    await session.switch_model("cli:claude-code?model=sonnet")
    try:
        await _submit(session, "what is 6*7?")
        made = calls()
        assert len(made) == 2
        assert "42" in made[1]["stdin"] and "[python output]" in made[1]["stdin"]
        assert made[0]["argv"][made[0]["argv"].index("--model") + 1] == "sonnet"
        assert "--tools" in made[0]["argv"] and made[0]["cwd"] != str(tmp_path)
        assert session.transcript[-1].text == "It is 42."
        assert session.usage.total_cost_usd == pytest.approx(0.002)
    finally:
        await session.shutdown()


async def test_a_sub_agent_can_run_on_codex_while_the_root_does_not(tmp_path, fake_agents):
    calls = fake_agents("```python\nFINAL('from codex')\n```", "Done.")
    session, provider, _ = await _setup(tmp_path, [
        code_turn("print(await rlm('say hi', model='cli:codex'))"), text_turn("done"),
    ])
    try:
        await _submit(session, "delegate")
        made = calls()
        assert len(made) == 2 and made[0]["argv"][0] == "exec"
        assert "say hi" in made[0]["stdin"] and "## Running code" in made[0]["stdin"]
        outputs = [m.text for m in session.transcript if getattr(m, "role", "") == "toolResult"]
        assert "from codex" in outputs[0]
    finally:
        await session.shutdown()


async def test_an_unknown_agent_is_refused_at_spawn(tmp_path):
    session, _, _ = await _setup(tmp_path, [])
    try:
        with pytest.raises(ValueError, match="unknown agent"):
            session.agent_tree.spawn("root", "t", None, model="cli:nope")
    finally:
        await session.shutdown()


FAKE_ACP = r"""
import json, os, sys
log = []
def send(m):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", **m}) + "\n"); sys.stdout.flush()
def read():
    return json.loads(sys.stdin.readline())
for line in sys.stdin:
    msg = json.loads(line)
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params", {})
    log.append({"method": method, "params": params})
    if method == "initialize":
        send({"id": rid, "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif method == "session/new":
        send({"id": rid, "result": {"sessionId": "s1", "modes": {"currentModeId": "default",
              "availableModes": [{"id": "default"}, {"id": "plan"}]}}})
    elif method == "session/set_mode":
        send({"id": rid, "result": {}})
    elif method == "session/prompt":
        send({"id": 900, "method": "session/request_permission", "params": {"sessionId": "s1",
              "options": [{"optionId": "yes", "kind": "allow_once"}, {"optionId": "no", "kind": "reject_once"}]}})
        log.append({"permission_reply": read()})
        send({"id": 901, "method": "fs/read_text_file", "params": {"path": "x"}})
        log.append({"fs_reply": read()})
        for chunk in ["Running it.\n```python\n", "print(6 * 7)\n```"]:
            send({"method": "session/update", "params": {"sessionId": "s1", "update": {
                  "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": chunk}}}})
        with open(os.environ["FAKE_ACP_LOG"], "w") as f:
            json.dump(log, f)
        send({"id": rid, "result": {"stopReason": "end_turn",
              "usage": {"inputTokens": 30, "outputTokens": 7, "totalTokens": 37}}})
"""


async def test_an_acp_agent_answers_through_the_protocol(tmp_path, monkeypatch):
    from rlmagent.model.external import ExternalAgentProvider

    script = tmp_path / "fake_acp.py"
    script.write_text(FAKE_ACP)
    home = tmp_path / "home"
    home.mkdir()
    (home / "acp_agents.json").write_text(json.dumps({"fake": {
        "command": sys.executable, "args": [str(script)], "readonly_mode": "plan"}}))
    monkeypatch.setenv("RLM_AGENT_HOME", str(home))
    monkeypatch.setenv("FAKE_ACP_LOG", str(tmp_path / "acp.json"))

    events = [e async for e in ExternalAgentProvider().stream_response(
        model="acp:fake", system="SYS", messages=[HumanEntry(content="what is 6*7?")],
        tools=[SimpleNamespace(name="python")])]
    message = events[-1].message
    assert events[-1].reason == "toolUse"
    assert message.content[0].text == "Running it."
    assert message.content[1].arguments["code"] == "print(6 * 7)\n"
    assert message.usage.total_tokens == 0          # no usage over ACP; --max-calls is the budget
    log = json.loads((tmp_path / "acp.json").read_text())
    prompt = next(e for e in log if e.get("method") == "session/prompt")
    assert "SYS" in prompt["params"]["prompt"][0]["text"] and "what is 6*7?" in prompt["params"]["prompt"][0]["text"]
    assert {"method": "session/set_mode", "params": {"sessionId": "s1", "modeId": "plan"}} in log
    permission = next(e["permission_reply"] for e in log if "permission_reply" in e)
    assert permission["result"]["outcome"] == {"outcome": "selected", "optionId": "no"}
    assert next(e["fs_reply"] for e in log if "fs_reply" in e)["error"]["code"] == -32601


async def test_an_agent_marked_approve_gets_its_permission_requests_allowed(tmp_path, monkeypatch):
    from rlmagent.model.external import ExternalAgentProvider

    script = tmp_path / "fake_acp.py"
    script.write_text(FAKE_ACP)
    home = tmp_path / "home"
    home.mkdir()
    (home / "acp_agents.json").write_text(json.dumps({"fake": {
        "command": sys.executable, "args": [str(script)], "approve": True}}))
    monkeypatch.setenv("RLM_AGENT_HOME", str(home))
    monkeypatch.setenv("FAKE_ACP_LOG", str(tmp_path / "acp.json"))
    [e async for e in ExternalAgentProvider().stream_response(
        model="acp:fake", system="s", messages=[HumanEntry(content="hi")], tools=[])]
    log = json.loads((tmp_path / "acp.json").read_text())
    permission = next(e["permission_reply"] for e in log if "permission_reply" in e)
    assert permission["result"]["outcome"] == {"outcome": "selected", "optionId": "yes"}


async def test_an_acp_agent_that_exits_is_an_error_not_a_hang(tmp_path, monkeypatch):
    from rlmagent.model.external import ExternalAgentProvider

    home = tmp_path / "home"
    home.mkdir()
    (home / "acp_agents.json").write_text(json.dumps({"dead": {
        "command": sys.executable, "args": ["-c", "pass"]}}))
    monkeypatch.setenv("RLM_AGENT_HOME", str(home))
    events = [e async for e in ExternalAgentProvider().stream_response(
        model="acp:dead", system="s", messages=[HumanEntry(content="hi")], tools=[])]
    assert events[-1].type == "error" and "exited" in events[-1].error.error_message


def test_a_bridge_preset_needs_acp_install_first(tmp_path, monkeypatch):
    from rlmagent.model.external.acp_install import bridge_spec, marker_path

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="rlm-agent acp install"):
        bridge_spec("codex")
    marker_path().write_text(json.dumps({"marker_version": 1, "bridges": {"codex": "2.2.2"},
                                         "bridge_packages": {"codex": "@acme/codex-bridge"}}))
    assert bridge_spec("codex") == "@acme/codex-bridge@2.2.2"
    assert bridge_spec("claude-code") == "@agentclientprotocol/claude-agent-acp"


def test_install_keeps_recorded_versions_unless_asked_to_update(tmp_path, monkeypatch):
    from rlmagent.model.external import acp_install

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    calls = []

    def fake_npm(*args):
        calls.append(args)
        return "9.9.9" if args[0] == "view" else "ok"
    monkeypatch.setattr(acp_install, "_npm", fake_npm)
    assert acp_install.install()["bridges"] == {"claude-code": "9.9.9", "codex": "9.9.9"}
    marker = json.loads(acp_install.marker_path().read_text())
    marker["bridges"]["codex"] = "1.0.0"
    acp_install.marker_path().write_text(json.dumps(marker))
    assert acp_install.install()["bridges"]["codex"] == "1.0.0"
    assert acp_install.install(update=True)["bridges"]["codex"] == "9.9.9"
    assert ("cache", "add", "@agentclientprotocol/codex-acp@9.9.9") in calls


def test_acp_runs_default_to_fifty_calls():
    from rlmagent.cli.main import _build_run_parser

    ns = _build_run_parser().parse_args([])
    assert ns.max_calls is None
    import inspect
    from rlmagent import runtime
    assert '50 if str(model).startswith("acp:")' in inspect.getsource(runtime.make_agent_tree)


async def test_fake_tool_call_markup_gets_one_correction(tmp_path, fake_agents):
    calls = fake_agents('<invoke name="bash"><parameter name="command">ls</parameter></invoke>',
                        "Running.\n```python\nprint(1)\n```")
    from types import SimpleNamespace

    events = [e async for e in ExternalAgentProvider().stream_response(
        model="cli:claude-code", system="s", messages=[HumanEntry(content="list files")],
        tools=[SimpleNamespace(name="python")])]
    made = calls()
    assert len(made) == 2 and "does nothing here" in made[1]["stdin"]
    assert events[-1].reason == "toolUse" and events[-1].message.usage.cost.total == pytest.approx(0.002)
