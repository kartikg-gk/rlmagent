import json
import os
import sys

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
    assert parse_model("cli:codex?model=gpt-5.5") == ("codex", "gpt-5.5")
    assert parse_model("cli:claude-code") == ("claude-code", None)
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
