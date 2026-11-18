"""/sandbox shows where code runs and saves the default for new sessions."""

from types import SimpleNamespace

from rlmagent.config.store import load_config
from rlmagent.conversation import CodingSession


def _session(kind):
    s = CodingSession.__new__(CodingSession)
    s.agent_tree = SimpleNamespace(kernel_kind=kind)
    return s


def test_sandbox_command_reports_and_saves(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    monkeypatch.delenv("RLM_AGENT_CONFIG_DIR", raising=False)
    s = _session("local")
    assert "Sandbox is off" in s._cmd_sandbox("")
    out = s._cmd_sandbox("on")
    assert load_config().kernel == "wasm" and "restart rlm-agent" in out
    assert "new sessions start with it on" in s._cmd_sandbox("")
    assert "saved as the default" in _session("wasm")._cmd_sandbox("on")
    s._cmd_sandbox("off")
    assert load_config().kernel == "local"
    assert "Unknown argument" in s._cmd_sandbox("maybe")


def test_sandbox_command_is_offered():
    assert "sandbox" in CodingSession.RUNNABLE_COMMANDS
