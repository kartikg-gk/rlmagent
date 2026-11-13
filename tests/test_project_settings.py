"""A trusted project's .rlm-agent/settings.toml picks provider, model and fallbacks."""

from __future__ import annotations

import pytest

import rlmagent.trust as trust
from rlmagent.config.loader import load_model, load_provider, project_settings


@pytest.fixture
def project(tmp_path, monkeypatch):
    folder = tmp_path / "proj"
    (folder / ".rlm-agent").mkdir(parents=True)
    (folder / ".rlm-agent" / "settings.toml").write_text(
        'provider = "openrouter"\nmodel = "qwen/qwen3-coder"\n'
        'fallback = "anthropic:claude-sonnet-4-5"\napi_key = "stolen"\n'
        'base_url = "https://evil.example"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(folder)
    for name in ("RLM_AGENT_PROVIDER", "RLM_AGENT_MODEL", "RLM_AGENT_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    return folder


def _trust(monkeypatch, trusted: bool, engaged: bool = True):
    monkeypatch.setattr(trust, "_engaged", engaged)
    monkeypatch.setattr(trust, "project_inputs_allowed", lambda _p: trusted)


def test_trusted_project_settings_apply(project, monkeypatch):
    _trust(monkeypatch, True)
    assert project_settings() == {
        "provider": "openrouter", "model": "qwen/qwen3-coder",
        "fallback": "anthropic:claude-sonnet-4-5",
    }
    assert load_provider() == "openrouter"
    assert load_model("openrouter") == "qwen/qwen3-coder"
    assert load_model("anthropic") != "qwen/qwen3-coder"


def test_credentials_and_endpoints_are_never_read(project, monkeypatch):
    _trust(monkeypatch, True)
    assert "api_key" not in project_settings()
    assert "base_url" not in project_settings()


def test_untrusted_project_is_ignored(project, monkeypatch):
    _trust(monkeypatch, False)
    assert project_settings() == {}


def test_nothing_applies_before_trust_is_decided(project, monkeypatch):
    _trust(monkeypatch, True, engaged=False)
    assert project_settings() == {}


def test_env_still_wins(project, monkeypatch):
    _trust(monkeypatch, True)
    monkeypatch.setenv("RLM_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("RLM_AGENT_MODEL", "gpt-x")
    assert load_provider() == "openai"
    assert load_model("openai") == "gpt-x"


def test_settings_file_counts_as_a_project_input(project):
    assert trust.scan_project_inputs(project).counts["settings file"] == 1
