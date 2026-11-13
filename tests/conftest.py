"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rlmagent.trust import reset_trust_cache


@pytest.fixture(autouse=True)
def _fresh_trust_state() -> Iterator[None]:
    """Project-trust resolutions are per process; keep them per test."""
    reset_trust_cache()
    yield
    reset_trust_cache()


@pytest.fixture(autouse=True)
def _private_app_home(tmp_path_factory, monkeypatch) -> None:
    """No test reads or writes the developer's real configuration or sessions."""
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path_factory.mktemp("app-home")))
    for name in ("RLM_AGENT_CONFIG_DIR", "RLM_AGENT_PROVIDER", "RLM_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
