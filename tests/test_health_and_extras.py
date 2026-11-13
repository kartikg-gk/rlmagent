import os

import httpx
import pytest

from rlmagent import worker
from rlmagent.agentnet import AgentNet
from rlmagent.cli import health
from rlmagent.websearch import make_web_search


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    for name in ("BRAVE_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "home"


def test_doctor_finds_and_fixes_stale_workers_and_sessions(tmp_path, capsys):
    worker._write(worker.folder("old") / "meta.json",
                  {"pid": 999999, "state": "busy", "beat": 0, "session": "x"})
    gone = AgentNet("s-gone", str(tmp_path))
    gone.register()
    meta = gone._folder("s-gone") / "meta.json"
    meta.write_text(meta.read_text().replace(f'"pid": {os.getpid()}', '"pid": 999999'))
    names = {c.name: c for c in health.checks()}
    assert not names["worker old"].ok and names["worker old"].fix == "mark it stopped"
    assert not names["session s-gone"].ok
    assert health.run([], doctor=True) == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "doctor --fix would mark it stopped" in out
    health.run(["--fix"], doctor=True)
    assert "fixing: marked stopped" in capsys.readouterr().out
    assert worker.state_of("old")["state"] == "stopped" and not meta.exists()
    assert health.run([], doctor=False) == 0


async def test_web_search_needs_a_key_and_the_network(monkeypatch):
    search = make_web_search(lambda: False)
    with pytest.raises(ValueError, match="network access is off"):
        await search("x")
    search = make_web_search(lambda: True)
    with pytest.raises(ValueError, match="no search key"):
        await search("x")

    def reply(request):
        assert request.headers["X-API-KEY"] == "k"
        return httpx.Response(200, json={"organic": [
            {"title": "T", "link": "https://a.example", "snippet": "S"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setenv("SERPER_API_KEY", "k")
    assert await search("x") == [{"title": "T", "url": "https://a.example", "snippet": "S"}]


async def test_btw_answers_without_adding_to_the_conversation(tmp_path):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([text_turn("It is about tests.")]), provider_name="replay",
        model="m", system="s", tools=[], cwd=str(tmp_path))
    before = len(session.transcript)
    assert (await session.handle_command("/btw")).startswith("Usage: /btw QUESTION")
    assert await session.handle_command("/btw what is this about?") == "It is about tests."
    assert len(session.transcript) == before
    await session.shutdown()


async def test_a_requested_compaction_runs_when_the_turn_ends(tmp_path, monkeypatch):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([text_turn("ok")] * 3), provider_name="replay", model="m",
        system="s", tools=[], cwd=str(tmp_path))
    ran = []

    async def compact():
        ran.append(True)

    monkeypatch.setattr(session, "compact", compact)
    import dataclasses

    from rlmagent import conversation

    monkeypatch.setattr(conversation, "_CONTEXT_LIMITS",
                        dataclasses.replace(conversation._CONTEXT_LIMITS, min_entries=1))
    async for _ in session.submit("one"):
        pass
    assert ran == []
    session.compact_requested = True
    async for _ in session.submit("two"):
        pass
    assert ran == [True] and session.compact_requested is False
    await session.shutdown()


def test_config_show_lists_environment_overrides(tmp_path, monkeypatch, capsys):
    from rlmagent.cli.config_cmd import handle_config

    monkeypatch.setenv("RLM_AGENT_MODEL", "small-model")
    monkeypatch.setenv("RLM_AGENT_API_KEY", "sk-test-1234567890")
    monkeypatch.delenv("RLM_AGENT_PROVIDER", raising=False)
    assert handle_config(["show"]) == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "RLM_AGENT_MODEL=small-model" in out
    assert "sk-test-1234567890" not in out and "RLM_AGENT_API_KEY=" in out
