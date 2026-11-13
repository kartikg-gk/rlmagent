"""Queued strip, folded errors, shell box, command echo, goal readout, pickers, Ctrl+C."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlmagent.tui.app import RlmAgentApp
from rlmagent.tui.choice_screen import ChoiceScreen
from rlmagent.tui.messages import SHELL_TAIL, ErrorBlock, QueuedStrip, ShellBlock
from rlmagent.tui.status import StatusBar
from rlmagent.tui.transcript import TranscriptView


def _snapshot(**overrides):
    base = dict(provider="p", model="m", context_percent=3, input_tokens=0, output_tokens=0,
                cached_tokens=0, cost=0.0, budget="", branch="", plan_mode=False)
    return SimpleNamespace(**{**base, **overrides})


def test_shell_block_keeps_only_the_last_lines():
    output = "\n".join(f"line {i}" for i in range(50))
    block = ShellBlock("seq 50", output, exit_code=0)
    assert block.hidden_lines == 50 - SHELL_TAIL
    assert block.border_title == "$ seq 50"
    assert not block.has_class("shell-failed")
    assert ShellBlock("false", "", exit_code=1).has_class("shell-failed")


def test_error_block_headline_is_the_first_line():
    assert ErrorBlock("boom\ntrace 1\ntrace 2").headline == "boom"


def test_status_shows_the_active_goal():
    bar = StatusBar()
    bar.show(_snapshot(), goal="make every test pass in the repo and push the branch")
    assert "◎ make every test pass" in bar.line and "…" in bar.line
    bar.show(_snapshot())
    assert "◎" not in bar.line


@pytest.mark.asyncio
async def test_queue_strip_and_folded_error_render():
    app = RlmAgentApp()
    async with app.run_test() as pilot:
        strip = app.query_one(QueuedStrip)
        assert strip.display is False
        app._queued[:] = ["first", "second\nmore"]
        app._show_queue()
        await pilot.pause()
        assert strip.display is True
        transcript = app.query_one(TranscriptView)
        transcript.add_notice("bad thing\nline two", error=True)
        transcript.add_command("/shell ls -la")
        transcript.add_shell("ls", "a\nb", 0)
        await pilot.pause()
        assert transcript.query(ErrorBlock)
        assert transcript.query(ShellBlock)


@pytest.mark.asyncio
async def test_choice_screen_returns_the_pick():
    picked = []
    app = RlmAgentApp()
    async with app.run_test() as pilot:
        app.push_screen(ChoiceScreen("Effort", ["off", "low", "high"], "low"), picked.append)
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
    assert picked == ["high"]


def test_second_ctrl_c_quits_at_once():
    app = RlmAgentApp()
    exits, stops = [], []
    app.exit = lambda *a, **k: exits.append(1)
    app.action_stop = lambda: stops.append(1) or True  # a run is going
    app.notify = lambda *a, **k: None
    app.action_cancel()
    assert stops and not exits  # the first press only stops the run
    app.action_cancel()
    assert exits  # the second, right after, quits even though a run was going


@pytest.mark.asyncio
async def test_btw_follow_ups_see_earlier_side_questions(tmp_path):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([text_turn("x")]), provider_name="replay",
        model="m", system="s", tools=[], cwd=str(tmp_path))
    prompts = []

    async def answer(prompt, _system):
        prompts.append(prompt)
        return f"answer {len(prompts)}"

    session._utility_completion = answer
    assert await session.handle_command("/btw what file?") == "answer 1"
    assert await session.handle_command("/btw and why?") == "answer 2"
    assert "what file?" in prompts[1] and "answer 1" in prompts[1]
    assert await session.handle_command("/btw clear") == "Side thread cleared."
    await session.handle_command("/btw fresh")
    assert "what file?" not in prompts[2]
    await session.shutdown()


async def _session(tmp_path):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    return await CodingSession.create(
        provider=ReplayProvider([text_turn("x")]), provider_name="replay",
        model="m", system="s", tools=[], cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_logs_changelog_update_and_share_run_in_the_session(tmp_path, monkeypatch):
    import rlmagent.model.transport.client as client

    monkeypatch.setenv("RLM_AGENT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("RLM_AGENT_OFFLINE", raising=False)
    session = await _session(tmp_path)
    assert "No logs yet" in await session.handle_command("/logs")
    folder = tmp_path / "logs" / str(session.session_id)
    folder.mkdir(parents=True)
    (folder / "run.log").write_text("\n".join(f"l{i}" for i in range(100)), encoding="utf-8")
    shown = await session.handle_command("/logs 5")
    assert shown.endswith("l99") and "l94" not in shown and "l95" in shown

    document = {"info": {"version": "9.9.9"}, "releases": {
        "9.9.9": [{"upload_time": "2026-11-12T10:00:00"}],
        "0.1.0": [{"upload_time": "2026-01-01T10:00:00"}]}}
    monkeypatch.setattr(client, "fetch_json", lambda *a, **k: document)
    log = await session.handle_command("/changelog")
    assert log.index("9.9.9") < log.index("0.1.0")
    assert "9.9.9 is available" in await session.handle_command("/update")

    preview = await session.handle_command("/share")
    assert "Nothing uploaded" in preview and "SHA-256:" in preview
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert "SHA-256" in await session.handle_command("/share publish deadbeef")
    await session.shutdown()


def test_prices_come_only_from_the_listing():
    from rlmagent.model.discovery import PRICES, price_label, remember_prices

    PRICES.pop("openrouter", None)
    remember_prices("openrouter", [
        {"id": "a/model", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
        {"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "no/price"},
    ])
    assert price_label("openrouter", "a/model") == "$3.00/$15.00 per M"
    assert price_label("openrouter", "free/model") == "free"
    assert price_label("openrouter", "no/price") == ""
    assert price_label("anthropic", "anything") == ""


@pytest.mark.asyncio
async def test_sidebar_groups_sessions_and_searches():
    from rlmagent.tui.adapter import SessionEntry
    from rlmagent.tui.sidebar import SectionHeading, SessionItem, Sidebar

    entries = [
        SessionEntry("aaa111", "Fix the parser", 3.0, True, "running"),
        SessionEntry("bbb222", "Write docs", 2.0, False, "idle"),
        SessionEntry("ccc333", "Old spike", 1.0, False, "inactive"),
    ]
    app = RlmAgentApp()
    async with app.run_test() as pilot:
        bar = app.query_one(Sidebar)
        await bar.show(entries)
        await pilot.pause()
        headings = [h.query_one("Label").content for h in bar.query(SectionHeading)]
        assert [str(h) for h in headings] == ["Running (1)", "Idle (1)", "Inactive (1)"]
        search = bar.query_one("#session-search")
        async def rows_after(text):
            search.value = text
            for _ in range(20):  # the rebuild is async; wait for it to land
                await pilot.pause(0.05)
                rows = [i.session_id for i in bar.query(SessionItem)]
                if bar._shape and bar._shape[0] == text:
                    return rows
            return rows

        assert await rows_after("docs") == ["bbb222"]
        assert await rows_after("ccc") == ["ccc333"]


def test_busy_session_names_its_holder(tmp_path):
    import os

    from rlmagent.harness.session.lease import SessionBusy, describe_holder

    holder = tmp_path / "x.holder"
    holder.write_text('{"pid": 4242, "since": 0, "command": "rlm-agent worker run"}', encoding="utf-8")
    text = describe_holder(holder)
    assert "process 4242" in text and "rlm-agent worker run" in text
    assert describe_holder(tmp_path / "missing.holder") == ""
    assert issubclass(SessionBusy, RuntimeError) and os.getpid()


def test_attach_reports_drops_returns_and_version_mismatch(monkeypatch):
    from rlmagent.tui.worker_attach import WorkerAttachApp

    notes = []
    view = SimpleNamespace(add_notice=lambda text, error=False: notes.append(text))
    app = WorkerAttachApp("w1")
    import importlib

    monkeypatch.setattr(importlib.import_module("rlmagent.cli.main"), "get_version", lambda: "1.0.0")
    app.note_connection(view, {"alive": True, "version": "0.9.0"})
    assert any("0.9.0" in n and "1.0.0" in n for n in notes)
    app.note_connection(view, {"alive": True, "version": "0.9.0"})
    assert sum("0.9.0" in n for n in notes) == 1  # warned once
    app.note_connection(view, {"alive": False})
    app.note_connection(view, {"alive": True, "version": "1.0.0"})
    assert any("stopped" in n for n in notes) and any("back" in n for n in notes)


@pytest.mark.asyncio
async def test_first_run_welcomes_then_connects_more_than_one_provider():
    from rlmagent.config.store import load_config
    from rlmagent.tui.app import CONNECT_MORE, START_NOW, SetupApp, save_setup
    from rlmagent.tui.choice_screen import ChoiceScreen
    from rlmagent.tui.model_screen import ModelChoice, ModelScreen

    app = SetupApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChoiceScreen)  # the welcome
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelScreen)
        app.screen.dismiss(ModelChoice("openai", "OpenAI", "gpt-x", "sk-test-1"))
        await pilot.pause()
        assert isinstance(app.screen, ChoiceScreen)
        assert app.screen.options == [START_NOW, CONNECT_MORE]
        app.screen.dismiss(CONNECT_MORE)
        await pilot.pause()
        assert isinstance(app.screen, ModelScreen)
        app.screen.dismiss(ModelChoice("mistral", "Mistral", "m-large", "mk-test-2"))
        await pilot.pause()
        app.screen.dismiss(START_NOW)
        await pilot.pause()
    assert app.choice.provider == "openai" and [c.provider for c in app.extra] == ["mistral"]

    save_setup(app.choice, app.extra)
    config = load_config()
    assert config.provider == "openai" and config.model == "gpt-x"
    assert set(config.api_keys) == {"openai", "mistral"}
