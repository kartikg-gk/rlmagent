"""The chat screen's keys, end to end: step rows, opening them, the side panel, the dashboard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlmagent.tui.app import RlmAgentApp
from rlmagent.tui.logo import Logo
from rlmagent.tui.run_view import DashboardScreen
from rlmagent.tui.side_panel import SidePanel
from rlmagent.tui.steps import StepBook, StepCard
from rlmagent.tui.transcript import TranscriptView


def _book() -> StepBook:
    book = StepBook()
    book.feed({"kind": "run_started", "run_id": "root", "depth": 0, "model": "m"})
    for number, code in enumerate(["x = 1", "print(x + 1)"], 1):
        book.feed({"kind": "step_started", "run_id": "root", "step": number, "started": 1.0})
        book.feed({"kind": "code_generated", "run_id": "root", "step": number, "code": code})
        book.feed({"kind": "output_received", "run_id": "root", "step": number, "output": "2"})
        book.feed({"kind": "step_completed", "run_id": "root", "step": number, "usage": {}, "ended": 1.5})
    book.feed({"kind": "run_started", "run_id": "kid-001", "parent_run_id": "root", "depth": 1,
               "model": "m", "child_id": "c1"})
    return book


@pytest.mark.asyncio
async def test_rows_open_one_by_one_and_all_at_once():
    book = _book()
    app = RlmAgentApp()
    async with app.run_test(size=(150, 40)) as pilot:
        view = app.query_one(TranscriptView)
        assert view.query(Logo)  # the name heads the conversation
        for key in book.order:
            view.show_step(book.steps[key])
        await pilot.pause()
        cards = list(view.query(StepCard))
        assert len(cards) == 2 and all(card.collapsed for card in cards)
        await pilot.press("ctrl+o")
        assert not any(card.collapsed for card in cards)
        await pilot.press("ctrl+o")
        assert all(card.collapsed for card in cards)


@pytest.mark.asyncio
async def test_side_panel_tabs_and_dashboard_round_trip():
    book = _book()
    app = RlmAgentApp()
    async with app.run_test(size=(150, 40)) as pilot:
        app._bridge = SimpleNamespace(steps=book, session=SimpleNamespace(
            skills=[SimpleNamespace(name="websearch", description="Search")]))
        await pilot.pause(0.3)
        panel = app.query_one(SidePanel)
        assert panel.display
        tabs = panel.query_one("TabbedContent")
        assert tabs.active == "tab-skills"
        await pilot.press("ctrl+t")
        assert tabs.active == "tab-agents"
        assert "subagents 1" in str(tabs.get_tab("tab-agents").label)
        await pilot.press("f7")
        assert not panel.display
        await pilot.press("f5")
        await pilot.pause(0.3)
        assert isinstance(app.screen, DashboardScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, DashboardScreen)
        app._bridge = None


@pytest.mark.asyncio
async def test_narrow_terminal_hides_the_panel():
    app = RlmAgentApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert not app.query_one(SidePanel).display
