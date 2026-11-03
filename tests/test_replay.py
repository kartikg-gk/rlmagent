import json
import os
import time
from dataclasses import asdict, dataclass

from textual.app import App

from rlmagent.tui.inspect import AgentTreeScreen, StepScreen, agent_label
from rlmagent.tui.replay import ReplayApp, agent_rows, format_rows, load_events, runs_of
from rlmagent.tui.steps import StepCard


@dataclass
class Spend:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None


def _events():
    rows = []
    for run in range(2):
        rows += [
            {"kind": "run_started", "run_id": "root", "parent_run_id": None, "depth": 0},
            {"kind": "step_started", "run_id": "root", "step": 1, "started": 1.0},
            {"kind": "code_generated", "run_id": "root", "step": 1, "code": "x = 1"},
            {"kind": "run_started", "run_id": "child123456", "parent_run_id": "root", "depth": 1},
            {"kind": "step_started", "run_id": "child123456", "step": 1, "started": 1.5},
            {"kind": "step_completed", "run_id": "child123456", "step": 1,
             "usage": asdict(Spend(total_tokens=40, cost=0.002)), "ended": 2.0},
            {"kind": "run_completed", "run_id": "child123456"},
            {"kind": "step_completed", "run_id": "root", "step": 1,
             "usage": asdict(Spend(total_tokens=100, cost=0.01)), "ended": 3.0},
            {"kind": "run_completed", "run_id": "root"},
        ]
    return rows


def _saved(tmp_path):
    path = tmp_path / "s1.backend.events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in _events()) + "\nnot json\n")
    return path


def test_saved_events_split_into_runs_with_per_agent_totals(tmp_path):
    books = runs_of(load_events(_saved(tmp_path)))
    assert len(books) == 2
    rows = agent_rows(books)
    assert [(r["run"], r["agent"], r["depth"], r["tokens"]) for r in rows] == [
        (1, "root", 0, 100), (1, "child123456", 1, 40),
        (2, "root", 0, 100), (2, "child123456", 1, 40)]
    assert rows[0]["seconds"] == 2.0 and rows[1]["cost"] == 0.002
    text = format_rows(rows)
    assert "123456" in text and "total: 4 agents, 4 steps, 280 tokens, cost $0.0240" in text
    assert format_rows([]).startswith("No agent steps")
    assert agent_label(books[0], "child123456").startswith("123456 · d1 · done · 1 steps")


async def test_replay_shows_cards_and_opens_a_step_and_the_tree(tmp_path):
    books = runs_of(load_events(_saved(tmp_path)))
    app = ReplayApp("s1", books)
    async with app.run_test() as pilot:
        await pilot.pause()
        cards = list(app.query(StepCard))
        assert len(cards) == 4 and cards[1].styles.margin.left == 2
        cards[0].focus()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, StepScreen) and app.screen.step.code == "x = 1"
        await pilot.press("escape")
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, AgentTreeScreen)


def test_session_steps_command_and_latest_session_skip_event_logs(tmp_path, capsys):
    from rlmagent.cli.main import _handle_session, _latest_session_id

    (tmp_path / "s1.jsonl").write_text("")
    time.sleep(0.01)
    _saved(tmp_path)
    later = time.time() + 5
    os.utime(tmp_path / "s1.backend.events.jsonl", (later, later))
    assert _latest_session_id(tmp_path) == "s1"
    assert _handle_session(["steps", "--session-dir", str(tmp_path)]) == 0
    assert "total: 4 agents" in capsys.readouterr().out
    assert _handle_session(["steps", "s1", "--json", "--session-dir", str(tmp_path)]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 4
