import asyncio
import sys

import pytest

from rlmagent.autopilot import Autopilot


def _pilot(tmp_path, used=None):
    counter = used if used is not None else [0]
    pilot = Autopilot(path=tmp_path / "s.goal.json", cwd=str(tmp_path),
                      used_tokens=lambda: counter[0])
    return pilot, counter


async def test_goal_is_kept_reminded_and_finished(tmp_path):
    pilot, used = _pilot(tmp_path)
    assert pilot.reminder() is None
    assert pilot.command_goal("").startswith("No goal")
    assert "Goal (active): make the tests pass" in pilot.command_goal("make the tests pass --tokens 900")
    assert "make the tests pass" in pilot.reminder()
    used[0] = 300
    assert (await pilot.goal_status())["tokens_used"] == 300
    again, _ = _pilot(tmp_path, used)
    again.load()
    assert again.goal.objective == "make the tests pass" and again.goal.token_budget == 900
    pilot.command_goal("pause")
    assert pilot.reminder() is None
    pilot.command_goal("resume")
    assert await pilot.goal_done("all green") == "Goal marked done."
    assert pilot.goal.status == "done" and await pilot.goal_done("x") == "There is no active goal."
    assert pilot.command_goal("clear").startswith("No goal")


async def test_a_failing_gate_keeps_the_goal_open(tmp_path):
    pilot, _ = _pilot(tmp_path)
    pilot.set_goal("ship it")
    pilot.command_autonomous(f'gate "{sys.executable}" -c "import sys; print(42); sys.exit(3)"')
    answer = await pilot.goal_done("done")
    assert answer.startswith("The gate command failed") and "exited 3" in answer and "42" in answer
    assert pilot.goal.status == "active"
    pilot.command_autonomous(f'gate "{sys.executable}" -c "pass"')
    assert await pilot.goal_done("done") == "Goal marked done."


async def test_named_gates_report_independent_attempts_and_recheck_on_retry(tmp_path):
    pilot, _ = _pilot(tmp_path)
    pilot.set_goal("verify release")
    pilot.command_autonomous(f'gate add build "{sys.executable}" -c "pass"')
    pilot.command_autonomous(
        f'gate add check "{sys.executable}" -c "from pathlib import Path; assert Path(\'ready\').exists()"'
    )
    assert "check: exited 1" in await pilot.goal_done("done")
    assert pilot.goal.status == "active"
    assert pilot.gate_results == {
        "build": {"status": "passed", "attempts": 1},
        "check": {"status": "failed", "attempts": 1},
    }
    reopened, _ = _pilot(tmp_path)
    reopened.load()
    assert reopened.gate_results == pilot.gate_results
    (tmp_path / "ready").touch()
    assert await reopened.goal_done("verified") == "Goal marked done."
    assert all(value == {"status": "passed", "attempts": 2} for value in reopened.gate_results.values())


async def test_interrupted_gate_leaves_goal_open_and_records_attempt(tmp_path, monkeypatch):
    pilot, _ = _pilot(tmp_path)
    pilot.set_goal("verify release")
    pilot.command_autonomous("gate add build pretend")
    entered = asyncio.Event()

    async def wait_for_cancel(command):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(pilot, "run_gate", wait_for_cancel)
    task = asyncio.create_task(pilot.goal_done("complete"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    reopened, _ = _pilot(tmp_path)
    reopened.load()
    assert reopened.goal.status == "active"
    assert reopened.gate_results == {"build": {"status": "interrupted", "attempts": 1}}


def test_autonomous_turns_stop_at_their_limits(tmp_path):
    pilot, used = _pilot(tmp_path)
    assert pilot.start_autonomous().startswith("Set a goal first")
    pilot.set_goal("refactor")
    pilot.command_autonomous("turns 2")
    pilot.start_autonomous()
    assert pilot.next_prompt().startswith("Continue toward the goal")
    assert pilot.next_prompt() and pilot.turns == 2
    assert pilot.next_prompt() is None and pilot.stopped_because == "reached 2 turns"

    pilot.command_autonomous("turns 50")
    pilot.command_autonomous("tokens 1000")
    pilot.start_autonomous()
    used[0] += 1500
    assert pilot.next_prompt() is None and pilot.stopped_because == "used 1,500 tokens"

    pilot.start_autonomous()
    pilot.goal.status = "done"
    assert pilot.next_prompt() is None and pilot.stopped_because == "goal is done"
    assert "Autonomous mode off (goal is done)" in pilot.command_autonomous("status")
    assert pilot.command_autonomous("bogus").startswith("Usage")


def test_autonomous_detects_failed_tool_cycle_across_worker_restart(tmp_path):
    from rlmagent.harness.contracts.transcript import CallBlock, ModelEntry, TextSegment, ToolOutcomeEntry

    def turn(index, target, *, failed=True):
        call_id = f"call-{index}"
        return [
            ModelEntry(content=[CallBlock(id=call_id, name="Read", arguments={"path": target})]),
            ToolOutcomeEntry(tool_call_id=call_id, tool_name="Read", content="same output",
                             is_error=failed),
            ModelEntry(content=[TextSegment(text=f"Different wording {index}")]),
        ]

    pilot, _ = _pilot(tmp_path)
    pilot.durable = True
    pilot.set_goal("find the file")
    pilot.start_autonomous()
    for index, target in enumerate(("a", "b", "a", "b")):
        pilot.observe_turn(turn(index, target))
    assert pilot.autonomous
    pilot.save()

    resumed, _ = _pilot(tmp_path)
    resumed.durable = True
    resumed.load()
    resumed.observe_turn(turn(4, "a"))
    assert resumed.autonomous
    resumed.observe_turn(turn(5, "b"))
    assert resumed.stopped_because == "the same failed operations repeated without progress"

    resumed.start_autonomous()
    for index, target in enumerate(("a", "b", "a", "b", "a", "b")):
        resumed.observe_turn(turn(index, target, failed=index != 3))
    assert resumed.autonomous


def test_autonomous_detects_repeated_node_cell_failure(tmp_path):
    from rlmagent.harness.contracts.transcript import ModelEntry, TextSegment

    pilot, _ = _pilot(tmp_path)
    pilot.set_goal("verify the code")
    pilot.start_autonomous()
    for index in range(3):
        pilot.observe_turn(
            [ModelEntry(content=[TextSegment(text=f"Attempt {index}")])],
            [{"kind": "code_generated", "run_id": "root", "step": 1, "code": "raise ValueError()"},
             {"kind": "output_received", "run_id": "root", "step": 1, "output": "ValueError", "error": True}],
        )
    assert not pilot.autonomous
    assert "failed operations" in pilot.stopped_because


def test_worker_recovers_queued_turn_and_never_replays_an_admitted_turn(tmp_path):
    pilot, used = _pilot(tmp_path)
    pilot.durable = True
    pilot.set_goal("verify changes")
    pilot.command_autonomous("turns 1")
    pilot.start_autonomous()
    queued = pilot.next_prompt()

    restarted, _ = _pilot(tmp_path, used)
    restarted.durable = True
    restarted.load()
    assert restarted.next_prompt() == queued
    assert restarted.turns == 1
    restarted.start_worker_turn(continuation=True)

    interrupted, _ = _pilot(tmp_path, used)
    interrupted.durable = True
    interrupted.load()
    assert interrupted.next_prompt() is None
    assert interrupted.goal.status == "active"
    assert "interrupted" in interrupted.stopped_because


async def test_goal_commands_reach_the_session(tmp_path):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([text_turn("ok")]), provider_name="replay", model="m",
        system="s", tools=[], cwd=str(tmp_path))
    assert await session.handle_command("/goal") == "Goals are not available here."
    session.autopilot, _ = _pilot(tmp_path)
    assert "Goal (active): fix it" in await session.handle_command("/goal fix it")
    assert "Autonomous: turn 0" in await session.handle_command("/autonomous")
    async for _ in session.submit("start"):
        pass
    assert any("[Goal] Keep working toward: fix it" in getattr(m, "text", "")
               for m in session.transcript)
    await session.shutdown()


async def test_repeated_session_replies_stop_autonomous_turns(tmp_path):
    from _scripted import text_turn

    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider

    session = await CodingSession.create(
        provider=ReplayProvider([text_turn(text) for text in ("waiting", "working", "working", "working")]),
        provider_name="replay", model="m", system="s", tools=[], cwd=str(tmp_path),
    )
    session.autopilot, _ = _pilot(tmp_path)
    session.autopilot.set_goal("finish it")
    session.autopilot.start_autonomous()
    try:
        prompt = "start"
        for index in range(4):
            async for _ in session.submit(prompt):
                pass
            prompt = session.autopilot.next_prompt()
            assert (prompt is None) == (index == 3)
        assert session.autopilot.goal.status == "active"
        assert session.autopilot.stopped_because == "no visible progress across 3 turns"
    finally:
        await session.shutdown()
