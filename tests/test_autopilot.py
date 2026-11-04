import sys

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
