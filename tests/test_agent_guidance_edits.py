"""The agent saves, changes and removes its own guidance from a cell while it works."""

import pytest
from _scripted import text_turn

from rlmagent.conversation import CodingSession
from rlmagent.model.scripted import ReplayProvider
from rlmagent.refinement.controller import SELF_EDIT_LIMIT, Refinement
from rlmagent.refinement.store import HarnessStore


async def session_for(path):
    session = await CodingSession.create(
        provider=ReplayProvider([text_turn("ok")]), provider_name="scripted", model="test",
        system="Base policy", tools=[],
    )
    session._skills = []
    session.refinement = Refinement(session, HarnessStore(path))
    return session


@pytest.mark.asyncio
async def test_save_is_written_at_once_versioned_and_readable(tmp_path):
    path = tmp_path / "state.json"
    session = await session_for(path)
    refinement = session.refinement
    result = refinement.agent_edit("save", "focused-check", content="Run pytest -q tests/unit first.",
                                   reason="The user corrected the check command")
    assert result["changed"] and result["revision"] == 1 and result["undo"] == "/refine rollback 0"
    assert path.exists() and HarnessStore(path).read().entries["focused-check"].kind == "prompt"
    assert refinement.read_guidance("focused-check")["content"].startswith("Run pytest")
    note = refinement.active.history[-1]
    assert note.evidence == "The user corrected the check command"
    # The prompt picks it up at the next turn boundary.
    assert "focused-check" not in session._system
    refinement.at_boundary()
    assert "focused-check" in session._system


@pytest.mark.asyncio
async def test_update_keeps_fields_and_remove_deletes(tmp_path):
    session = await session_for(tmp_path / "state.json")
    refinement = session.refinement
    refinement.agent_edit("save", "style", kind="prompt", title="Style", content="Use tabs.", reason="seen")
    refinement.agent_edit("save", "style", content="Use four spaces.", reason="user said so")
    entry = refinement.read_guidance("style")
    assert entry["title"] == "Style" and entry["content"] == "Use four spaces."
    unchanged = refinement.agent_edit("save", "style", content="Use four spaces.", reason="again")
    assert unchanged["changed"] is False
    refinement.agent_edit("remove", "style", reason="no longer true")
    assert refinement.read_guidance() == []
    assert refinement.active.revision == 3


@pytest.mark.asyncio
async def test_refusals(tmp_path):
    session = await session_for(tmp_path / "state.json")
    refinement = session.refinement
    with pytest.raises(ValueError, match="reason"):
        refinement.agent_edit("save", "x", content="text", reason="")
    with pytest.raises(ValueError):
        refinement.agent_edit("save", "Bad Id", content="text", reason="r")
    with pytest.raises(ValueError):
        refinement.agent_edit("save", "wf", kind="skill", content="no frontmatter", reason="r")
    with pytest.raises(ValueError, match="No guidance"):
        refinement.agent_edit("remove", "missing", reason="r")
    assert "off" in await refinement.command("self off")
    with pytest.raises(PermissionError, match="off"):
        refinement.agent_edit("save", "x", content="text", reason="r")
    assert "on" in await refinement.command("self on")
    session.set_plan_mode(True)
    with pytest.raises(PermissionError, match="read-only"):
        refinement.agent_edit("save", "x", content="text", reason="r")
    session.set_plan_mode(False)
    assert refinement.active.revision == 0


@pytest.mark.asyncio
async def test_per_turn_limit_resets_at_boundary(tmp_path):
    refinement = (await session_for(tmp_path / "state.json")).refinement
    for n in range(SELF_EDIT_LIMIT):
        refinement.agent_edit("save", f"note-{n}", content=f"fact {n}", reason="r")
    with pytest.raises(PermissionError, match="per turn"):
        refinement.agent_edit("save", "one-more", content="fact", reason="r")
    refinement.at_boundary()
    refinement.agent_edit("save", "one-more", content="fact", reason="r")


@pytest.mark.asyncio
async def test_prompt_tells_the_agent_only_when_it_may_edit(tmp_path):
    session = await session_for(tmp_path / "state.json")
    assert "save_guidance" in session._system
    await session.refinement.command("self off")
    session.refinement.at_boundary()
    assert "save_guidance" not in session._system


@pytest.mark.asyncio
async def test_rollback_undoes_an_agent_edit(tmp_path):
    session = await session_for(tmp_path / "state.json")
    refinement = session.refinement
    refinement.agent_edit("save", "fact", content="Wrong fact.", reason="r")
    await refinement.command("rollback 0")
    assert refinement.read_guidance() == []
