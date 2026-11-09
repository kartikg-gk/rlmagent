from types import SimpleNamespace

import pytest
from _scripted import text_turn

from rlmagent.agents.trace import TracedProvider, Tracer
from rlmagent.harness.contracts.transcript import HumanEntry
from rlmagent.model.scripted import ReplayProvider
from rlmagent.steering import Steering


async def test_steering_receipt_survives_restart_and_rejects_changed_payload(tmp_path):
    ledger = Steering(tmp_path / "instructions.json")
    assert ledger.accept("Use the focused check", "message-1")["status"] == "accepted"
    restored = Steering(ledger.path)
    assert restored.accept("Use the focused check", "message-1")["status"] == "accepted"
    with pytest.raises(ValueError):
        restored.accept("Different instruction", "message-1")
    model = ReplayProvider([text_turn("done")])
    tree = SimpleNamespace(
        provider=model,
        tracer=Tracer(),
        root_id=lambda: "root",
        session=SimpleNamespace(steering=restored),
    )
    provider = TracedProvider(tree, "root")
    events = [
        event
        async for event in provider.stream_response(
            model="m", system="s", messages=[HumanEntry(content="work")], tools=[]
        )
    ]
    assert events[-1].type == "done"
    assert "focused check" in model.calls[0][2][-1].text
    receipt = Steering(ledger.path).receipts()[0]
    assert receipt["status"] == "consumed" and receipt["deliveries"]["root"]["requestId"]
    # Independent of transcript rewinds: guidance and receipt remain available.
    text, pending = restored.context("root", root=True)
    assert "focused check" in text and pending == []


async def test_cancelled_model_boundary_keeps_steering_pending(tmp_path):
    from rlmagent.harness.contracts.transcript import ModelEntry
    from rlmagent.harness.provider.wire import StreamFaultEvent

    class Model:
        async def stream_response(self, **kw):
            yield StreamFaultEvent(reason="aborted", error=ModelEntry(stop_reason="aborted"))

    ledger = Steering(tmp_path / "instructions.json")
    ledger.accept("child instruction", "child-message", target="child")
    tree = SimpleNamespace(
        provider=Model(),
        tracer=Tracer(),
        root_id=lambda: "root",
        session=SimpleNamespace(steering=ledger),
    )
    events = [
        event
        async for event in TracedProvider(tree, "child").stream_response(
            model="m", system="s", messages=[], tools=[]
        )
    ]
    assert events[-1].reason == "aborted"
    assert ledger.receipts()[0]["status"] == "accepted"
    assert ledger.context("root", root=True) == ("", [])


async def test_session_refinement_is_inherited_by_branch_without_sibling_changes(tmp_path):
    from rlmagent.conversation import CodingSession
    from rlmagent.refinement.controller import attach
    from rlmagent.refinement.store import Guidance

    session = await CodingSession.create(
        provider=ReplayProvider([]),
        provider_name="scripted",
        model="m",
        system="base",
        tools=[],
        sessions_dir=tmp_path / "sessions",
        cwd=str(tmp_path),
    )
    controller = attach(session, tmp_path / "home", persist=True)
    try:
        # Materialize a saved branch point without a model request.
        root = await session.branch("starting point")
        assert "session" in await controller.command("scope session")
        original = controller.store.read()
        inherited = original.model_copy(deep=True)
        inherited.entries["base-note"] = Guidance(
            id="base-note", kind="prompt", title="Base", content="Inherited guidance"
        )
        inherited.revision += 1
        controller.store.commit(original, inherited)
        controller.at_boundary()
        child = await session.branch("child branch")
        baseline = controller.store.read()
        candidate = baseline.model_copy(deep=True)
        candidate.entries["child-note"] = Guidance(
            id="child-note", kind="prompt", title="Child", content="Only this branch"
        )
        candidate.revision += 1
        controller.store.commit(baseline, candidate)
        controller.at_boundary()
        assert "Only this branch" in session._system
        await session.rewind(root)
        assert "Inherited guidance" in session._system and "Only this branch" not in session._system
        assert controller.project_store.read().entries == {}
        await session.rewind(child)
        assert "Only this branch" in session._system
        assert "Restored" in await controller.command(f"rollback {inherited.revision}")
        assert "Only this branch" not in session._system
    finally:
        await session.shutdown()


async def test_external_refinement_gate_change_prevents_review_activation(tmp_path):
    import json

    from test_refinement import good_turns, session_for

    session = await session_for(good_turns(), tmp_path / "state.json")
    session.refinement.home = tmp_path
    try:
        session.harness._running = True
        await session.handle_command("/refine improve focused checks")
        await session.refinement.task
        (tmp_path / "refinement.json").write_text(json.dumps({"enabled": False}))
        session.harness._running = False
        session.refinement.at_boundary()
        assert not session.refinement.store.path.exists()
        assert session._system == "Base policy"
        assert session.refinement.draft is None
    finally:
        await session.shutdown()
