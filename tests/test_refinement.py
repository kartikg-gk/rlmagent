import asyncio
import json
from types import SimpleNamespace

import pytest
from _scripted import text_turn

from rlmagent.conversation import CodingSession
from rlmagent.harness.contracts.transcript import CallBlock, HumanEntry, ModelEntry
from rlmagent.harness.provider.wire import StreamCloseEvent
from rlmagent.model.scripted import ReplayProvider
from rlmagent.refinement.controller import Refinement, attach
from rlmagent.refinement.engine import Draft, evidence_text
from rlmagent.refinement.store import Guidance, HarnessStore, Snapshot

DECISION = {
    "summary": "Keep the check",
    "evidence": "The user corrected our check command",
    "expected": "Use the focused check next time",
}
ENTRY = {
    "id": "focused-check",
    "kind": "prompt",
    "title": "Focused check",
    "content": "Run python verify_project.py --focused when checking this project.",
}


def action(tool, arguments):
    return text_turn(json.dumps({"tool": tool, "arguments": arguments}))


def good_turns(entry=None):
    return [
        action("guidance_inspect", {}),
        action("guidance_edit", {"action": "create", "entry": entry or ENTRY}),
        text_turn(json.dumps(DECISION)),
    ]


async def session_for(turns, path=None):
    session = await CodingSession.create(
        provider=ReplayProvider(turns),
        provider_name="scripted",
        model="test",
        system="Base policy",
        tools=[],
    )
    session._skills = []
    session.refinement = Refinement(session, HarnessStore(path))
    session.refinement.review_first = False  # these tests cover what happens after approval
    return session


async def finish(session):
    await session.refinement.task
    if session.refinement.phase == "awaiting review":
        await session.refinement.command("apply")
    assert session.refinement.phase in ("ready", "completed"), session.refinement.status()
    session.refinement.at_boundary()
    assert session.refinement.phase == "completed", session.refinement.status()


async def test_manual_job_is_single_flight_and_activates_only_at_boundary(tmp_path):
    path = tmp_path / "harness.json"
    session = await session_for(good_turns(), path)
    session.harness._running = True
    try:
        assert "idle" in await session.handle_command("/refine status")
        assert not session._provider.calls
        await session.handle_command("/refine Remember the check")
        first = session.refinement.task
        assert "Finish or cancel" in await session.handle_command("/refine another request")
        assert session.refinement.task is first
        await first
        assert session.refinement.phase == "ready"
        assert session._system == "Base policy" and not path.exists()
        assert session.refinement.read_guidance() == []
        session.refinement.at_boundary()
        assert not path.exists()
        session.harness._running = False
        session.refinement.at_boundary()
        assert ENTRY["content"] in session._system
        assert session._system.startswith("Base policy")
        state = HarnessStore(path).read()
        assert state.revision == 1 and state.history[0].expected == DECISION["expected"]
        assert state.history[0].evidence == DECISION["evidence"]
        assert session.refinement.read_guidance(ENTRY["id"])["content"] == ENTRY["content"]
    finally:
        await session.shutdown()


async def test_native_calls_use_only_the_restricted_tools():
    def call(name, arguments, call_id):
        return [
            StreamCloseEvent(
                reason="toolUse",
                message=ModelEntry(content=[CallBlock(name=name, arguments=arguments, id=call_id)]),
            )
        ]

    session = await session_for(
        [
            call("guidance_inspect", {}, "read"),
            call("guidance_edit", {"action": "create", "entry": ENTRY}, "stage"),
            text_turn(json.dumps(DECISION)),
        ]
    )
    try:
        await session.handle_command("/refine")
        await finish(session)
        for _, _, _, tools in session._provider.calls:
            assert {tool.name for tool in tools} == {"guidance_inspect", "guidance_edit"}
        assert session.refinement.active.entries[ENTRY["id"]].content == ENTRY["content"]
    finally:
        await session.shutdown()


async def test_bad_response_does_not_commit_staged_edits(tmp_path):
    path = tmp_path / "state.json"
    session = await session_for(good_turns()[:-1] + text_turn("not json"), path)
    try:
        await session.handle_command("/refine")
        await session.refinement.task
        assert session.refinement.phase == "failed"
        session.refinement.at_boundary()
        assert not path.exists() and not session.refinement.active.entries
    finally:
        await session.shutdown()


async def test_cancel_before_task_starts_and_shutdown_discard_work(tmp_path):
    session = await session_for(good_turns(), tmp_path / "state.json")
    await session.handle_command("/refine")
    await session.handle_command("/refine cancel")
    await session.shutdown()
    assert not session.refinement.store.path.exists()


async def test_abort_discards_ready_candidate(tmp_path):
    session = await session_for(good_turns(), tmp_path / "state.json")
    session.harness._running = True
    try:
        await session.handle_command("/refine")
        await session.refinement.task
        session.abort()
        session.harness._running = False
        session.refinement.at_boundary()
        assert session.refinement.phase == "cancelled"
        assert not session.refinement.active.entries
    finally:
        await session.shutdown()


@pytest.mark.parametrize("gate", ["plan", "readonly"])
async def test_write_gate_is_rechecked_before_commit(tmp_path, gate):
    session = await session_for(good_turns(), tmp_path / "state.json")
    session.harness._running = True
    try:
        await session.handle_command("/refine")
        await session.refinement.task
        if gate == "plan":
            session.set_plan_mode(True)
        else:
            session.improver = SimpleNamespace(writable=False)
        session.harness._running = False
        session.refinement.at_boundary()
        assert session.refinement.phase == "cancelled"
        assert not session.refinement.store.path.exists()
        assert "read-only" in await session.handle_command("/refine more")
    finally:
        await session.shutdown()


async def test_new_session_inherits_project_harness_and_other_project_does_not(tmp_path):
    session = await session_for(good_turns())
    session._cwd = str(tmp_path / "project-a")
    controller = attach(session, tmp_path / "home", persist=True)
    try:
        await session.handle_command("/refine")
        await finish(session)
        next_session = await session_for([])
        next_session._cwd = session.cwd
        attach(next_session, tmp_path / "home", persist=True)
        assert ENTRY["content"] in next_session._system
        other = await session_for([])
        other._cwd = str(tmp_path / "project-b")
        attach(other, tmp_path / "home", persist=True)
        assert ENTRY["content"] not in other._system
        await next_session.shutdown()
        await other.shutdown()
        assert controller.active.revision == 1
    finally:
        await session.shutdown()


async def test_stale_candidate_rejected_without_overwriting_concurrent_change(tmp_path):
    path = tmp_path / "state.json"
    session = await session_for(good_turns(), path)
    session.harness._running = True
    try:
        await session.handle_command("/refine")
        await session.refinement.task
        store = HarnessStore(path)
        candidate = Snapshot(revision=1, entries={"other": Guidance(**{**ENTRY, "id": "other"})})
        store.commit(Snapshot(), candidate)
        session.harness._running = False
        session.refinement.at_boundary()
        assert session.refinement.phase == "failed"
        assert "changed" in session.refinement.status()
        assert set(store.read().entries) == {"other"}
    finally:
        await session.shutdown()


async def test_progressive_skill_is_validated_and_available_to_slash_expansion(tmp_path):
    content = "---\nname: focused-check\ndescription: Verify a project change\n---\nRun the focused checker."
    entry = {**ENTRY, "kind": "skill", "content": content}
    session = await session_for(good_turns(entry), tmp_path / "state.json")
    installed = tmp_path / "SKILL.md"
    installed.write_text("Installed content")
    try:
        await session.handle_command("/refine")
        await finish(session)
        assert "Run the focused checker." not in session._system
        assert "Verify a project change" in session._system
        assert "Run the focused checker." in session.expand_skill("/skill:focused-check")
        assert session.refinement.read_guidance("focused-check")["content"] == content
        assert installed.read_text() == "Installed content"
    finally:
        await session.shutdown()


@pytest.mark.parametrize(
    "content",
    [
        "plain text",
        "---\nname: wrong\ndescription: check\n---\nSteps",
        "---\nname: focused-check\nname: focused-check\ndescription: x\n---\nSteps",
        "---\nname: focused-check\ndescription: !!python/object:bad {}\n---\nSteps",
    ],
)
def test_invalid_or_unsafe_skill_metadata_is_rejected(content):
    with pytest.raises(ValueError):
        Guidance(**{**ENTRY, "kind": "skill", "content": content})


def test_draft_requires_inspection_and_typed_edit_and_limits_mutations():
    draft = Draft(Snapshot())
    with pytest.raises(ValueError, match="Inspect"):
        draft.edit({"action": "create", "entry": ENTRY})
    draft.inspect({})
    with pytest.raises(ValueError):
        draft.edit({"action": "create", "entry": {**ENTRY, "path": "../../file"}})
    assert draft.state.entries == {}
    draft.edit({"action": "create", "entry": ENTRY})
    draft.edit({"action": "update", "entry": {**ENTRY, "content": "Revised check"}})
    assert draft.state.entries[ENTRY["id"]].content == "Revised check"
    draft.edit({"action": "delete", "id": ENTRY["id"]})
    assert not draft.finish(DECISION).changed
    for _ in range(6):
        with pytest.raises(ValueError):
            draft.edit({"action": "delete", "id": "missing"})
    with pytest.raises(ValueError, match="allowance"):
        draft.edit({"action": "create", "entry": ENTRY})


def test_corrupt_store_does_not_silently_reset_and_snapshot_is_detached(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("broken")
    with pytest.raises(ValueError):
        HarnessStore(path).read()
    assert path.read_text() == "broken"
    store = HarnessStore(None)
    candidate = Snapshot(revision=1, entries={ENTRY["id"]: Guidance(**ENTRY)})
    store.commit(Snapshot(), candidate)
    candidate.entries.clear()
    assert store.read().entries


def test_evidence_preserves_tool_calls_and_errors():
    from rlmagent.harness.contracts.transcript import ToolOutcomeEntry

    messages = [
        HumanEntry(content="Fix this"),
        ModelEntry(content=[CallBlock(name="python", id="c1", arguments={"code": "run_check()"})]),
        ToolOutcomeEntry(
            tool_name="python", tool_call_id="c1", content="failed check", is_error=True
        ),
    ]
    evidence = evidence_text(messages)
    assert "run_check()" in evidence and "failed check" in evidence and "isError" in evidence


async def test_job_returns_immediately_while_provider_waits_and_cancel_stops_it():
    class WaitingProvider:
        def __init__(self):
            self.entered = asyncio.Event()
            self.exited = asyncio.Event()

        async def stream_response(self, **kwargs):
            try:
                self.entered.set()
                await asyncio.Event().wait()
                yield None
            finally:
                self.exited.set()

    session = await session_for([])
    session._provider = provider = WaitingProvider()
    try:
        result = await asyncio.wait_for(session.handle_command("/refine"), 1)
        assert "reviewing" in result
        await asyncio.wait_for(provider.entered.wait(), 1)
        await session.handle_command("/refine cancel")
        await asyncio.wait_for(provider.exited.wait(), 1)
        assert session.refinement.phase == "cancelled"
    finally:
        await session.shutdown()


async def test_session_replacement_keeps_command_without_baking_old_guidance(tmp_path):
    session = await session_for(good_turns(), tmp_path / "state.json")
    await session.handle_command("/refine")
    await finish(session)
    replacement = await session.new_session()
    try:
        assert "completed" not in await replacement.handle_command("/refine status")
        assert replacement._system.count(ENTRY["content"]) == 1
        assert replacement.refinement.base_system == "Base policy"
    finally:
        await replacement.shutdown()


async def test_readonly_kernel_guidance_and_new_child_prompts(tmp_path):
    from rlmness import Allowance

    from rlmagent.agents.tree import AgentNode, AgentTree

    session = await session_for(good_turns())
    tree = AgentTree(
        provider=session._provider,
        provider_name="scripted",
        model="test",
        cwd=str(tmp_path),
        allowance=Allowance(),
        sessions_dir=None,
        system_for=lambda node, tools: "Child base",
        first_message_for=lambda node: node.task,
    )
    session.agent_tree = tree
    session._cwd = str(tmp_path)
    attach(session, tmp_path / "home", persist=False)
    try:
        await session.handle_command("/refine")
        await finish(session)
        tree.nodes["root"] = AgentNode(id="root", parent_id=None, depth=0)
        result = await tree._handle("root", "guidance", {"id": ENTRY["id"]})
        assert result["content"] == ENTRY["content"]
        child = AgentNode(id="child", parent_id="root", depth=1)
        assert ENTRY["content"] in tree.system_for(child, [])
    finally:
        session.agent_tree = None
        await session.shutdown()


def test_cross_process_compare_and_swap_allows_only_one_writer(tmp_path):
    import os
    import subprocess
    import sys
    import time

    script = tmp_path / "writer.py"
    script.write_text(
        """import sys,time
from pathlib import Path
from rlmagent.refinement.store import HarnessStore, Guidance
folder=Path(sys.argv[1]); name=sys.argv[2]
store=HarnessStore(folder/'state.json'); original=store.read()
candidate=original.model_copy(deep=True); candidate.revision+=1
candidate.entries[name]=Guidance(id=name,kind='prompt',title=name,content=name)
(folder/(name+'.ready')).touch()
while not (folder/'go').exists(): time.sleep(.01)
try:
    store.commit(original,candidate)
    print('committed')
except ValueError:
    print('stale')
""",
        encoding="utf-8",
    )
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        for name in ("first", "second")
    ]
    try:
        deadline = time.monotonic() + 15
        while not all((tmp_path / f"{name}.ready").exists() for name in ("first", "second")):
            assert time.monotonic() < deadline, "Writers did not reach the commit barrier"
            time.sleep(0.02)
        (tmp_path / "go").touch()
        outputs = [process.communicate(timeout=15) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        assert sorted(stdout.strip() for stdout, _ in outputs) == ["committed", "stale"]
        state = HarnessStore(tmp_path / "state.json").read()
        assert state.revision == 1 and len(state.entries) == 1
        assert not list(tmp_path.glob(".harness-*"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


async def test_streaming_api_adapter_runs_review_without_network(tmp_path):
    import httpx

    from rlmagent.model.oai_compatible import OpenAIProvider
    from rlmagent.model.settings import Credential, OpenAIProfile, RetryPolicy

    replies = iter(
        [
            ("guidance_inspect", {}),
            ("guidance_edit", {"action": "create", "entry": ENTRY}),
            (None, DECISION),
        ]
    )
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/chat/completions"
        assert {tool["function"]["name"] for tool in payload["tools"]} == {
            "guidance_inspect",
            "guidance_edit",
        }
        name, arguments = next(replies)
        if name is not None:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"op-{len(requests)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
            }
            reason = "tool_calls"
        else:
            delta = {"content": json.dumps(arguments)}
            reason = "stop"
        frames = [
            {
                "id": "review",
                "model": "fixture",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
        return httpx.Response(
            200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )

    provider = OpenAIProvider(
        OpenAIProfile(
            name="fixture",
            credential=Credential("test-placeholder", "https://fixture.invalid/v1"),
            retry=RetryPolicy(max_retries=0),
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    session = await CodingSession.create(
        provider=provider, provider_name="fixture", model="fixture", system="Base policy", tools=[]
    )
    controller = Refinement(session, HarnessStore(tmp_path / "state.json"))
    session.refinement = controller
    try:
        await session.handle_command("/refine")
        await finish(session)
        assert len(requests) == 3
        assert controller.active.entries[ENTRY["id"]].content == ENTRY["content"]
        assert any(message["role"] == "tool" for message in requests[-1]["messages"])
    finally:
        await session.shutdown()
        await provider.close()


async def test_a_proposal_waits_for_review_before_it_is_saved(tmp_path):
    path = tmp_path / "harness.json"
    session = await session_for(good_turns(), path)
    session.refinement.review_first = True
    assert (await session.handle_command("/refine show")).startswith("No proposal is waiting")
    assert (await session.handle_command("/refine apply")).startswith("No proposal is waiting")
    await session.handle_command("/refine Remember the check")
    await session.refinement.task
    assert session.refinement.phase == "awaiting review"
    shown = await session.handle_command("/refine show")
    assert shown.startswith("add prompt focused-check: Focused check")
    assert "+Run python verify_project.py --focused" in shown
    assert "Finish or cancel" in await session.handle_command("/refine again")
    assert not HarnessStore(path).read().entries
    await session.handle_command("/refine apply")
    assert session.refinement.phase == "completed"
    assert "focused-check" in HarnessStore(path).read().entries


async def test_a_dropped_proposal_saves_nothing(tmp_path):
    path = tmp_path / "harness.json"
    session = await session_for(good_turns(), path)
    session.refinement.review_first = True
    await session.handle_command("/refine Remember the check")
    await session.refinement.task
    await session.handle_command("/refine cancel")
    assert session.refinement.phase == "cancelled" and session.refinement.draft is None
    assert not HarnessStore(path).read().entries
