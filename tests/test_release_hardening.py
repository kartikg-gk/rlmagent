import asyncio
from types import SimpleNamespace

import pytest
from _scripted import code_turn, text_turn

from rlmagent.conversation import CodingSession
from rlmagent.harness.contracts.transcript import HumanEntry
from rlmagent.harness.provider.wire import StreamFaultEvent
from rlmagent.model.deadline import bounded_stream
from rlmagent.model.scripted import ReplayProvider


async def session_with_history(model):
    session = await CodingSession.create(provider=model, provider_name="fixture", model="m", system="base")
    for text in ("first", "second", "third"):
        async for _ in session.submit(text):
            pass
    return session


async def test_compaction_retries_invalid_summaries_without_executing_tools():
    model = ReplayProvider([text_turn("one"), text_turn("two"), text_turn("three"),
        text_turn(""), code_turn("raise AssertionError('must never execute')"), text_turn("valid summary")])
    session = await session_with_history(model)
    try:
        result = await session.compact()
        assert "valid summary" in result.summary
        assert session.compactions == 1
        assert len(model.calls) == 6
    finally:
        await session.shutdown()


async def test_failed_compaction_keeps_history_and_does_not_spend_compaction_limit():
    model = ReplayProvider([text_turn("one"), text_turn("two"), text_turn("three"),
        text_turn(""), text_turn(" "), text_turn("")])
    session = await session_with_history(model)
    session.policy = SimpleNamespace(max_compactions=1)
    before = session.transcript
    try:
        with pytest.raises(ValueError, match="3 invalid summaries"):
            await session.compact()
        assert session.transcript == before
        assert getattr(session, "compactions", 0) == 0
        assert len(model.calls) == 6
    finally:
        await session.shutdown()


async def test_session_write_failure_is_explicit_and_header_is_retryable(tmp_path):
    session = await CodingSession.create(provider=ReplayProvider([]), provider_name="fixture",
        model="m", system="base", sessions_dir=tmp_path)
    path = session._vault.path
    path.mkdir()  # A real filesystem obstruction, independent of mocks.
    try:
        with pytest.raises(OSError, match="Could not create session"):
            await session._persist_entry(HumanEntry(content="retain me"))
        assert session._pending_header is not None
        path.rmdir()
        await session._persist_entry(HumanEntry(content="saved after repair"))
        records = await session._vault.read_all()
        assert records[-1].message.text == "saved after repair"
    finally:
        await session.shutdown()


async def test_deadline_ends_the_entire_stream_and_closes_provider():
    closed = asyncio.Event()
    async def slow():
        try:
            await asyncio.sleep(10)
            yield None
        finally:
            closed.set()
    events = [event async for event in bounded_stream(slow(), "fixture", 0.02)]
    assert closed.is_set()
    assert len(events) == 1 and isinstance(events[0], StreamFaultEvent)
    assert "deadline" in events[0].error.error_message


async def test_cancelling_a_stream_propagates_and_closes_provider():
    started, closed = asyncio.Event(), asyncio.Event()
    async def slow():
        try:
            started.set()
            await asyncio.sleep(10)
            yield None
        finally:
            closed.set()
    async def consume():
        return [event async for event in bounded_stream(slow(), "fixture", 60)]
    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


def test_missing_sandbox_never_silently_executes_on_host(tmp_path, monkeypatch):
    from rlmagent.kernel import wasm
    from rlmagent.runtime import make_agent_tree
    monkeypatch.setattr(wasm, "available", lambda: False)
    with pytest.raises(ValueError, match="explicitly select --kernel local"):
        make_agent_tree(SimpleNamespace(kernel="wasm"), ReplayProvider([]), "fixture", "m",
                        None, cwd=str(tmp_path), backend=True)


@pytest.mark.parametrize("engine", ["session", "rlmness"])
async def test_request_deadline_applies_through_the_real_session_builder(tmp_path, monkeypatch, engine):
    import importlib

    from rlmagent.runtime import build_session
    cli = importlib.import_module("rlmagent.cli.main")
    closed = asyncio.Event()
    class SlowProvider:
        async def stream_response(self, **kwargs):
            try:
                await asyncio.sleep(10)
                yield None
            finally:
                closed.set()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli, "_resolve_provider", lambda _: SlowProvider())
    ns = cli._build_run_parser().parse_args(["-p", "--no-session", "--no-approve",
        "--provider", "openai", "--model", "fixture", "--engine", engine,
        "--kernel", "local", "--max-calls", "1", "--request-timeout", "0.03"])
    session = await build_session(ns)
    try:
        events = [event async for event in session.submit("work")]
        assert closed.is_set()
        assert session.agent_tree.request_timeout == 0.03
        assert any("deadline" in str(event).lower() for event in events)
    finally:
        await session.shutdown()


async def test_kernel_and_shell_do_not_inherit_host_credentials(tmp_path, monkeypatch):
    from rlmagent.agents.approval import Approver
    from rlmagent.agents.workspace import Workspace
    from rlmagent.kernel.session import KernelSession
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("RLM_AGENT_CONTROL_SOCKET", "synthetic-control")
    kernel = KernelSession(cwd=str(tmp_path), timeout=20)
    try:
        result = await kernel.run("import os; print(os.getenv('OPENAI_API_KEY'), os.getenv('RLM_AGENT_CONTROL_SOCKET'))")
        assert result.error is None and result.output.strip() == "None None"
    finally:
        await kernel.shutdown()
    result = await Workspace(tmp_path, Approver("auto")).run(
        'printf "%s|%s" "$OPENAI_API_KEY" "$RLM_AGENT_CONTROL_SOCKET"')
    assert result["exit"] == 0 and result["output"] == "|"
