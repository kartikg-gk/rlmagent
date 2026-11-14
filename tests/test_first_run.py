"""What a new user meets when the sandbox cannot start: a notice saying why and what to do."""

from __future__ import annotations

import argparse

import pytest

from rlmagent.tui.app import RlmAgentApp, sandbox_note
from rlmagent.tui.messages import NoticeBlock
from rlmagent.tui.transcript import TranscriptView


def test_sandbox_note_only_when_node_is_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    note = sandbox_note(argparse.Namespace(kernel="wasm"))
    assert "Node.js" in note and "doctor --fix" in note and "--kernel local" in note
    assert sandbox_note(argparse.Namespace(kernel="local")) == ""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node")
    assert sandbox_note(argparse.Namespace(kernel="wasm")) == ""


@pytest.mark.asyncio
async def test_startup_note_is_shown_in_the_chat():
    app = RlmAgentApp(startup_note="Node.js 18+ was not found")
    async with app.run_test(size=(150, 40)) as pilot:
        await pilot.pause()
        notices = [str(n.render()) for n in app.query_one(TranscriptView).query(NoticeBlock)]
        assert any("Node.js" in text for text in notices)


@pytest.mark.asyncio
async def test_no_note_no_notice():
    app = RlmAgentApp()
    async with app.run_test(size=(150, 40)) as pilot:
        await pilot.pause()
        assert not app.query_one(TranscriptView).query(NoticeBlock)


@pytest.mark.asyncio
async def test_prepare_sandbox_installs_on_first_use(monkeypatch):
    from rlmness import home, wasm_runtime

    from rlmagent.runtime import prepare_sandbox

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    ready = {"done": False}
    monkeypatch.setattr(wasm_runtime, "wasm_available", lambda: ready["done"])

    def fake_setup(print=print):
        ready["done"] = True
        return 0

    monkeypatch.setattr(home, "setup", fake_setup)
    said = []
    assert await prepare_sandbox(argparse.Namespace(kernel="wasm"), say=said.append) == ""
    assert ready["done"] and any("sandbox" in line for line in said)


@pytest.mark.asyncio
async def test_prepare_sandbox_reports_failure_and_never_switches_kernel(monkeypatch):
    from rlmness import home, wasm_runtime

    from rlmagent.runtime import prepare_sandbox

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(wasm_runtime, "wasm_available", lambda: False)
    monkeypatch.setattr(home, "setup", lambda print=print: 1)
    ns = argparse.Namespace(kernel="wasm")
    note = await prepare_sandbox(ns, say=lambda *a: None)
    assert "doctor --fix" in note and "--kernel local" in note
    assert ns.kernel == "wasm"


@pytest.mark.asyncio
async def test_prepare_sandbox_skips_local_kernel(monkeypatch):
    from rlmagent.runtime import prepare_sandbox

    monkeypatch.setattr("shutil.which", lambda name: None)
    assert await prepare_sandbox(argparse.Namespace(kernel="local")) == ""
