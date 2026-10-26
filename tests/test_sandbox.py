import asyncio
import json

import pytest
from _scripted import code_turn, text_turn
from rlmness import Allowance

from rlmagent.agents.approval import ALWAYS, DENY, ONCE, Approver, Refused, Request, check
from rlmagent.agents.tree import AgentTree
from rlmagent.agents.workspace import Workspace
from rlmagent.conversation import CodingSession
from rlmagent.kernel.wasm import available
from rlmagent.model.scripted import ReplayProvider
from rlmagent.tools import build_tool_registry

needs_wasm = pytest.mark.skipif(not available(), reason="WebAssembly sandbox not set up")


# -- approval ---------------------------------------------------------------

async def test_policies_and_always_for_the_exact_command_only():
    with pytest.raises(Refused, match="needs approval"):
        await Approver("deny").approve(Request("run", "python -m pytest"))
    assert await Approver("auto").approve(Request("run", "python -m pytest"))
    with pytest.raises(Refused, match="destructive"):
        await Approver("auto").approve(Request("run", "rm -rf /"))
    asked = []

    async def ask(request):
        asked.append(request.target)
        return ALWAYS if request.target == "make test" else DENY
    approver = Approver("ask", ask)
    await approver.approve(Request("run", "make test"))
    await approver.approve(Request("run", "make test"))
    with pytest.raises(Refused, match="refused"):
        await approver.approve(Request("run", "make test && make deploy"))
    assert asked == ["make test", "make test && make deploy"]


def test_an_approval_covers_only_the_payload_it_was_given():
    approved = Request("write", "a.py", ("x = 1\n",)).digest
    check(approved, Request("write", "a.py", ("x = 1\n",)))
    with pytest.raises(Refused, match="changed after approval"):
        check(approved, Request("write", "a.py", ("import os; os.remove('b')\n",)))


async def test_workspace_functions_stay_inside_and_ask_before_changing(tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "a.py").write_text("x = 1\n")
    (tmp_path / "secret.txt").write_text("no")
    answers = iter([ONCE, DENY])

    async def ask(request):
        return next(answers)
    ws = Workspace(tmp_path / "proj", Approver("ask", ask))
    assert await ws.read_file("a.py") == "x = 1\n"
    assert await ws.list_files("*.py") == ["a.py"]
    with pytest.raises(ValueError, match="outside the workspace"):
        await ws.read_file("../secret.txt")
    await ws.edit_file("a.py", "x = 1", "x = 2")
    assert (tmp_path / "proj" / "a.py").read_text() == "x = 2\n"
    with pytest.raises(Refused, match="the user refused"):
        await ws.write_file("b.py", "y = 1\n")
    assert not (tmp_path / "proj" / "b.py").exists()
    result = await Workspace(tmp_path / "proj", Approver("auto")).run("echo hello")
    assert result["exit"] == 0 and "hello" in result["output"]


# -- the sandbox ------------------------------------------------------------

async def _tree(tmp_path, streams=(), policy="auto"):
    provider = ReplayProvider(list(streams))
    tree = AgentTree(provider=provider, provider_name="replay", model="m", cwd=str(tmp_path),
                     allowance=Allowance(max_cost=10.0), sessions_dir=None,
                     system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
                     kernel="wasm", approver=Approver(policy))
    await tree.start()
    return tree, provider


@needs_wasm
async def test_the_sandbox_sees_no_host_files_env_or_network(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_TEST_SECRET", "sk-should-not-leak")
    (tmp_path / "a.py").write_text("x = 1\n")
    tree, _ = await _tree(tmp_path)
    kernel = tree.root_kernel()
    try:
        out = await kernel.run(
            "import os\n"
            "print(os.environ.get('RLM_AGENT_TEST_SECRET'), os.path.exists('a.py'),"
            f" os.path.exists({str(tmp_path / 'a.py')!r}))\n"
            "try:\n    import js\nexcept ImportError:\n    print('no js')\n"
            "print(await read_file('a.py'))")
        assert out.error is None, out.error
        assert out.output.split() == ["None", "False", "False", "no", "js", "x", "=", "1"]
    finally:
        await tree.close()


@needs_wasm
async def test_writes_need_approval_inside_the_sandbox(tmp_path):
    tree, _ = await _tree(tmp_path, policy="deny")
    kernel = tree.root_kernel()
    try:
        out = await kernel.run("await write_file('b.py', 'y = 1')")
        assert "needs approval" in (out.error or "") and not (tmp_path / "b.py").exists()
        out = await kernel.run("import os\nprint(os.listdir('.'))")
        assert out.error is None
    finally:
        await tree.close()


@needs_wasm
async def test_a_sandboxed_tree_delegates_and_gets_final_back(tmp_path):
    tree, provider = await _tree(tmp_path, [
        code_turn("print(await rlm('double it', 21))"), text_turn("done"),
        code_turn("FINAL(CONTEXT * 2)"), text_turn("ok"),
    ])
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name="replay", model="m", system="s",
        tools=build_tool_registry(tree.root_kernel()), cwd=str(tmp_path))
    await session.set_name("t")
    session.agent_tree = tree
    tree.attach("root", session)
    try:
        async for _ in session.submit("delegate"):
            pass
        outputs = [m.text for m in session.transcript if getattr(m, "role", "") == "toolResult"]
        assert "42" in outputs[0], outputs
    finally:
        await session.shutdown()


@needs_wasm
async def test_files_named_in_the_prompt_reach_the_sandbox(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    tree, provider = await _tree(tmp_path, [
        code_turn("print(PROMPT['files'][0]['content'])"), text_turn("ok")])
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name="replay", model="m", system="s",
        tools=build_tool_registry(tree.root_kernel()), cwd=str(tmp_path))
    await session.set_name("t")
    session.agent_tree = tree
    tree.attach("root", session)
    try:
        async for _ in session.submit("fix calc.py"):
            pass
        outputs = [m.text for m in session.transcript if getattr(m, "role", "") == "toolResult"]
        assert "return a - b" in outputs[0], outputs
    finally:
        await session.shutdown()


def test_the_cli_defaults_to_the_sandbox():
    from rlmagent.cli.main import _build_run_parser

    ns = _build_run_parser().parse_args([])
    assert ns.kernel == "wasm" and ns.auto_approve is False
    assert json.dumps(_build_run_parser().parse_args(["--kernel", "local"]).kernel) == '"local"'


__all__ = ["asyncio"]
