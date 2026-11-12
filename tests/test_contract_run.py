import copy
import json

import pytest
from _scripted import text_turn
from test_contract import VALID

from rlmagent.contract import parse_contract
from rlmagent.contract_run import build_contract_session
from rlmagent.model.scripted import ReplayProvider


async def test_contract_new_token_budget_counts_uncached_usage_and_refuses_next_call(
        tmp_path, monkeypatch):
    from rlmagent.agents.budget import BudgetedProvider
    from rlmagent.harness.contracts.transcript.diagnostics import UsageStats

    # This test is about settling measured usage; sizing requests up front is below.
    monkeypatch.setattr(BudgetedProvider, "request_size", lambda self, system, messages: 0)

    first = text_turn("first")
    first[0].message.usage = UsageStats(input=2, output=3, cache_read=100, total_tokens=105)
    provider = ReplayProvider([first, text_turn("must not run")])
    session = await build_contract_session(
        _contract(tmp_path, tools=[], limits__max_new_tokens=5), provider=provider,
    )
    try:
        for prompt in ("first", "second"):
            async for _ in session.submit(prompt):
                pass
        assert len(provider.calls) == 1
        assert session.agent_tree.allowance.new_token_budget.used == 5
        assert "5 new tokens" in session.agent_tree.provider.refusal
    finally:
        await session.shutdown()



async def test_a_request_that_could_pass_the_budget_is_not_sent(tmp_path):
    provider = ReplayProvider([text_turn("must not run")])
    session = await build_contract_session(
        _contract(tmp_path, tools=[], limits__max_new_tokens=1000), provider=provider,
    )
    try:
        async for _ in session.submit("hello"):
            pass
        assert provider.calls == []
        refusal = session.agent_tree.provider.refusal
        assert "the next request could use about" in refusal and "0 of 1000" in refusal
    finally:
        await session.shutdown()


def test_request_size_counts_the_prompt_and_the_longest_reply():
    from types import SimpleNamespace

    from rlmagent.agents.budget import estimate_prompt_tokens, reply_limit

    assert estimate_prompt_tokens("a" * 30, [SimpleNamespace(text="b" * 30)]) == 21
    inner = SimpleNamespace(_profile=SimpleNamespace(max_tokens=4000))
    assert reply_limit(SimpleNamespace(inner=inner)) == 4000
    assert reply_limit(SimpleNamespace()) == 8192

def _contract(tmp_path, **changes):
    doc = copy.deepcopy(VALID)
    doc["workdir"] = str(tmp_path)
    doc["trace"]["file"] = str(tmp_path / "trace.jsonl")
    for path, value in changes.items():
        *parents, last = path.split("__")
        target = doc
        for key in parents:
            target = target[key]
        target[last] = value
    return parse_contract(doc)


async def _session(tmp_path, monkeypatch, **changes):
    monkeypatch.setenv("TEST_KEY", "k")
    return await build_contract_session(_contract(tmp_path, **changes), provider=ReplayProvider([]))


async def test_contract_run_uses_only_what_the_contract_says(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("PROJECT-SECRET-INSTRUCTIONS")
    monkeypatch.setenv("RLM_AGENT_MODEL", "some-other-model")
    session = await _session(tmp_path, monkeypatch, tools=["python", "Read"],
                             limits__max_calls=7, prompt__append="APPENDED")
    try:
        assert session.model == "deepseek/deepseek-v4-flash"
        assert [t.name for t in session.tools] == ["Read", "python"]
        assert "PROJECT-SECRET-INSTRUCTIONS" not in session.harness.settings.system
        assert "APPENDED" in session.harness.settings.system and "gather_rlm" in session.harness.settings.system
        assert session.agent_tree.allowance.max_calls == 7
        with pytest.raises(RuntimeError, match="model specified by the runtime contract"):
            session.agent_tree.spawn("root", "change provider", None, model="cli:claude-code")
    finally:
        await session.shutdown()


async def test_the_trace_starts_with_the_contract(tmp_path, monkeypatch):
    session = await _session(tmp_path, monkeypatch)
    await session.shutdown()
    lines = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert lines[0]["kind"] == "run" and lines[0]["contract"]["format"] == "rlmagent.run/1"
    assert lines[-1]["kind"] == "summary"


async def test_sub_agent_prompts_get_their_own_additions(tmp_path, monkeypatch):
    from rlmagent.agents.tree import AgentNode

    session = await _session(tmp_path, monkeypatch, limits__max_depth=2,
                             prompt__append_sub_agent="MID", prompt__append_leaf="LEAF")
    try:
        system_for = session.agent_tree.system_for
        mid = system_for(AgentNode(id="root.1", depth=1, parent_id="root"), [])
        leaf = system_for(AgentNode(id="root.1.2", depth=2, parent_id="root.1"), [])
        assert "MID" in mid and "LEAF" not in mid
        assert "LEAF" in leaf and "MID" not in leaf
    finally:
        await session.shutdown()


async def test_system_file_replaces_the_identity(tmp_path, monkeypatch):
    (tmp_path / "role.txt").write_text("You grade pull requests.")
    session = await _session(tmp_path, monkeypatch, prompt__system_file=str(tmp_path / "role.txt"))
    try:
        assert session.harness.settings.system.startswith("You grade pull requests.")
        assert "You are rlm-agent" not in session.harness.settings.system
    finally:
        await session.shutdown()


def test_cli_refuses_a_bad_contract_before_running(tmp_path, capsys):
    from rlmagent.cli.main import main

    bad = tmp_path / "run.json"
    doc = copy.deepcopy(VALID)
    del doc["limits"]["max_calls"]
    bad.write_text(json.dumps(doc))
    assert main(["--contract", str(bad), "-p", "hi"]) == 2
    assert "limits.max_calls" in capsys.readouterr().err


def test_cli_refuses_a_missing_key_variable(tmp_path, capsys, monkeypatch):
    from rlmagent.cli.main import main

    monkeypatch.delenv("TEST_KEY", raising=False)
    doc = copy.deepcopy(VALID)
    doc["workdir"] = str(tmp_path)
    doc["trace"]["file"] = str(tmp_path / "t.jsonl")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    assert main(["--contract", str(path), "-p", "hi"]) == 2
    assert "TEST_KEY" in capsys.readouterr().err


def test_cli_contract_needs_a_prompt(tmp_path, capsys, monkeypatch):
    import io

    from rlmagent.cli.main import main

    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    path = tmp_path / "run.json"
    path.write_text(json.dumps(VALID))
    assert main(["--contract", str(path)]) == 2
    assert "needs a prompt" in capsys.readouterr().err


def test_the_kernel_section_is_optional_and_strict(tmp_path):
    import pytest
    from rlmagent.contract import ContractError

    assert _contract(tmp_path).kernel.kind == "local"
    assert _contract(tmp_path).kernel.approve == "deny"
    c = _contract(tmp_path, kernel={"kind": "wasm", "approve": "auto", "network": True}, tools=["python", "Read"])
    assert (c.kernel.kind, c.kernel.approve, c.kernel.network) == ("wasm", "auto", True)
    for bad, named in [({"kind": "docker", "approve": "deny", "network": False}, "kernel.kind"),
                       ({"kind": "wasm", "approve": "ask", "network": False}, "kernel.approve"),
                       ({"kind": "wasm"}, "kernel.approve"),
                       ({"kind": "wasm", "approve": "deny", "network": False, "x": 1}, "kernel.x")]:
        with pytest.raises(ContractError, match=named):
            _contract(tmp_path, kernel=bad)
    with pytest.raises(ContractError, match="Write and Edit"):
        _contract(tmp_path, kernel={"kind": "wasm", "approve": "deny", "network": False})


async def test_a_wasm_contract_runs_sandboxed(tmp_path, monkeypatch):
    import pytest
    from rlmagent.kernel.wasm import WasmKernel, available

    if not available():
        pytest.skip("sandbox not set up")
    session = await _session(tmp_path, monkeypatch, tools=["python", "Read"],
                             kernel={"kind": "wasm", "approve": "deny", "network": False})
    try:
        tree = session.agent_tree
        assert isinstance(tree.root_kernel(), WasmKernel) and tree.approver.policy == "deny"
        assert "read_file" in session.harness.settings.system
        assert json.loads(json.dumps(tree.kernel_kind)) == "wasm"
    finally:
        await session.shutdown()


async def test_strict_budget_is_shared_by_recursive_node_calls_and_compaction(tmp_path):
    from rlmagent.harness.contracts.transcript import HumanEntry, ModelEntry, TextSegment
    from rlmagent.harness.contracts.transcript.diagnostics import UsageStats
    from rlmagent.kernel.wasm import available
    from rlmagent.model.token_bounds import TokenReservation

    if not available():
        pytest.skip("Node and Pyodide are unavailable")

    class CountedFixture(ReplayProvider):
        def __init__(self, turns):
            super().__init__(turns)
            self.balances = []

        async def prepare_token_request(self, *, remaining, **request):
            self.balances.append(remaining)
            return TokenReservation(3, min(4, remaining - 3), lambda: self.stream_response(**request))

    turns = [
        text_turn("```python\nvalue = await rlm('21', instruction='double the input')\nFINAL(value)\n```"),
        text_turn("```python\nFINAL(int(PROMPT) * 2)\n```"),
        text_turn("fixture compaction summary"),
    ]
    for turn in turns:
        turn[0].message.usage = UsageStats(input=3, output=2, total_tokens=5)
    provider = CountedFixture(turns)
    session = await build_contract_session(
        _contract(tmp_path, tools=["python"], limits__max_new_tokens=15,
                  limits__strict_new_tokens=True,
                  kernel={"kind": "wasm", "approve": "deny", "network": False}),
        provider=provider, engine="rlmness",
    )
    try:
        async for _ in session.submit("compute through a child"):
            pass
        assert session.transcript[-1].text == "42"
        messages = [message for index in range(12) for message in (
            HumanEntry(content=f"turn {index}"),
            ModelEntry(content=[TextSegment(text=f"answer {index}")]))]
        session.harness.set_messages(messages)
        summary = await session.compact()
        assert "fixture compaction summary" in summary.summary
        assert provider.balances == [15, 10, 5]
        assert session.agent_tree.allowance.new_token_budget.used == 15
        async for _ in session.submit("this request cannot start"):
            pass
        assert len(provider.calls) == 3
    finally:
        await session.shutdown()
