import copy
import json

from test_contract import VALID

from rlmagent_app.contract import parse_contract
from rlmagent_app.contract_run import build_contract_session
from rlmagent_model.scripted import ReplayProvider


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
    finally:
        await session.shutdown()


async def test_the_trace_starts_with_the_contract(tmp_path, monkeypatch):
    session = await _session(tmp_path, monkeypatch)
    await session.shutdown()
    lines = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert lines[0]["kind"] == "run" and lines[0]["contract"]["format"] == "rlmagent.run/1"
    assert lines[-1]["kind"] == "summary"


async def test_sub_agent_prompts_get_their_own_additions(tmp_path, monkeypatch):
    from rlmagent_app.agents.tree import AgentNode

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
    from rlmagent_app.cli.main import main

    bad = tmp_path / "run.json"
    doc = copy.deepcopy(VALID)
    del doc["limits"]["max_calls"]
    bad.write_text(json.dumps(doc))
    assert main(["--contract", str(bad), "-p", "hi"]) == 2
    assert "limits.max_calls" in capsys.readouterr().err


def test_cli_refuses_a_missing_key_variable(tmp_path, capsys, monkeypatch):
    from rlmagent_app.cli.main import main

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

    from rlmagent_app.cli.main import main

    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    path = tmp_path / "run.json"
    path.write_text(json.dumps(VALID))
    assert main(["--contract", str(path)]) == 2
    assert "needs a prompt" in capsys.readouterr().err
