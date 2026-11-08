import copy
import importlib
import json
from types import SimpleNamespace

import pytest
from _scripted import text_turn

from rlmagent.model.scripted import ReplayProvider
from rlmagent.runtime import build_session, make_agent_tree
from rlmagent.stops import of_session


@pytest.mark.parametrize(
    "model,requested,expected",
    [
        ("fixture", None, 200),
        ("acp:fixture", None, 50),
        ("fixture", 0, 0),
        ("acp:fixture", 0, 0),
        ("fixture", 3, 3),
    ],
)
def test_tree_defaults_apply_only_to_an_unspecified_call_budget(
    tmp_path, model, requested, expected
):
    tree = make_agent_tree(
        SimpleNamespace(max_calls=requested, kernel="local"),
        ReplayProvider([]),
        "fixture",
        model,
        None,
        cwd=str(tmp_path),
        backend=True,
    )
    assert tree.allowance.max_calls == expected


@pytest.mark.parametrize("engine", ["session", "rlmness"])
async def test_ordinary_cli_zero_budget_makes_no_provider_request(tmp_path, monkeypatch, engine):
    cli = importlib.import_module("rlmagent.cli.main")
    model = ReplayProvider([text_turn("This response must not be requested")])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli, "_resolve_provider", lambda name: model)
    ns = cli._build_run_parser().parse_args(
        [
            "-p",
            "--no-session",
            "--no-approve",
            "--provider",
            "openai",
            "--model",
            "fixture",
            "--engine",
            engine,
            "--kernel",
            "local",
            "--max-calls",
            "0",
            "--max-tool-output",
            "0",
        ]
    )
    session = await build_session(ns)
    try:
        async for _ in session.submit("Answer without exceeding the zero budget"):
            pass
        tree = session.agent_tree
        assert model.calls == [] and tree.allowance.calls == 0
        assert tree.policy.max_tool_output_bytes == 0
        assert of_session(session, tree.provider.refusal)["stop_reason"] == "max_calls"
    finally:
        await session.shutdown()


def test_saved_request_totals_survive_reopen_without_changing_runtime_budget(tmp_path):
    from rlmagent.agents.trace import Tracer
    from rlmagent.harness.contracts.transcript import UsageStats

    path = tmp_path / "requests.jsonl"
    first = Tracer()
    first.end(
        first.begin("root", "fixture"),
        UsageStats(input=10, output=5, cache_read=3, reasoning=2),
        "stop",
    )
    first.persist(path)
    reopened = Tracer()
    reopened.persist(path)
    reopened.persist(path)
    reopened.end(reopened.begin("child", "fixture"), UsageStats(input=2, output=1), "stop")
    result = reopened.lifetime_usage()
    assert result["totals"]["calls"] == 2 and result["totals"]["output"] == 6
    assert result["totals"]["cache_read"] == 3 and result["totals"]["reasoning"] == 2
    assert result["coverage"]["incompleteUsageRecords"] == 0
    assert sum(row["calls"] for row in reopened._totals.values()) == 1
    restored = Tracer()
    restored.persist(path)
    assert restored.lifetime_usage() == result


def test_legacy_and_duplicate_usage_records_remain_explicit_and_idempotent(tmp_path):
    from rlmagent.agents.trace import Tracer

    path = tmp_path / "requests.jsonl"
    row = {"kind": "call", "id": "one", "agent": "root", "input": 4, "output": 2, "cost": 0}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    tracer = Tracer()
    tracer.persist(path)
    result = tracer.lifetime_usage()
    assert result["totals"]["calls"] == 1 and result["totals"]["output"] == 2
    assert result["coverage"] == {
        "incompleteUsageRecords": 1,
        "unreportedUsageRecords": 1,
        "unpricedRecords": 1,
    }
    content = path.read_text() + json.dumps({**row, "output": 999}) + "\n"
    path.write_text(content)
    with pytest.raises(ValueError, match="conflicting"):
        Tracer().persist(path)
    assert path.read_text() == content


def account_document(tmp_path, provider):
    from test_contract import VALID

    document = copy.deepcopy(VALID)
    document["workdir"] = str(tmp_path)
    document["model"] = {
        "provider": provider,
        "name": "cli:claude-code?model=haiku" if provider == "claude-subscription" else "model-id",
        "api_key_env": None,
        "base_url": None,
    }
    return document


@pytest.mark.parametrize("provider", ["codex", "claude-subscription"])
def test_contract_accounts_require_explicit_account_credentials_and_fixed_routes(
    tmp_path, provider
):
    from rlmagent.acp.runtime import parse
    from rlmagent.contract import ContractError, parse_contract
    from rlmagent.contract_run import _provider
    from rlmagent.model.settings import Credential

    document = account_document(tmp_path, provider)
    contract = parse_contract(document)
    assert contract.api_key() is None
    runtime = parse({"contract": document, "credential": {"account": provider}})
    assert runtime.api_key is None and runtime.headers == ()
    with pytest.raises(ContractError, match="explicitly"):
        parse({"contract": document, "credential": {"api_key": "fixture", "headers": {}}})
    with pytest.raises(ContractError, match="injected"):
        _provider(contract, Credential("fixture", "https://unrelated.test"))
    for field, value in (("base_url", "https://unrelated.test"), ("api_key_env", "OTHER_KEY")):
        changed = copy.deepcopy(document)
        changed["model"][field] = value
        with pytest.raises(ContractError, match="null"):
            parse_contract(changed)


async def test_codex_contract_ignores_ambient_api_key_endpoint_and_transport(tmp_path, monkeypatch):
    from rlmagent.contract import parse_contract
    from rlmagent.contract_run import _provider

    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.test")
    monkeypatch.setenv("RLM_AGENT_CODEX_TRANSPORT", "websocket")
    provider = _provider(parse_contract(account_document(tmp_path, "codex")))
    try:
        assert provider._profile.credential.api_key is None
        assert provider._profile.credential.base_url == "https://api.openai.com/v1"
        assert provider._profile.endpoint == "responses" and provider._profile.transport == "http"
        assert provider._profile.resolver is not None
    finally:
        await provider.close()


def test_zero_output_policy_disables_the_cap_explicitly():
    from rlmagent.policy import cap_output

    content = "long output\n" * 2000
    assert cap_output(content, 0) == content
    assert len(cap_output(content, 1000)) < len(content)


async def test_snapshot_exposes_distinct_lifetime_and_runtime_scopes(tmp_path):
    from rlmagent.acp.server import Server
    from rlmagent.agents.trace import Tracer
    from rlmagent.conversation import CodingSession
    from rlmagent.harness.contracts.transcript import UsageStats

    path = tmp_path / "requests.jsonl"
    old = Tracer()
    old.end(old.begin("root", "fixture"), UsageStats(input=10, output=5), "stop")
    old.persist(path)
    current = Tracer()
    current.persist(path)
    current.end(current.begin("root", "fixture"), UsageStats(input=2, output=1), "stop")
    session = await CodingSession.create(
        provider=ReplayProvider([]),
        provider_name="fixture",
        model="fixture",
        system="base",
        tools=[],
    )
    session.agent_tree = SimpleNamespace(tracer=current, records={})
    try:
        value = Server(lambda: None, lambda value: None)._snapshot(session)
        assert value["treeTotals"]["calls"] == 1
        assert value["lifetimeTotals"]["calls"] == 2
        assert value["lifetimeTotals"]["output"] == 6
        assert value["lifetimeAccountingScope"] == "persistedRequestLedger"
        assert value["lifetimeCoverage"]["incompleteUsageRecords"] == 0
    finally:
        del session.agent_tree
        await session.shutdown()


async def test_real_session_resume_reloads_request_ledger_totals(tmp_path, monkeypatch):
    from rlmagent.acp.server import Server

    cli = importlib.import_module("rlmagent.cli.main")
    models = iter([ReplayProvider([text_turn("first")]), ReplayProvider([text_turn("second")])])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli, "_resolve_provider", lambda name: next(models))
    arguments = [
        "-p",
        "--no-approve",
        "--provider",
        "openai",
        "--model",
        "fixture",
        "--engine",
        "session",
        "--kernel",
        "local",
        "--session-dir",
        str(tmp_path / "sessions"),
    ]
    parser = cli._build_run_parser()
    first = await build_session(parser.parse_args(arguments))
    identity = first.session_id
    try:
        async for _ in first.submit("first task"):
            pass
    finally:
        await first.shutdown()
    resumed = await build_session(parser.parse_args([*arguments, "--resume", identity]))
    try:
        async for _ in resumed.submit("second task"):
            pass
        value = Server(lambda: None, lambda value: None)._snapshot(resumed)
        assert value["treeTotals"]["calls"] == 1
        assert value["lifetimeTotals"]["calls"] == 2
    finally:
        await resumed.shutdown()
