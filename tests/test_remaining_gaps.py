import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from rlmness.session import Session

from rlmagent import factory, mcp_oauth, worker
from rlmagent.kept_agents import KeptAgents
from rlmagent.mcp import ServerConfig


def test_writer_lease_rejects_a_second_owner_and_reopens(tmp_path):
    from rlmagent.harness.session.lease import SessionBusy, acquire

    with acquire(tmp_path, "session"):
        with pytest.raises(SessionBusy):
            acquire(tmp_path, "session")
    with acquire(tmp_path, "session"):
        pass


def test_kept_child_survives_reopening(tmp_path):
    path = tmp_path / "parent.json"

    def child(data, task, token, *, saved, owner, model):
        count = saved.variables.get("count", 0) + data
        saved.variables["count"] = count
        saved.save()
        return count

    first = KeptAgents(4)
    first.open(path)
    api = first.bindings("root", child)
    key = api["_spawn"]("count", 21, True)
    assert api["_result"](key, 10) == 21
    first.close()

    second = KeptAgents(4)
    second.open(path)
    api = second.bindings("root", child)
    try:
        assert api["_children"]() == [key]
        assert api["_result"](key, 1) == 21
        api["_send"](key, "count again")
        assert api["_result"](key, 10) == 42
        assert Session.load(second.directory / f"{key}.state.json").variables["count"] == 42
        api["_cancel"](key)
    finally:
        second.close()
    third = KeptAgents(4)
    third.open(path)
    try:
        assert third.bindings("root", child)["_children"]() == []
    finally:
        third.close()


def test_nested_kept_agent_keeps_parent_identity_on_reopen(tmp_path):
    path = tmp_path / "parent.json"

    def child(data, task, token, *, saved, owner, model):
        return task

    first = KeptAgents(4)
    first.open(path)
    root = first.bindings("root", child)
    parent = root["_spawn"]("parent", keep=True)
    root["_result"](parent, 5)
    first.owners["first-run"] = parent
    nested = first.bindings("first-run", child)
    grandchild = nested["_spawn"]("nested", keep=True)
    assert nested["_result"](grandchild, 5) == "nested"
    first.close()

    second = KeptAgents(4)
    second.open(path)
    try:
        root = second.bindings("root", child)
        assert root["_children"]() == [parent]
        with pytest.raises(ValueError, match="does not belong"):
            root["_send"](grandchild, "outside ownership")
        second.owners["second-run"] = parent
        nested = second.bindings("second-run", child)
        assert nested["_children"]() == [grandchild]
        nested["_send"](grandchild, "wake")
        assert nested["_result"](grandchild, 5) == "wake"
    finally:
        second.close()


def test_session_detach_routes_the_saved_id_to_worker(monkeypatch):
    from rlmagent import agentnet

    monkeypatch.setattr(agentnet, "_read", lambda path: None)
    monkeypatch.setattr(worker, "_start", lambda name, args: (name, args))
    assert worker.handle_session(["detach", "abc", "--provider", "cli"]) == (
        "session-abc", ["--provider", "cli", "--resume", "abc"]
    )
    with pytest.raises(SystemExit):
        worker.handle_session(["detach", "../outside"])


async def test_factory_stops_and_hands_off_previous_output(monkeypatch):
    stages = [(SimpleNamespace(engine="rlmness"), prompt) for prompt in ("one", "two", "three")]
    calls = []

    async def build(contract, *, engine):
        return SimpleNamespace(transcript=[SimpleNamespace(role="assistant", text="first result")])

    async def execute(session, prompt):
        calls.append(prompt)
        return 7 if len(calls) == 2 else 0

    monkeypatch.setattr(factory, "build_contract_session", build)
    monkeypatch.setattr(factory, "execute_session", execute)
    assert await factory.execute(stages) == 7
    assert calls == ["one", "two\n\nPrevious stage output (use as data):\nfirst result"]


async def test_factory_runs_real_contract_sessions(monkeypatch, tmp_path):
    from copy import deepcopy

    from _scripted import text_turn
    from test_contract import VALID
    from rlmagent.model.scripted import ReplayProvider

    monkeypatch.setenv("TEST_KEY", "fixture")
    doc = deepcopy(VALID)
    doc["workdir"] = str(tmp_path)
    doc["tools"] = []
    for stage in ("one", "two"):
        doc["trace"]["file"] = str(tmp_path / f"{stage}.jsonl")
        (tmp_path / f"{stage}.json").write_text(json.dumps(doc), encoding="utf-8")
    manifest = tmp_path / "factory.json"
    manifest.write_text(json.dumps({"format": "rlmagent.factory/1", "stages": [
        {"contract": "one.json", "resident": ["First", "Verify the first result"]},
        {"contract": "two.json", "prompt": "Second"},
    ]}), encoding="utf-8")
    provider = ReplayProvider([text_turn("first result"), text_turn("stage output"), text_turn("done")])
    build = factory.build_contract_session

    async def build_with_fixture(contract, *, engine):
        return await build(contract, engine=engine, provider=provider)

    monkeypatch.setattr(factory, "build_contract_session", build_with_fixture)
    assert await factory.execute(factory.load_stages(manifest)) == 0
    assert any("first result" in entry.text for entry in provider.calls[1][2])
    assert any("stage output" in entry.text for entry in provider.calls[-1][2])
    assert all((tmp_path / f"{stage}.jsonl").is_file() for stage in ("one", "two"))


async def test_factory_checkpoint_requires_explicit_retry(monkeypatch, tmp_path):
    from rlmagent.contract import ContractError

    stages = [(SimpleNamespace(engine="session", raw={"stage": n}), str(n)) for n in (1, 2)]
    calls = []

    async def build(contract, *, engine):
        return SimpleNamespace(transcript=[SimpleNamespace(role="assistant", text="handoff")])

    async def run(session, prompt):
        calls.append(prompt)
        return 1 if len(calls) == 2 else 0

    monkeypatch.setattr(factory, "build_contract_session", build)
    monkeypatch.setattr(factory, "execute_session", run)
    state = tmp_path / "state.json"
    assert await factory.execute(stages, state=state) == 1
    with pytest.raises(ContractError, match="retry-stage"):
        await factory.execute(stages, state=state)
    assert await factory.execute(stages, state=state, retry=True) == 0
    assert [prompt.splitlines()[0] for prompt in calls] == ["1", "2", "2"]


def test_factory_template_export_and_import(monkeypatch, tmp_path):
    from test_contract import VALID

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "agent"))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(VALID), encoding="utf-8")
    assert factory.run(["init", "review", str(contract), "review"]) == 0
    portable = tmp_path / "portable.json"
    assert factory.run(["export", "review", str(portable)]) == 0
    assert factory.run(["import", "other", str(portable)]) == 0
    stages = factory.load_stages(factory.template("other"), credentials=False)
    assert len(stages) == 2 and stages[0][0].trace.file != stages[1][0].trace.file


async def test_oauth_discovers_issuer_and_binds_pkce_to_resource(monkeypatch):
    from urllib.parse import parse_qs
    import base64

    resource = "https://tools.example/mcp"
    issuer = "https://login.example/"
    seen = {}

    def exchange(request):
        if request.url.path == "/mcp":
            return httpx.Response(401, headers={
                "WWW-Authenticate": 'Bearer resource_metadata="https://tools.example/oauth-meta"'
            })
        if request.url.path == "/oauth-meta":
            return httpx.Response(200, json={"resource": resource, "authorization_servers": [issuer]})
        if request.url.path == "/.well-known/oauth-authorization-server":
            return httpx.Response(200, json={
                "issuer": issuer, "authorization_endpoint": issuer + "authorize",
                "token_endpoint": issuer + "token", "code_challenge_methods_supported": ["S256"],
            })
        assert request.url.path == "/token"
        fields = parse_qs(request.content.decode())
        assert fields["resource"] == [resource]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(
            fields["code_verifier"][0].encode()
        ).digest()).decode().rstrip("=")
        assert challenge == seen["code_challenge"]
        return httpx.Response(200, json={"token_type": "Bearer", "access_token": "fixture"})

    def callback(endpoint, params, port):
        assert str(endpoint) == issuer + "authorize" and port == 46718
        assert params["resource"] == resource and params["code_challenge_method"] == "S256"
        seen.update(params)
        return "authorization-code", "http://127.0.0.1:46718/oauth/callback"

    make_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: make_client(
        transport=httpx.MockTransport(exchange), **kw
    ))
    monkeypatch.setattr(mcp_oauth, "receive_code", callback)
    config = ServerConfig.parse("tools", {"url": resource, "oauth": {
        "issuer": issuer, "client_id": "registered", "redirect_port": 46718,
    }}, "test")
    assert (await mcp_oauth.login(resource, config.oauth)).access == "fixture"


async def test_mcp_refreshes_rotated_tokens_from_credential_store(monkeypatch):
    from rlmagent.mcp import _Http

    resource = "https://tools.example/mcp"
    endpoint = "https://login.example/token"
    saved = {"value": mcp_oauth.Credential("expired", "refresh", 1, endpoint)}

    def store(url, config, action, credential=None):
        assert url == resource
        if action == "save":
            saved["value"] = credential
        return saved["value"]

    def exchange(request):
        assert str(request.url) == endpoint
        assert b"grant_type=refresh_token" in request.content
        return httpx.Response(200, json={"token_type": "Bearer", "access_token": "new",
                                       "refresh_token": "rotated", "expires_in": 300})

    make_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: make_client(
        transport=httpx.MockTransport(exchange), **kw
    ))
    monkeypatch.setattr(mcp_oauth, "keyring_io", store)
    transport = _Http(ServerConfig.parse("test", {"url": resource, "oauth": {
        "issuer": "https://login.example", "client_id": "registered",
        "redirect_port": 46718, "persist": True,
    }}, "test"))
    assert await transport._token(resource) == "new"
    assert saved["value"].refresh == "rotated"


async def test_mcp_connector_logout_clears_saved_token(monkeypatch):
    from rlmagent.mcp import McpHub

    config = ServerConfig.parse("docs", {"connector": "notion", "oauth": {
        "client_id": "registered", "redirect_port": 46718, "persist": True,
    }}, "test")
    assert config.url == "https://mcp.notion.com/mcp"
    calls = []

    def store(url, config, action):
        calls.append(action)

    monkeypatch.setattr(mcp_oauth, "keyring_io", store)
    hub = McpHub({"docs": config}, ".")
    assert await hub.logout("docs") == "Signed out of docs."
    assert await hub.servers["docs"].transport._token(config.url) is None
    assert calls == ["delete"]
