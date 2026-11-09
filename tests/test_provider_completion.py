import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest


async def test_copilot_sdk_cannot_run_its_own_tools_or_load_project_config():
    pytest.importorskip("copilot")
    from rlmagent.harness.contracts.transcript import HumanEntry
    from rlmagent.model.copilot import CopilotProvider

    seen = {}

    class Session:
        def on(self, callback):
            self.callback = callback

        async def send_and_wait(self, prompt, **kw):
            assert "task" in prompt
            self.callback(
                SimpleNamespace(
                    type="assistant.usage",
                    data=SimpleNamespace(input_tokens=12, output_tokens=3, cache_read_tokens=4),
                )
            )
            return SimpleNamespace(data=SimpleNamespace(content="```python\nprint(42)\n```"))

        async def disconnect(self):
            seen["disconnected"] = True

    class Client:
        def __init__(self, **kw):
            seen["client"] = kw

        async def start(self):
            pass

        async def create_session(self, **kw):
            seen["session"] = kw
            assert kw["on_permission_request"](None, None).kind == "reject"
            return Session()

        async def stop(self):
            seen["stopped"] = True

    provider = CopilotProvider(client_factory=Client)
    events = [
        event
        async for event in provider.stream_response(
            model="auto",
            system="policy",
            messages=[HumanEntry(content="task")],
            tools=[SimpleNamespace(name="python")],
        )
    ]
    assert events[-1].type == "done" and events[-1].reason == "toolUse"
    assert events[-1].message.tool_calls[0].arguments == {"code": "print(42)\n"}
    assert events[-1].message.usage.input == 8
    assert seen["session"]["available_tools"] == [] and seen["session"]["mcp_servers"] == {}
    assert seen["session"]["enable_config_discovery"] is False
    assert seen["session"]["enable_file_hooks"] is False
    assert seen["stopped"] and seen["disconnected"]


async def test_registered_oauth_uses_only_the_supplied_public_client_and_rotates_tokens(
    tmp_path, monkeypatch
):
    from rlmagent.model import registered_oauth as auth

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    settings = {
        "client_id": "own-public-client",
        "authorization_endpoint": "https://accounts.x.ai/authorize",
        "token_endpoint": "https://accounts.x.ai/token",
        "scope": "inference offline_access",
        "resource": auth.BASE,
        "redirect_port": 0,
    }
    path = tmp_path / "client.json"
    path.write_text(json.dumps(settings))
    saved = None

    def store(action, value=None, *, namespace):
        nonlocal saved
        assert namespace == auth.NAME
        if action == "save":
            saved = value
        return saved

    def callback(endpoint, params, **kw):
        assert params["client_id"] == settings["client_id"]
        assert params["code_challenge_method"] == "S256"
        return {"code": "fixture-code"}, "http://127.0.0.1:12345/auth/callback"

    def serve(request):
        fields = parse_qs(request.content.decode())
        assert fields["client_id"] == [settings["client_id"]]
        if fields["grant_type"] == ["authorization_code"]:
            assert fields["code"] == ["fixture-code"] and fields["code_verifier"]
            access, refresh = "first", "refresh-first"
        else:
            assert fields["refresh_token"] == ["refresh-first"]
            access, refresh = "second", "refresh-second"
        return httpx.Response(
            200,
            json={
                "token_type": "Bearer",
                "access_token": access,
                "refresh_token": refresh,
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(auth, "_store", store)
    monkeypatch.setattr(auth, "_callback", callback)
    monkeypatch.setattr(
        auth,
        "build_async_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    )
    await auth.login(path)
    assert saved["access_token"] == "first"
    saved["expires_at"] = 0
    assert (await auth.resolve()).api_key == "second"
    assert saved["refresh_token"] == "refresh-second"
    await auth.logout()
    assert saved == {"registration": settings}


async def test_provider_scoped_cycles_restore_model_overrides_without_changing_route(
    tmp_path, monkeypatch
):
    from rlmagent.conversation import CodingSession
    from rlmagent.model.preferences import read
    from rlmagent.model.scripted import ReplayProvider

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))

    class Model(ReplayProvider):
        def set_service_tier(self, value):
            self.tier = value

    model = Model([])
    session = await CodingSession.create(
        provider=model, provider_name="fixture", model="a", tools=[], system="s"
    )
    try:
        await session.handle_command("/model cycle add a")
        await session.handle_command("/model cycle add b")
        await session.handle_command("/fast on")
        assert model.tier == "priority"
        await session.handle_command("/model cycle next")
        assert session.model == "b" and model.tier is None
        await session.handle_command("/model cycle next")
        assert session.model == "a" and model.tier == "priority"
        assert read("other-provider")["cycle"] == []
        assert session.provider_name == "fixture"
    finally:
        await session.shutdown()


async def test_offline_discovery_performs_no_network_request(monkeypatch):
    from rlmagent.model import discovery

    monkeypatch.setenv("RLM_AGENT_OFFLINE", "1")
    monkeypatch.setattr(
        discovery, "build_async_client", lambda **kw: pytest.fail("Unexpected network client")
    )
    with pytest.raises(ValueError, match="offline"):
        await discovery.discover_models("openai")


async def test_retry_headers_have_one_logical_key_and_distinct_attempts(monkeypatch):
    from rlmagent.agents.trace import TracedProvider, Tracer
    from rlmagent.model import oai_compatible
    from rlmagent.model.settings import Credential, OpenAIProfile, RetryPolicy

    headers = []

    def serve(request):
        headers.append(dict(request.headers))
        if len(headers) == 1:
            return httpx.Response(503, text="temporary")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"type":"response.completed","response":{"id":"r","status":"completed",'
                '"usage":{"input_tokens":3,"output_tokens":1},"output":[]}}\n\n'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    monkeypatch.setattr(oai_compatible, "build_async_client", lambda **kw: client)
    monkeypatch.setattr(oai_compatible, "compute_delay", lambda *a, **kw: 0)
    monkeypatch.setenv("RLM_AGENT_IDEMPOTENCY", "0")
    provider = oai_compatible.OpenAIProvider(
        OpenAIProfile(
            name="fixture",
            idempotency=True,
            credential=Credential("fixture", "https://fixture.test/v1"),
            endpoint="responses",
            retry=RetryPolicy(max_retries=1),
        )
    )
    tracer = Tracer()
    tree = SimpleNamespace(provider=provider, tracer=tracer)
    try:
        events = [
            event
            async for event in TracedProvider(tree, "root").stream_response(
                model="m", system="s", messages=[], tools=[]
            )
        ]
        assert events[-1].type == "done" and len(headers) == 2
        assert headers[0]["idempotency-key"] == headers[1]["idempotency-key"]
        assert [row["x-stainless-retry-count"] for row in headers] == ["0", "1"]
        assert headers[0]["x-rlmagent-attempt-id"] != headers[1]["x-rlmagent-attempt-id"]
        assert headers[0]["x-rlmagent-request-id"] == headers[1]["x-rlmagent-request-id"]
        assert len([row for row in tracer.records if row["kind"] == "model_attempt"]) == 2
    finally:
        await provider.close()
