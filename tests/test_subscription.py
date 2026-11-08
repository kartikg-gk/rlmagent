"""Subscription identity, permission and renewal boundaries."""

import json
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import urlopen

import httpx
import pytest

from rlmagent.model import subscription as auth
from rlmagent.model.settings import ConfigError


@pytest.fixture
def signing():
    jwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    public.update(kid="test-key", alg="RS256", use="sig")

    def token(**overrides):
        now = int(time.time())
        return jwt.encode(
            {
                "iss": auth.ISSUER,
                "aud": "oaiapp_test",
                "sub": "account",
                "exp": now + 300,
                "iat": now,
                "nonce": "test-nonce",
                **overrides,
            },
            key,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )

    return public, token


@pytest.mark.asyncio
async def test_identity_checks_signature_audience_nonce_and_expiry(signing):
    public, token = signing
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"keys": [public]}))
    ) as client:
        metadata = {"jwks_uri": auth.ISSUER + "/keys"}
        assert (await auth._identity(client, metadata, token(), "oaiapp_test", nonce="test-nonce"))[
            "sub"
        ] == "account"
        for invalid in (
            token(aud="another"),
            token(nonce="wrong"),
            token(exp=1),
            token(iss="https://invalid.example"),
        ):
            with pytest.raises(ConfigError, match="validation failed"):
                await auth._identity(client, metadata, invalid, "oaiapp_test", nonce="test-nonce")


def test_permission_and_token_shape_fail_closed():
    valid = {
        "access_token": "access",
        "refresh_token": "refresh",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": auth.SCOPES,
        "id_token": "identity",
    }
    for patch in (
        {"scope": "openid"},
        {"expires_in": float("nan")},
        {"access_token": "bad\r\nvalue"},
        {"token_type": "Other"},
    ):
        with pytest.raises(ConfigError):
            auth._tokens({**valid, **patch})
    old = auth._tokens(valid)
    rotated = auth._tokens(
        {
            "access_token": "new",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "rotated",
        },
        old,
    )
    assert rotated["refresh_token"] == "rotated" and rotated["scopes"] == old["scopes"]


def test_loopback_ignores_wrong_state_and_returns_exact_redirect(monkeypatch):
    results = []

    def browser(target):
        params = parse_qs(urlsplit(target).query)
        redirect = params["redirect_uri"][0]

        def callback():
            try:
                urlopen(redirect + "?state=wrong&code=ignored", timeout=3)
            except Exception:
                pass
            with urlopen(
                redirect
                + "?"
                + urlencode({"state": "expected", "code": "valid", "client_id": "oaiapp_test"}),
                timeout=3,
            ) as reply:
                results.append(reply.status)

        threading.Thread(target=callback, daemon=True).start()
        return True

    monkeypatch.setattr(auth.webbrowser, "open", browser)
    value, redirect = auth._callback(auth.ISSUER + "/authorize", {"state": "expected"}, timeout=5)
    assert value["code"] == "valid" and redirect.startswith("http://127.0.0.1:")
    assert redirect.endswith("/auth/callback")


def test_refresh_rotates_under_lock_and_preserves_account(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    saved = {
        "client_id": "oaiapp_test",
        "subject": "account",
        "host_id": "host",
        "refresh_token": "old",
        "access_token": "expired",
        "expires_at": 0,
        "scopes": auth.SCOPES.split(),
        "id_token": "identity",
    }

    def store(action, value=None):
        nonlocal saved
        if action == "save":
            saved = value
        return dict(saved)

    def transport(request):
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(
                200,
                json={
                    "issuer": auth.ISSUER,
                    "authorization_endpoint": auth.ISSUER + "/authorize",
                    "token_endpoint": auth.ISSUER + "/token",
                    "jwks_uri": auth.ISSUER + "/keys",
                },
            )
        data = parse_qs(request.content.decode())
        assert data["refresh_token"] == ["old"] and data["client_id"] == ["oaiapp_test"]
        assert "scope" not in data
        return httpx.Response(
            200,
            json={
                "access_token": "new",
                "refresh_token": "rotated",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(auth, "_store", store)
    monkeypatch.setattr(
        auth,
        "build_async_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    credential = auth._resolve_sync()
    assert credential.api_key == "new" and saved["refresh_token"] == "rotated"
    assert saved["subject"] == "account"
    assert auth._resolve_sync().api_key == "new"


@pytest.mark.asyncio
async def test_failed_revocation_clears_tokens_but_keeps_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    saved = {
        "client_id": "oaiapp_test",
        "subject": "account",
        "host_id": "host",
        "refresh_token": "secret",
        "access_token": "secret",
    }

    def store(action, value=None):
        nonlocal saved
        if action == "save":
            saved = value
        return dict(saved)

    async def fail(client):
        raise OSError("offline")

    monkeypatch.setattr(auth, "_store", store)
    monkeypatch.setattr(auth, "_metadata", fail)
    assert await auth.logout() is False
    assert saved == {"client_id": "oaiapp_test", "subject": "account", "host_id": "host"}


@pytest.mark.asyncio
async def test_discovery_rejects_foreign_token_destination():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "issuer": auth.ISSUER,
                    "authorization_endpoint": auth.ISSUER + "/authorize",
                    "jwks_uri": auth.ISSUER + "/keys",
                    "token_endpoint": "https://foreign.example/token",
                },
            )
        )
    ) as client:
        with pytest.raises(ConfigError, match="invalid endpoint"):
            await auth._metadata(client)


@pytest.mark.asyncio
async def test_subscription_discovery_uses_account_catalog(monkeypatch):
    from rlmagent.model.discovery import discover_models
    from rlmagent.model.settings import Credential

    async def credential():
        return Credential("fixture-subscription", auth.RESOURCE)

    monkeypatch.setattr(auth, "resolve", credential)

    def serve(request):
        assert str(request.url) == auth.RESOURCE + "/models"
        assert request.headers["Authorization"] == "Bearer fixture-subscription"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "second", "visibility": "list"},
                    {"slug": "hidden", "visibility": "hide"},
                    {"slug": "first", "visibility": "list"},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        assert await discover_models("codex", client=client) == ["second", "first"]
        with pytest.raises(ValueError):
            await discover_models("codex", client=client, base_url="https://other.example")


@pytest.mark.asyncio
async def test_subscription_uses_responses_without_storage_and_sends_tier(monkeypatch):
    from rlmagent.harness.contracts.transcript import HumanEntry
    from rlmagent.model import oai_compatible
    from rlmagent.model.settings import Credential, OpenAIProfile, RetryPolicy

    def serve(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert payload["stream"] is True and payload["store"] is False
        assert payload["service_tier"] == "priority"
        assert request.headers["Authorization"] == "Bearer fixture-subscription"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"type":"response.completed","response":{"id":"r","status":"completed",'
                '"output":[],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    monkeypatch.setattr(oai_compatible, "build_async_client", lambda **kw: client)

    async def credential():
        return Credential("fixture-subscription", auth.RESOURCE)

    provider = oai_compatible.OpenAIProvider(
        OpenAIProfile(
            name="codex",
            credential=Credential(None, auth.RESOURCE),
            resolver=credential,
            endpoint="responses",
            retry=RetryPolicy(max_retries=0),
            service_tier="priority",
        )
    )
    try:
        events = [
            event
            async for event in provider.stream_response(
                model="account-model", system="test", messages=[HumanEntry(content="go")], tools=[]
            )
        ]
        assert events[-1].type == "done"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_subscription_provider_builder_ignores_ambient_api_key(monkeypatch):
    from rlmagent.config.config import resolve_provider

    monkeypatch.setenv("OPENAI_API_KEY", "other-account")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example")
    provider = resolve_provider("codex")
    try:
        assert provider._profile.credential.api_key is None
        assert provider._profile.credential.base_url == auth.RESOURCE
        assert provider._profile.resolver is auth.resolve
        assert provider._profile.endpoint == "responses"
    finally:
        await provider.close()


def test_large_credential_records_commit_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    records = {}
    fail = False

    class Ring:
        def get_password(self, service, account):
            return records.get(account)

        def set_password(self, service, account, value):
            assert len(value) < 1000
            if fail and ":" not in account:
                raise OSError("manifest write failed")
            records[account] = value

        def delete_password(self, service, account):
            records.pop(account, None)

    monkeypatch.setattr(auth, "_dependencies", lambda: (None, Ring()))
    old = {"id_token": "fixture" * 2000, "access_token": "old"}
    auth._store("save", old)
    snapshot = dict(records)
    assert auth._store("load") == old
    fail = True
    with pytest.raises(ConfigError):
        auth._store("save", {"access_token": "new"})
    assert records == snapshot and auth._store("load") == old
    fail = False
    auth._store("save", {"access_token": "new"})
    assert auth._store("load") == {"access_token": "new"}
    assert len(records) == 2
    auth._store("delete")
    assert records == {}


@pytest.mark.asyncio
async def test_browser_login_validates_before_persisting(signing, tmp_path, monkeypatch):
    public, token = signing
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    saved = []
    attempt = {}

    def store(action, value=None):
        if action == "save":
            saved.append(value)
        return None

    def callback(endpoint, params, **kwargs):
        attempt.update(params)
        assert params["client_id"] == "dynamic_agent_client"
        assert params["code_challenge_method"] == "S256"
        assert params["agent_name_hint"] == "rlm-agent"
        return {
            "code": "fixture-code",
            "client_id": "oaiapp_test",
        }, "http://127.0.0.1:12345/auth/callback"

    def serve(request):
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(
                200,
                json={
                    "issuer": auth.ISSUER,
                    "authorization_endpoint": auth.ISSUER + "/authorize",
                    "token_endpoint": auth.ISSUER + "/token",
                    "jwks_uri": auth.ISSUER + "/keys",
                },
            )
        if request.url.path == "/keys":
            return httpx.Response(200, json={"keys": [public]})
        data = parse_qs(request.content.decode())
        assert data["redirect_uri"] == ["http://127.0.0.1:12345/auth/callback"]
        assert data["client_id"] == ["oaiapp_test"]
        import base64
        import hashlib

        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(data["code_verifier"][0].encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert challenge == attempt["code_challenge"]
        return httpx.Response(
            200,
            json={
                "access_token": "fixture-access",
                "refresh_token": "fixture-refresh",
                "token_type": "Bearer",
                "scope": auth.SCOPES,
                "expires_in": 3600,
                "id_token": token(nonce=attempt["nonce"]),
            },
        )

    monkeypatch.setattr(auth, "_store", store)
    monkeypatch.setattr(auth, "_callback", callback)
    monkeypatch.setattr(
        auth,
        "build_async_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    )
    await auth.login()
    assert len(saved) == 1 and saved[0]["subject"] == "account"
