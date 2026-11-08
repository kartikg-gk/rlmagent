"""Public-client subscription sign-in with an app-owned credential record."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import secrets
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

from rlmagent.discovery import default_paths
from rlmagent.model.settings import ConfigError, Credential
from rlmagent.model.transport.client import build_async_client
from rlmagent.refinement.store import _lock

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PERMISSION = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PERMISSION}"


def _dependencies():
    try:
        import jwt
        import keyring
    except ImportError:
        raise ConfigError("Install rlmagent[subscription] to use subscription sign-in") from None
    return jwt, keyring


def _store(action: str, record: dict | None = None, *, namespace: str | None = None) -> dict | None:
    _, keyring = _dependencies()
    # Isolated app homes must never accidentally share the selected login.
    identity = str(default_paths().home.resolve()) + ("\0" + namespace if namespace else "")
    account = hashlib.sha256(identity.encode()).hexdigest()
    service = "rlmagent-subscription"

    def parts(manifest):
        if not isinstance(manifest, dict) or manifest.get("format") != "chunks-v1":
            return []
        generation, count = manifest["generation"], manifest["count"]
        if (
            uuid.UUID(hex=generation).hex != generation
            or type(count) is not int
            or not 1 <= count <= 128
        ):
            raise ValueError("Invalid credential manifest")
        return [f"{account}:{generation}:{index}" for index in range(count)]

    def cleanup(names):
        for name in names:
            try:
                keyring.delete_password(service, name)
            except Exception:
                pass

    try:
        raw = keyring.get_password(service, account)
        previous = json.loads(raw) if raw else None
        old_parts = parts(previous)
        if action == "load":
            if not old_parts:
                return previous
            chunks = [keyring.get_password(service, name) for name in old_parts]
            if any(chunk is None for chunk in chunks):
                raise ValueError("Credential record is incomplete")
            return json.loads("".join(chunks))
        if action == "save":
            encoded = json.dumps(record, ensure_ascii=True)
            chunks = [encoded[index : index + 768] for index in range(0, len(encoded), 768)]
            manifest = {"format": "chunks-v1", "generation": uuid.uuid4().hex, "count": len(chunks)}
            names = parts(manifest)
            try:
                for name, chunk in zip(names, chunks, strict=True):
                    keyring.set_password(service, name, chunk)
                keyring.set_password(service, account, json.dumps(manifest))
            except Exception:
                cleanup(names)
                raise
        elif raw is not None:
            keyring.delete_password(service, account)
        cleanup(old_parts)
    except Exception:
        raise ConfigError("Could not access the platform subscription credential store") from None
    return None


def _host_id() -> str:
    path = default_paths().home / "subscription-host"
    with _lock(path.with_suffix(".lock")):
        if path.exists():
            return str(uuid.UUID(path.read_text(encoding="utf-8").strip()))
        host = str(uuid.uuid4())
        path.write_text(host, encoding="utf-8")
        return host


async def _metadata(client) -> dict:
    reply = await client.get(f"{ISSUER}/.well-known/openid-configuration")
    reply.raise_for_status()
    data = reply.json()
    if data.get("issuer") != ISSUER:
        raise ConfigError("Subscription identity issuer did not match")
    for key in ("authorization_endpoint", "token_endpoint", "jwks_uri", "revocation_endpoint"):
        value = data.get(key)
        if key == "revocation_endpoint" and value is None:
            continue
        url = urlsplit(value or "")
        if (
            url.scheme != "https"
            or url.netloc != "auth.openai.com"
            or url.username
            or url.fragment
            or url.query
        ):
            raise ConfigError("Subscription discovery returned an invalid endpoint")
    return data


def _callback(
    endpoint: str,
    params: dict,
    *,
    timeout: float = 300,
    cancelled: threading.Event | None = None,
    port: int = 0,
) -> tuple[dict, str]:
    result: dict = {}

    class Callback(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(3)
            super().setup()

        def log_message(self, *_args):
            pass

        def do_GET(self):
            uri = urlsplit(self.path)
            query = parse_qs(uri.query, keep_blank_values=True)
            valid = (
                uri.path == "/auth/callback"
                and len(query.get("state", [])) == 1
                and secrets.compare_digest(query["state"][0], params["state"])
            )
            if valid and not result:
                if any(len(values) != 1 for values in query.values()):
                    result["error"] = "invalid_callback"
                else:
                    result.update({key: values[0] for key, values in query.items()})
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(b"Return to rlm-agent." if valid else b"Invalid login callback.")

    with HTTPServer(("127.0.0.1", port), Callback) as server:
        redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
        target = endpoint + "?" + urlencode({**params, "redirect_uri": redirect})
        if not webbrowser.open(target):
            raise ConfigError("Could not open the subscription sign-in browser")
        deadline = time.monotonic() + timeout
        while (
            not result
            and time.monotonic() < deadline
            and not (cancelled is not None and cancelled.is_set())
        ):
            server.timeout = min(1, max(0, deadline - time.monotonic()))
            server.handle_request()
    if not result or result.get("error") or not result.get("code"):
        raise ConfigError("Subscription sign-in was denied, invalid, or timed out")
    return result, redirect


async def _identity(
    client, metadata: dict, token: str, client_id: str, *, nonce: str | None = None
) -> dict:
    jwt, _ = _dependencies()
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") not in ("RS256", "ES256") or not header.get("kid"):
            raise ValueError()
        response = await client.get(metadata["jwks_uri"])
        response.raise_for_status()
        keys = [key for key in response.json()["keys"] if key.get("kid") == header["kid"]]
        if len(keys) != 1:
            raise ValueError()
        key = jwt.PyJWK.from_dict(keys[0], algorithm=header["alg"]).key
        claims = jwt.decode(
            token,
            key,
            algorithms=[header["alg"]],
            issuer=ISSUER,
            audience=client_id,
            leeway=5,
            options={"require": ["sub", "exp", "iat"]},
        )
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise ValueError()
        if nonce is not None and claims.get("nonce") != nonce:
            raise ValueError()
        return claims
    except Exception:
        raise ConfigError("Subscription identity token validation failed") from None


def _tokens(reply: dict, previous: dict | None = None) -> dict:
    old = previous or {}
    access = reply.get("access_token")
    refresh = reply.get("refresh_token", old.get("refresh_token"))
    scope = reply.get("scope", " ".join(old.get("scopes", [])))
    lifetime = reply.get("expires_in")
    if (
        reply.get("token_type", "").lower() != "bearer"
        or not isinstance(access, str)
        or not access
        or any(c.isspace() for c in access)
        or not isinstance(refresh, str)
        or not refresh
        or not isinstance(scope, str)
        or isinstance(lifetime, bool)
        or not isinstance(lifetime, (int, float))
        or not math.isfinite(lifetime)
        or lifetime <= 0
    ):
        raise ConfigError("Subscription token response was invalid")
    scopes = scope.split()
    if PERMISSION not in scopes:
        raise ConfigError("Sign-in did not grant ChatGPT plan usage")
    return {
        **old,
        "access_token": access,
        "refresh_token": refresh,
        "scopes": scopes,
        "expires_at": time.time() + lifetime,
        "id_token": reply.get("id_token", old.get("id_token")),
    }


async def login(*, new_account: bool = False) -> None:
    """Complete browser consent and validate identity before replacing saved credentials."""
    _dependencies()

    def selected():
        with _lock(default_paths().home / "subscription-refresh.lock"):
            return _store("load")

    old = None if new_account else await asyncio.to_thread(selected)
    host = await asyncio.to_thread(_host_id)
    verifier = secrets.token_urlsafe(64)
    nonce = secrets.token_urlsafe(32)
    params = {
        "client_id": old["client_id"] if old else "dynamic_agent_client",
        "ext_agent_host_id": f"urn:uuid:{host}",
        "response_type": "code",
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": secrets.token_urlsafe(32),
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode(),
    }
    if old:
        if old.get("id_token"):
            params["id_token_hint"] = old["id_token"]
    else:
        params["agent_name_hint"] = "rlm-agent"
    try:
        async with build_async_client(timeout=20, follow_redirects=False) as client:
            metadata = await _metadata(client)
            cancelled = threading.Event()
            try:
                callback, redirect = await asyncio.to_thread(
                    _callback, metadata["authorization_endpoint"], params, cancelled=cancelled
                )
            finally:
                cancelled.set()
            client_id = callback.get("client_id", old["client_id"] if old else "")
            if (
                not client_id
                or client_id == "dynamic_agent_client"
                or (old and client_id != old["client_id"])
            ):
                raise ConfigError("Subscription registration returned an invalid client ID")
            response = await client.post(
                metadata["token_endpoint"],
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": callback["code"],
                    "code_verifier": verifier,
                    "redirect_uri": redirect,
                    "resource": RESOURCE,
                },
            )
            response.raise_for_status()
            tokens = _tokens(response.json())
            claims = await _identity(client, metadata, tokens["id_token"], client_id, nonce=nonce)
            if old and claims["sub"] != old["subject"]:
                raise ConfigError("Subscription sign-in returned a different account")
            tokens.update(client_id=client_id, subject=claims["sub"], host_id=host)

            def save():
                with _lock(default_paths().home / "subscription-refresh.lock"):
                    _store("save", tokens)

            await asyncio.to_thread(save)
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("Subscription sign-in failed; saved credentials were retained") from None


def _resolve_sync() -> Credential:
    # Hold a cross-process lock across refresh and store to protect rotating tokens.
    with _lock(default_paths().home / "subscription-refresh.lock"):
        record = _store("load")
        if not record or not record.get("refresh_token"):
            raise ConfigError("Run rlm-agent provider login codex first")
        if record["expires_at"] <= time.time() + 60:

            async def renew():
                async with build_async_client(timeout=20, follow_redirects=False) as client:
                    metadata = await _metadata(client)
                    response = await client.post(
                        metadata["token_endpoint"],
                        data={
                            "grant_type": "refresh_token",
                            "client_id": record["client_id"],
                            "refresh_token": record["refresh_token"],
                            "resource": RESOURCE,
                        },
                    )
                    response.raise_for_status()
                    reply = response.json()
                    fresh = _tokens(reply, record)
                    if reply.get("id_token"):
                        identity = await _identity(
                            client, metadata, reply["id_token"], record["client_id"]
                        )
                        if identity["sub"] != record["subject"]:
                            raise ConfigError("Subscription refresh returned a different account")
                    return fresh

            try:
                record = asyncio.run(renew())
                _store("save", record)
            except Exception:
                raise ConfigError("Subscription refresh failed; sign in again") from None
        if PERMISSION not in record["scopes"]:
            raise ConfigError("Saved sign-in does not grant ChatGPT plan usage")
        return Credential(record["access_token"], RESOURCE)


async def resolve() -> Credential:
    return await asyncio.to_thread(_resolve_sync)


async def logout() -> bool:
    def revoke():
        with _lock(default_paths().home / "subscription-refresh.lock"):
            record = _store("load")
            confirmed = record is None or not record.get("refresh_token")

            async def request():
                async with build_async_client(timeout=20, follow_redirects=False) as client:
                    metadata = await _metadata(client)
                    response = await client.post(
                        metadata["revocation_endpoint"],
                        data={
                            "token": record["refresh_token"],
                            "token_type_hint": "refresh_token",
                            "client_id": record["client_id"],
                        },
                    )
                    response.raise_for_status()

            if record and record.get("refresh_token"):
                try:
                    asyncio.run(request())
                    confirmed = True
                except Exception:
                    pass
            if record:
                _store("save", {key: record[key] for key in ("client_id", "subject", "host_id")})
            return confirmed

    return await asyncio.to_thread(revoke)
