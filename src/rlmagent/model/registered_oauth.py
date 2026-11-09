"""OAuth for a user's registered xAI public client, without borrowed client IDs."""

import asyncio
import base64
import hashlib
import json
import math
import secrets
import threading
import time
from urllib.parse import urlsplit

from rlmagent.discovery import _resolve_home
from rlmagent.model.settings import ConfigError, Credential
from rlmagent.model.subscription import _callback, _store
from rlmagent.model.transport.client import build_async_client
from rlmagent.refinement.store import _lock

NAME = "xai-subscription"
BASE = "https://api.x.ai/v1"


def registration(value):
    required = {
        "client_id",
        "authorization_endpoint",
        "token_endpoint",
        "scope",
        "resource",
        "redirect_port",
    }
    if not isinstance(value, dict) or value.keys() != required:
        raise ConfigError("Registered OAuth client settings have missing or unknown fields")
    for key in ("client_id", "scope"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ConfigError("Registered OAuth client ID and scope must be nonempty")
    if type(value["redirect_port"]) is not int or not 0 <= value["redirect_port"] <= 65535:
        raise ConfigError("Invalid OAuth loopback port")
    for key in ("authorization_endpoint", "token_endpoint"):
        if not isinstance(value[key], str):
            raise ConfigError("Registered OAuth endpoints must be strings")
        uri = urlsplit(value[key])
        host = uri.hostname or ""
        allowed = host in ("x.ai", "grok.com") or host.endswith((".x.ai", ".grok.com"))
        if (
            not allowed
            or uri.scheme != "https"
            or uri.username
            or uri.password
            or uri.query
            or uri.fragment
        ):
            raise ConfigError("Registered OAuth endpoints must be pinned HTTPS provider endpoints")
    if value["resource"] is not None and value["resource"] != BASE:
        raise ConfigError("OAuth resource must match the public inference endpoint")
    return dict(value)


def _tokens(value, old):
    access = value.get("access_token")
    if (
        value.get("token_type", "").lower() != "bearer"
        or not isinstance(access, str)
        or not access
        or any(char.isspace() for char in access)
    ):
        raise ConfigError("Provider returned an invalid access token")
    lifetime = value.get("expires_in")
    if lifetime is not None and (
        type(lifetime) not in (int, float) or not math.isfinite(lifetime) or lifetime <= 0
    ):
        raise ConfigError("Provider returned an invalid token lifetime")
    refresh = value.get("refresh_token", old.get("refresh_token"))
    if refresh is not None and (not isinstance(refresh, str) or not refresh):
        raise ConfigError("Provider returned an invalid refresh token")
    return {
        "registration": old["registration"],
        "access_token": access,
        "refresh_token": refresh,
        "expires_at": time.time() + lifetime if lifetime is not None else None,
    }


async def login(path=None):
    def selected():
        with _lock(_resolve_home() / "xai-oauth.lock"):
            return _store("load", namespace=NAME)

    old = await asyncio.to_thread(selected)
    if path is not None:
        settings = registration(json.loads(path.read_text(encoding="utf-8")))
    elif old:
        settings = registration(old["registration"])
    else:
        raise ConfigError(
            "Supply --client-registration with your provider-issued public client settings"
        )
    verifier = secrets.token_urlsafe(64)
    params = {
        "client_id": settings["client_id"],
        "response_type": "code",
        "scope": settings["scope"],
        "state": secrets.token_urlsafe(32),
        "code_challenge_method": "S256",
        "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode(),
    }
    if settings["resource"]:
        params["resource"] = settings["resource"]
    cancelled = threading.Event()
    try:
        callback, redirect = await asyncio.to_thread(
            _callback,
            settings["authorization_endpoint"],
            params,
            cancelled=cancelled,
            port=settings["redirect_port"],
        )
        if callback.get("client_id", settings["client_id"]) != settings["client_id"]:
            raise ConfigError("Provider returned a different registered client")
        fields = {
            "grant_type": "authorization_code",
            "client_id": settings["client_id"],
            "code": callback["code"],
            "code_verifier": verifier,
            "redirect_uri": redirect,
        }
        if settings["resource"]:
            fields["resource"] = settings["resource"]
        async with build_async_client(timeout=20, follow_redirects=False) as client:
            reply = await client.post(settings["token_endpoint"], data=fields)
            reply.raise_for_status()
            record = _tokens(reply.json(), {"registration": settings})

        def save():
            with _lock(_resolve_home() / "xai-oauth.lock"):
                _store("save", record, namespace=NAME)

        await asyncio.to_thread(save)
    except ConfigError:
        raise
    except Exception:
        raise ConfigError(
            "Registered OAuth sign-in failed; saved credentials were retained"
        ) from None
    finally:
        cancelled.set()


def _resolve():
    with _lock(_resolve_home() / "xai-oauth.lock"):
        record = _store("load", namespace=NAME)
        if not record or not record.get("access_token"):
            raise ConfigError(
                "Run provider login xai-subscription with your registered client settings"
            )
        settings = registration(record["registration"])
        if record["expires_at"] is not None and record["expires_at"] < time.time() + 60:
            if not record["refresh_token"]:
                raise ConfigError("Registered OAuth login expired; sign in again")

            async def refresh():
                fields = {
                    "grant_type": "refresh_token",
                    "client_id": settings["client_id"],
                    "refresh_token": record["refresh_token"],
                }
                if settings["resource"]:
                    fields["resource"] = settings["resource"]
                async with build_async_client(timeout=20, follow_redirects=False) as client:
                    reply = await client.post(settings["token_endpoint"], data=fields)
                    reply.raise_for_status()
                    return _tokens(reply.json(), record)

            try:
                record = asyncio.run(refresh())
                _store("save", record, namespace=NAME)
            except Exception:
                raise ConfigError("Registered OAuth refresh failed; sign in again") from None
        return Credential(record["access_token"], BASE)


async def resolve():
    return await asyncio.to_thread(_resolve)


async def logout():
    def clear():
        with _lock(_resolve_home() / "xai-oauth.lock"):
            record = _store("load", namespace=NAME)
            if record:
                _store("save", {"registration": record["registration"]}, namespace=NAME)

    await asyncio.to_thread(clear)
