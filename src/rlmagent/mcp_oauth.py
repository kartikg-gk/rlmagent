"""Browser login for explicitly configured public MCP clients."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import asdict, dataclass, field
from urllib.parse import parse_qs, urlsplit

import httpx

from rlmagent.mcp import McpError


@dataclass
class Credential:
    access: str = field(repr=False)
    refresh: str | None = field(repr=False)
    expires: float | None
    endpoint: str
    audience: str | None = None


def _credential(reply, endpoint: str, previous: str | None = None,
                audience: str | None = None) -> Credential:
    access = reply["access_token"]
    if (reply.get("token_type", "").lower() != "bearer" or not isinstance(access, str)
            or not access or any(ch.isspace() for ch in access)):
        raise McpError("MCP authorization server returned an invalid bearer token")
    refresh = reply.get("refresh_token", previous)
    if refresh is not None and (not isinstance(refresh, str) or not refresh):
        raise McpError("MCP authorization server returned an invalid refresh token")
    seconds = reply.get("expires_in")
    lifetime = float(seconds) if seconds is not None else None
    if lifetime is not None and (not math.isfinite(lifetime) or lifetime < 0):
        raise McpError("MCP authorization server returned an invalid token lifetime")
    return Credential(access, refresh, time.time() + lifetime if lifetime is not None else None,
                      endpoint, audience)


def _account(resource: str, config: dict) -> str:
    return hashlib.sha256(json.dumps([
        resource, config["issuer"], config["client_id"], config.get("scope", ""),
    ]).encode()).hexdigest()


def keyring_io(resource: str, config: dict, action: str, credential=None):
    """Use the platform credential store when persistence is explicitly enabled."""
    if not config.get("persist"):
        return None
    try:
        import keyring
        from keyring.errors import KeyringError
    except ImportError:
        raise McpError("Persistent MCP login requires the mcp-keyring extra") from None
    service, account = "rlmagent-mcp", _account(resource, config)
    try:
        if action == "load":
            stored = keyring.get_password(service, account)
            return Credential(**json.loads(stored)) if stored else None
        if action == "save":
            keyring.set_password(service, account, json.dumps(asdict(credential)))
        elif keyring.get_password(service, account) is not None:
            keyring.delete_password(service, account)
    except (KeyringError, ValueError, TypeError, OSError):
        raise McpError("The platform credential store could not access the MCP login") from None


async def refresh(resource: str, config: dict, current: Credential) -> Credential:
    if not current.refresh:
        raise McpError("MCP login expired. Sign in again.")
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            response = await client.post(https_url(current.endpoint), data={
                "grant_type": "refresh_token", "refresh_token": current.refresh,
                **_client_fields(config), "resource": current.audience or str(https_url(resource)),
            })
            return _credential(response.raise_for_status().json(), current.endpoint,
                               current.refresh, current.audience)
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
        raise McpError("MCP login could not be refreshed. Sign in again.") from None


def https_url(value: str) -> httpx.URL:
    url = httpx.URL(value)
    if url.scheme != "https" or not url.host or url.userinfo or url.fragment:
        raise McpError("MCP OAuth endpoints must use HTTPS without embedded credentials")
    return url.copy_with(path="/") if url.path == "/" and not url.query else url


def _client_fields(config: dict) -> dict:
    client_id = config["client_id"]
    if not client_id:
        raise McpError("MCP OAuth client ID is unset")
    fields = {"client_id": client_id}
    if name := config.get("client_secret_env"):
        secret = os.environ.get(name)
        if not secret:
            raise McpError(f"MCP OAuth client secret environment variable {name} is unset")
        fields["client_secret"] = secret
    return fields


def receive_code(endpoint, params: dict, port: int) -> tuple[str, str]:
    state = params["state"]
    result = {}

    class Callback(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(5)
            super().setup()

        def log_message(self, *_args):
            pass

        def do_GET(self):
            request = urlsplit(self.path)
            query = parse_qs(request.query)
            supplied = query.get("state", [""])[0]
            valid = request.path == "/oauth/callback" and secrets.compare_digest(
                supplied.encode(), state.encode()
            )
            if valid:
                result["code"] = query.get("code", [""])[0]
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Return to rlm-agent." if valid else b"Invalid login callback.")

    with HTTPServer(("127.0.0.1", port), Callback) as server:
        redirect = f"http://127.0.0.1:{server.server_port}/oauth/callback"
        target = endpoint.copy_merge_params({**params, "redirect_uri": redirect})
        if not webbrowser.open(str(target)):
            raise McpError("Could not open a browser for MCP login")
        deadline = time.monotonic() + 120
        while not result and time.monotonic() < deadline:
            server.timeout = max(0, deadline - time.monotonic())
            server.handle_request()
    if not result.get("code"):
        raise McpError("MCP login was denied or timed out")
    return result["code"], redirect


async def login(resource: str, config: dict) -> Credential:
    resource_url = https_url(resource)
    issuer = https_url(config["issuer"])
    if issuer.query:
        raise McpError("MCP OAuth issuer must be a URL without a query")
    # Confirm the server advertises the configured issuer before requesting credentials.
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        try:
            response = await client.post(resource_url, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "rlm-agent", "version": "1"}},
            }, headers={"Accept": "application/json, text/event-stream"})
            challenge = response.headers.get("www-authenticate", "")
            match = re.search(r'\bresource_metadata="([^"]+)"', challenge, re.I)
            metadata_url = (https_url(match[1]) if match else resource_url.copy_with(
                path="/.well-known/oauth-protected-resource" + resource_url.path.rstrip("/"),
                query=None,
            ))
            if (metadata_url.host, metadata_url.port) != (resource_url.host, resource_url.port):
                raise McpError("MCP resource metadata must be on the configured server origin")
            metadata_reply = await client.get(metadata_url)
            if metadata_reply.status_code == 404 and not match:
                metadata_reply = await client.get(resource_url.copy_with(
                    path="/.well-known/oauth-protected-resource", query=None,
                ))
            metadata = metadata_reply.raise_for_status().json()
            audience = https_url(metadata["resource"])
            origin = resource_url.copy_with(path="/", query=None)
            issuers = [https_url(value) for value in metadata.get("authorization_servers", [])]
            if audience not in (resource_url, origin) or issuer not in issuers:
                raise McpError("MCP server metadata did not confirm the configured issuer and resource")
            auth_metadata = issuer.copy_with(
                path="/.well-known/oauth-authorization-server" + issuer.path.rstrip("/")
            )
            provider = (await client.get(auth_metadata)).raise_for_status().json()
            if https_url(provider["issuer"]) != issuer:
                raise McpError("MCP authorization metadata returned a different issuer")
            if "S256" not in provider.get("code_challenge_methods_supported", []):
                raise McpError("MCP authorization server did not advertise S256 PKCE")
            authorize = https_url(provider["authorization_endpoint"])
            token = https_url(provider["token_endpoint"])
            verifier = secrets.token_urlsafe(48)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            fields = {"client_id": config["client_id"], "resource": str(audience)}
            client_fields = _client_fields(config)
            params = {**fields, "response_type": "code", "state": secrets.token_urlsafe(32),
                      "code_challenge_method": "S256",
                      "code_challenge": challenge.decode().rstrip("=")}
            if config.get("scope"):
                params["scope"] = config["scope"]
            code, redirect = await asyncio.to_thread(
                receive_code, authorize, params, config["redirect_port"]
            )
            reply = (await client.post(token, data={
                **fields, **client_fields, "grant_type": "authorization_code", "code": code,
                "redirect_uri": redirect, "code_verifier": verifier,
            })).raise_for_status().json()
            return _credential(reply, str(token), audience=str(audience))
        except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError, OSError):
            raise McpError("MCP OAuth discovery or token exchange failed") from None
