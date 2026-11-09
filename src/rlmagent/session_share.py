"""Review-bound publication of an explicitly selected session export."""

import hashlib
import json
import secrets

import httpx

from rlmagent.model.transport.client import build_async_client


def prepare(session_id, html, jsonl):
    body = {
        "description": f"Session {session_id}",
        "public": False,
        "files": {
            "session.html": {"content": html},
            "session.jsonl": {"content": jsonl},
        },
    }
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    return body, hashlib.sha256(canonical).hexdigest()


async def publish(body, *, expected_digest, token, client=None):
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if not expected_digest or not secrets.compare_digest(digest, expected_digest):
        raise ValueError("Review the export and supply its exact SHA-256 before publishing")
    if not token or any(char.isspace() for char in token):
        raise ValueError("Set GH_TOKEN or GITHUB_TOKEN locally to publish")
    owned = client is None
    client = client or build_async_client(timeout=30, follow_redirects=False)
    try:
        reply = await client.post(
            "https://api.github.com/gists",
            json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        if reply.status_code != 201:
            raise ValueError(
                f"Gist publication failed (HTTP {reply.status_code}); no automatic retry"
            )
        return reply.json()["html_url"]
    except httpx.HTTPError:
        raise ValueError(
            "Publication could not be confirmed; no automatic retry was made"
        ) from None
    finally:
        if owned:
            await client.aclose()
