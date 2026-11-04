"""Remote model discovery and credential checks without generating a completion."""

from __future__ import annotations

import os

import httpx

from rlmagent.config.loader import load_api_key
from rlmagent.config.store import load_config, provider_info


async def discover_models(provider: str, *, api_key=None, base_url=None, client=None) -> list[str]:
    info = provider_info(provider)
    headers = {}
    if provider == "vertex":
        raise ValueError(
            "Remote model listing is unavailable for this cloud endpoint; "
            "enter a compatible model identifier"
        )
    else:
        config = load_config()
        url_var = {"anthropic": "ANTHROPIC_BASE_URL", "openai": "OPENAI_BASE_URL"}.get(
            provider, provider.upper() + "_BASE_URL"
        )
        url = base_url or config.base_urls.get(provider) or os.environ.get(url_var) or info.base_url
        key = api_key or load_api_key(provider)
    if provider == "anthropic":
        if key:
            headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
        endpoint = url.rstrip("/") + (
            "/models" if url.rstrip("/").endswith("/v1") else "/v1/models"
        )
    else:
        if key:
            headers["Authorization"] = f"Bearer {key}"
        endpoint = url.rstrip("/") + "/models"
    owned = client is None
    client = client or httpx.AsyncClient(timeout=20)
    try:
        response = await client.get(endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        return sorted(
            {
                str(row["id"])
                for row in data.get("data", [])
                if isinstance(row, dict) and row.get("id")
            }
        )
    finally:
        if owned:
            await client.aclose()


async def check_provider(provider: str, *, base_url=None, client=None) -> str:
    if provider == "vertex":
        from rlmagent.model.vertex import ApplicationCredentials

        await ApplicationCredentials()()
        return "Application credentials refreshed; inference access has not been tested."
    if provider in ("openrouter", "huggingface"):
        key = load_api_key(provider)
        if not key:
            raise ValueError("No provider credential configured")
        config = load_config()
        if provider == "openrouter":
            url = base_url or config.base_urls.get(provider) or provider_info(provider).base_url
            endpoint = url.rstrip("/") + "/key"
        else:
            if base_url:
                raise ValueError(
                    "Credential testing is unavailable for a custom endpoint on this provider"
                )
            endpoint = "https://huggingface.co/api/whoami-v2"
        owned = client is None
        client = client or httpx.AsyncClient(timeout=20)
        try:
            response = await client.get(endpoint, headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
        finally:
            if owned:
                await client.aclose()
        return "Credential endpoint accepted authentication; inference access has not been tested."
    models = await discover_models(provider, base_url=base_url, client=client)
    return (
        f"Model endpoint reachable; {len(models)} models returned. "
        "Inference access has not been tested."
    )
