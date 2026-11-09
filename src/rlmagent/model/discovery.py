"""Remote model discovery and credential checks without generating a completion."""

from __future__ import annotations

import os

from rlmagent.config.loader import load_api_key
from rlmagent.config.store import load_config, provider_info
from rlmagent.model.transport.client import build_async_client


async def discover_models(provider: str, *, api_key=None, base_url=None, client=None) -> list[str]:
    if os.environ.get("RLM_AGENT_OFFLINE") == "1":
        raise ValueError("Model discovery is disabled in offline startup mode")
    info = provider_info(provider)
    headers = {}
    if provider == "copilot":
        from copilot import CopilotClient

        sdk = CopilotClient()
        try:
            await sdk.start()
            return [model.id for model in await sdk.list_models()]
        finally:
            await sdk.stop()
    if provider == "claude-subscription":
        import shutil
        if shutil.which("claude") is None:
            raise ValueError("Install and sign in to the official Claude Code CLI")
        return list(info.models)
    if provider in ("codex", "xai-subscription"):
        if provider == "codex":
            from rlmagent.model.subscription import resolve
        else:
            from rlmagent.model.registered_oauth import resolve

        if api_key is not None or base_url is not None:
            raise ValueError("Subscription discovery uses the signed-in account and fixed endpoint")
        credential = await resolve()
        url, key = credential.base_url, credential.api_key
    elif provider == "vertex":
        raise ValueError(
            "Remote model listing is unavailable for this cloud endpoint; "
            "enter a compatible model identifier"
        )
    else:
        config = load_config()
        url_var = {"anthropic": "ANTHROPIC_BASE_URL", "openai": "OPENAI_BASE_URL"}.get(
            provider, provider.upper() + "_BASE_URL"
        )
        if provider == "azure":
            url_var = "AZURE_OPENAI_BASE_URL"
        elif provider == "google":
            url_var = "GOOGLE_API_BASE"
        url = base_url or config.base_urls.get(provider) or os.environ.get(url_var) or info.base_url
        key = api_key or load_api_key(provider)
    if provider == "anthropic":
        if key:
            headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
        endpoint = url.rstrip("/") + (
            "/models" if url.rstrip("/").endswith("/v1") else "/v1/models"
        )
    elif provider in ("google", "azure"):
        if key:
            headers["x-goog-api-key" if provider == "google" else "api-key"] = key
        endpoint = url.rstrip("/") + "/models"
    else:
        if key:
            headers["Authorization"] = f"Bearer {key}"
        endpoint = url.rstrip("/") + "/models"
    owned = client is None
    client = client or build_async_client(timeout=20)
    try:
        response = await client.get(endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        if provider == "codex":
            return [row["slug"] for row in data.get("models", [])
                    if isinstance(row, dict) and row.get("visibility") == "list"
                    and isinstance(row.get("slug"), str)]
        if provider == "google":
            models = list(data.get("models", []))
            seen = set()
            while token := data.get("nextPageToken"):
                if token in seen or len(seen) >= 1000:
                    raise ValueError("Invalid model-catalog pagination")
                seen.add(token)
                page = await client.get(endpoint, headers=headers, params={"pageToken": token})
                page.raise_for_status()
                data = page.json()
                models.extend(data.get("models", []))
            return sorted({row["name"].removeprefix("models/") for row in models
                           if isinstance(row, dict) and isinstance(row.get("name"), str)})
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
    if provider == "claude-subscription":
        await discover_models(provider)
        return "Official CLI found; account and inference access have not been tested."
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
        client = client or build_async_client(timeout=20)
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
