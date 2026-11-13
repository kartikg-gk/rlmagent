"""Provider resolution, catalog, and environment-based auto-detection."""

from __future__ import annotations

import os
import sys
from dataclasses import replace

from rlmagent.harness.provider.base import ModelProvider
from rlmagent.model.settings import (
    AnthropicProfile,
    ConfigError,
    OpenAIProfile,
    load_anthropic_profile,
    load_openai_profile,
)

# ---------------------------------------------------------------------------
# Known provider identifiers
# ---------------------------------------------------------------------------

_KNOWN_PROVIDERS = ("anthropic", "openai", "openrouter", "ollama", "huggingface", "scripted", "cli", "vertex",
                    "google", "mistral", "azure", "codex", "copilot", "claude-subscription", "xai", "xai-subscription")

# OpenAI-compatible backends reuse the OpenAI adapter with a different base URL.
_OPENAI_COMPATIBLE = ("openai", "openrouter", "ollama", "huggingface")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect_provider_name() -> str | None:
    """Infer the provider name from environment variables, then saved config."""
    explicit = os.environ.get("RLM_AGENT_PROVIDER")
    if explicit:
        return explicit.strip().lower()

    from rlmagent.config.store import load_config

    saved = load_config().provider
    if saved:
        return saved.strip().lower()

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    return None


# ---------------------------------------------------------------------------
# Builder helpers (one per provider)
# ---------------------------------------------------------------------------


def _build_anthropic(profile: AnthropicProfile) -> ModelProvider:
    """Instantiate an Anthropic adapter from a loaded profile."""
    try:
        from rlmagent.model.claude import AnthropicProvider  # type: ignore[import-not-found]
    except ImportError:
        raise ConfigError(
            "The 'anthropic' provider requires the httpx extra.\n"
            "  Install it with: pip install rlmagent[providers]"
        ) from None
    return AnthropicProvider(profile)  # type: ignore[return-value]


def _build_openai(profile: OpenAIProfile) -> ModelProvider:
    """Instantiate an OpenAI-compatible adapter from a loaded profile."""
    try:
        from rlmagent.model.oai_compatible import OpenAIProvider  # type: ignore[import-not-found]
    except ImportError:
        raise ConfigError(
            "The 'openai' provider requires the httpx extra.\n"
            "  Install it with: pip install rlmagent[providers]"
        ) from None
    return OpenAIProvider(profile)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_provider(name: str | None = None) -> ModelProvider:
    """Return a ready-to-use ``ModelProvider`` for *name*, with any fallbacks."""
    from rlmagent.model.failover import with_fallbacks

    return with_fallbacks(_resolve_one(name), _resolve_one)


def _resolve_one(name: str | None = None) -> ModelProvider:
    """Build the provider for *name* alone."""
    resolved = name or _detect_provider_name()
    if resolved is None:
        raise ConfigError(
            "No provider configured.\n"
            "  Set RLM_AGENT_PROVIDER or export an API key env var.\n"
            "  Run 'rlm-agent' to pick a provider and model, or 'rlm-agent config'."
        )

    resolved = resolved.strip().lower()
    if resolved == "copilot":
        from rlmagent.model.copilot import CopilotProvider
        return CopilotProvider()
    if resolved == "xai-subscription":
        from rlmagent.model.registered_oauth import BASE, resolve
        from rlmagent.model.settings import Credential

        return _build_openai(OpenAIProfile(name=resolved, credential=Credential(None, BASE),
                                          resolver=resolve, endpoint="chat"))
    if resolved == "claude-subscription":
        from rlmagent.model.external import ExternalAgentProvider
        return ExternalAgentProvider()

    if resolved == "codex":
        from rlmagent.model.settings import Credential
        from rlmagent.model.subscription import RESOURCE, resolve

        transport = os.environ.get("RLM_AGENT_CODEX_TRANSPORT", "http")
        if transport not in ("http", "websocket"):
            raise ConfigError("RLM_AGENT_CODEX_TRANSPORT must be http or websocket")
        return _build_openai(OpenAIProfile(
            name="codex", credential=Credential(None, RESOURCE),
            resolver=resolve, endpoint="responses", transport=transport,
        ))

    if resolved == "anthropic":
        profile = load_anthropic_profile()
        return _build_anthropic(profile)

    if resolved == "google":
        from rlmagent.model.google import build_google

        return build_google()

    if resolved in ("mistral", "azure", "xai"):
        from rlmagent.config.loader import load_api_key
        from rlmagent.config.store import load_config, provider_info
        from rlmagent.model.settings import Credential

        info = provider_info(resolved)
        key = load_api_key(resolved)
        if not key:
            raise ConfigError(f"{info.key_env} is not set")
        base = os.environ.get("AZURE_OPENAI_BASE_URL" if resolved == "azure" else resolved.upper() + "_BASE_URL")
        base = base or load_config().base_urls.get(resolved) or info.base_url
        if not base:
            raise ConfigError("Set AZURE_OPENAI_BASE_URL to your deployed /openai/v1 endpoint")
        credential = (Credential(None, base, (("api-key", key),)) if resolved == "azure"
                      else Credential(key, base))
        return _build_openai(OpenAIProfile(name=resolved, credential=credential,
                                          endpoint="responses" if resolved == "azure" else "chat"))

    if resolved in _OPENAI_COMPATIBLE:
        # Named for the chosen provider, so replies, stats, and errors say
        # "openrouter" or "ollama" rather than the shared adapter's "openai".
        profile = replace(load_openai_profile(), name=resolved)
        return _build_openai(profile)

    if resolved == "vertex":
        from rlmagent.model.vertex import build_vertex
        return build_vertex()

    if resolved == "cli":
        from rlmagent.model.external import ExternalAgentProvider

        return ExternalAgentProvider()  # type: ignore[return-value]

    if resolved == "scripted":
        from rlmagent.model.scripted import ReplayProvider

        return ReplayProvider([])  # type: ignore[return-value]

    raise ConfigError(
        f"Unknown provider: {resolved!r}.\n"
        f"  Known providers: {', '.join(_KNOWN_PROVIDERS)}"
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def _probe_status(provider_name: str) -> str:
    """Return a short status string for a provider: 'ready', 'no key', etc."""
    if provider_name == "codex":
        return "subscription  (rlm-agent provider login codex)"
    if provider_name in ("google", "mistral", "azure"):
        from rlmagent.config.loader import load_api_key
        from rlmagent.config.store import provider_info

        return "key configured" if load_api_key(provider_name) else (
            f"no key  (set {provider_info(provider_name).key_env})")
    if provider_name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "no key  (set ANTHROPIC_API_KEY)"
        try:
            load_anthropic_profile()
            return "ready"
        except ConfigError as exc:
            return f"misconfigured  ({exc})"

    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return "no key  (set OPENAI_API_KEY)"
        try:
            load_openai_profile()
            return "ready"
        except ConfigError as exc:
            return f"misconfigured  ({exc})"

    if provider_name == "scripted":
        return "available  (test only)"

    return "unknown"


def list_providers() -> None:
    """Print available providers and their configuration status to stderr."""
    active = _detect_provider_name()
    sys.stderr.write("Providers:\n")
    for name in _KNOWN_PROVIDERS:
        marker = "*" if name == active else " "
        status = _probe_status(name)
        sys.stderr.write(f"  {marker} {name:<12}  {status}\n")
    sys.stderr.flush()


def setup_provider() -> None:
    """Print environment-variable guidance for configuring each provider."""
    sys.stderr.write(
        "Provider setup\n"
        "\nCodex subscription:\n"
        "  Install rlmagent[subscription], then run rlm-agent provider login codex.\n"
        "  Use --provider codex; tokens stay in the platform credential store.\n"
        "\nNative providers:\n"
        "  GOOGLE_API_KEY, MISTRAL_API_KEY, or AZURE_OPENAI_API_KEY\n"
        "  Azure also requires AZURE_OPENAI_BASE_URL ending in /openai/v1.\n"
        "\n"
        "Set the following environment variables for your chosen provider.\n"
        "\n"
        "Anthropic:\n"
        "  ANTHROPIC_API_KEY       (required)\n"
        "  ANTHROPIC_BASE_URL      (default: https://api.anthropic.com)\n"
        "  ANTHROPIC_TIMEOUT       (seconds, default: 120)\n"
        "  ANTHROPIC_MAX_TOKENS    (default: 16384)\n"
        "  ANTHROPIC_EXTRA_HEADERS (semicolon-separated Key: Value pairs)\n"
        "\n"
        "OpenAI-compatible:\n"
        "  OPENAI_API_KEY          (required)\n"
        "  OPENAI_BASE_URL         (default: https://api.openai.com/v1)\n"
        "  OPENAI_TIMEOUT          (seconds, default: 120)\n"
        "  OPENAI_ORG_ID           (optional)\n"
        "  OPENAI_EXTRA_HEADERS    (semicolon-separated Key: Value pairs)\n"
        "\n"
        "Google Vertex:\n"
        "  Install rlmagent[vertex]; configure Application Default Credentials.\n"
        "  GOOGLE_CLOUD_PROJECT  (required unless ADC supplies a project)\n"
        "  GOOGLE_CLOUD_LOCATION (default: global)\n"
        "\n"
        "Shared:\n"
        "  RLM_AGENT_PROVIDER          (anthropic | openai | scripted)\n"
        "  RLM_AGENT_MODEL             (model identifier override)\n"
        "  RLM_AGENT_MAX_RETRIES       (default: 2)\n"
        "  RLM_AGENT_MAX_RETRY_DELAY   (seconds, default: 30)\n"
        "  RLM_AGENT_THINKING_ENABLED  (true/false, default: false)\n"
        "  RLM_AGENT_THINKING_BUDGET   (token count, optional)\n"
    )
    sys.stderr.flush()


def select_provider(name: str) -> None:
    """Validate a provider name for selection."""
    normalized = name.strip().lower()
    if normalized not in _KNOWN_PROVIDERS:
        raise ConfigError(
            f"Unknown provider: {name!r}.\n"
            f"  Known providers: {', '.join(_KNOWN_PROVIDERS)}"
        )


__all__ = [
    "list_providers",
    "resolve_provider",
    "select_provider",
    "setup_provider",
]
