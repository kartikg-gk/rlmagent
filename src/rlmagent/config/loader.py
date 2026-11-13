"""Resolve provider, model, and credentials from environment then saved config."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from rlmagent.config.store import RlmAgentConfig, load_config, provider_info

_DEFAULT_PROVIDER = "anthropic"

#: Per-project settings, inside the project. Only these keys are read: a
#: repository may pick its provider, model and fallbacks, never credentials,
#: endpoints or the trust policy.
PROJECT_SETTINGS = Path(".rlm-agent") / "settings.toml"
PROJECT_KEYS = ("provider", "model", "fallback")


def project_settings(cwd: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """The trusted project's own settings, or ``{}``."""
    folder = Path(cwd or os.getcwd())
    path = folder / PROJECT_SETTINGS
    if not path.is_file():
        return {}
    from rlmagent.trust import project_settings_allowed

    if not project_settings_allowed(folder):
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    return {key: value.strip() for key in PROJECT_KEYS
            if isinstance(value := data.get(key), str) and value.strip()}


def load_provider(config: RlmAgentConfig | None = None) -> str:
    """Active provider: ``RLM_AGENT_PROVIDER``, then config, then detection."""
    env = os.environ.get("RLM_AGENT_PROVIDER")
    if env:
        return env.strip().lower()

    if config is None and (project := project_settings().get("provider")):
        return project.lower()
    cfg = load_config() if config is None else config
    if cfg.provider:
        return cfg.provider.strip().lower()

    # No explicit choice — infer from whichever key is present.
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return _DEFAULT_PROVIDER


def load_model(provider: str, config: RlmAgentConfig | None = None) -> str | None:
    """Model for *provider*: ``RLM_AGENT_MODEL``, then config, then ``None``."""
    env = os.environ.get("RLM_AGENT_MODEL")
    if env:
        return env
    if config is None:
        project = project_settings()
        if project.get("model") and project.get("provider", provider).lower() == provider:
            return project["model"]
    cfg = load_config() if config is None else config
    if cfg.model and cfg.provider == provider:
        return cfg.model
    return None


def load_api_key(provider: str, config: RlmAgentConfig | None = None) -> str | None:
    """Key for *provider*: its env var, then config, then ``None``."""
    try:
        info = provider_info(provider)
    except KeyError:
        return None
    env = os.environ.get(info.key_env)
    if env:
        return env
    cfg = load_config() if config is None else config
    return cfg.api_keys.get(provider)


def apply_config(provider: str, config: RlmAgentConfig | None = None) -> None:
    """Export saved credentials for *provider* into this process's environment."""
    try:
        info = provider_info(provider)
    except KeyError:
        return

    if provider in ("vertex", "google", "mistral", "azure", "codex", "copilot", "claude-subscription", "xai", "xai-subscription"):
        return
    cfg = load_config() if config is None else config
    key = cfg.api_keys.get(provider)
    base_url = cfg.base_urls.get(provider) or info.base_url

    if provider == "anthropic":
        key_var, url_var = "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"
    else:
        key_var, url_var = "OPENAI_API_KEY", "OPENAI_BASE_URL"

    # A real env var for the provider's own key name wins over the config file.
    native = os.environ.get(info.key_env)
    if native:
        key = native

    if key and not os.environ.get(key_var):
        os.environ[key_var] = key
    if not os.environ.get(url_var):
        os.environ[url_var] = base_url

    # Ollama needs no credential but the OpenAI loader requires a non-empty key.
    if not info.needs_key and not os.environ.get(key_var):
        os.environ[key_var] = "ollama"


__all__ = ["apply_config", "load_api_key", "load_model", "load_provider"]
