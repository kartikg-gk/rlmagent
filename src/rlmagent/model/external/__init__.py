from rlmagent.model.external.cli import ExternalAgentProvider, is_external, parse_model


def with_external(provider):
    """`provider`, plus `cli:` and `acp:` models for any agent that asks for one."""
    if isinstance(provider, ExternalAgentProvider):
        return provider
    return ExternalAgentProvider(fallback=provider)


__all__ = ["ExternalAgentProvider", "is_external", "parse_model", "with_external"]
