from rlmagent.model.external.cli import PREFIX, ExternalAgentProvider, parse_model


def with_external(provider):
    """`provider`, plus `cli:` models for any agent that asks for one."""
    if isinstance(provider, ExternalAgentProvider):
        return provider
    return ExternalAgentProvider(fallback=provider)


__all__ = ["PREFIX", "ExternalAgentProvider", "parse_model", "with_external"]
