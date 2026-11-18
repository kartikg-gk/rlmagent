"""rlm-agent cli: the command-line entry point."""

__all__ = ["get_version", "main"]


def __getattr__(name: str):
    # Importing the package stays cheap; the full CLI loads when it is used.
    if name in __all__:
        from rlmagent.cli import main as module

        return getattr(module, name)
    raise AttributeError(name)
