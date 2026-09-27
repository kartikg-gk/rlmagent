"""Smoke test: the three packages and the core contracts import."""

import rlmagent_app
import rlmagent_harness
import rlmagent_model
from rlmagent_harness import AgentEvent, CallBlock, HumanEntry, ToolSpec


def test_packages_import() -> None:
    assert rlmagent_harness.__doc__
    assert rlmagent_model.__doc__
    assert rlmagent_app.__doc__


def test_contracts_importable() -> None:
    assert AgentEvent is not None
    assert ToolSpec is not None
    assert CallBlock is not None
    assert HumanEntry is not None
