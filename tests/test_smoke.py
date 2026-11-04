"""Smoke test: the three packages and the core contracts import."""

import rlmagent
import rlmagent.harness
import rlmagent.model
from rlmagent.harness import AgentEvent, CallBlock, HumanEntry, ToolSpec


def test_packages_import() -> None:
    assert rlmagent.harness.__doc__
    assert rlmagent.model.__doc__
    assert rlmagent.__doc__


def test_contracts_importable() -> None:
    assert AgentEvent is not None
    assert ToolSpec is not None
    assert CallBlock is not None
    assert HumanEntry is not None
