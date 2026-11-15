"""A one-shot run says on stderr what it could not do for lack of approval."""

from types import SimpleNamespace

import pytest

from rlmagent.agents.approval import Approver, Refused, Request
from rlmagent.cli.main import _report_unasked


@pytest.mark.asyncio
async def test_unasked_refusals_are_listed_and_reported(capsys):
    approver = Approver("deny")
    with pytest.raises(Refused):
        await approver.approve(Request(kind="write", target="mathutil.py"))
    session = SimpleNamespace(agent_tree=SimpleNamespace(approver=approver))
    assert _report_unasked(session) is True
    err = capsys.readouterr().err
    assert "write mathutil.py" in err and "--auto-approve" in err


@pytest.mark.asyncio
async def test_auto_approve_reports_nothing(capsys):
    approver = Approver("auto")
    await approver.approve(Request(kind="write", target="mathutil.py"))
    assert _report_unasked(SimpleNamespace(agent_tree=SimpleNamespace(approver=approver))) is False
    assert capsys.readouterr().err == ""
