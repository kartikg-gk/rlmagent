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
    assert "write mathutil.py" in err and "--read-only" in err


@pytest.mark.asyncio
async def test_auto_approve_reports_nothing(capsys):
    approver = Approver("auto")
    await approver.approve(Request(kind="write", target="mathutil.py"))
    assert _report_unasked(SimpleNamespace(agent_tree=SimpleNamespace(approver=approver))) is False
    assert capsys.readouterr().err == ""


def test_print_runs_act_unless_read_only_and_shell_follows_sandbox():
    from rlmagent.runtime import approval_policy, shell_access_for

    ns = SimpleNamespace(print_mode=True, prompt="x", auto_approve=False, read_only=False)
    assert approval_policy(ns) == "auto"
    ns.read_only = True
    assert approval_policy(ns) == "deny"
    assert approval_policy(SimpleNamespace(print_mode=False, prompt=None)) == "ask"
    assert shell_access_for(SimpleNamespace(kernel="local", shell_access=None)) == "host"
    assert shell_access_for(SimpleNamespace(kernel="wasm", shell_access=None)) == "workspace"
    assert shell_access_for(SimpleNamespace(kernel="local", shell_access="workspace")) == "workspace"


def test_claude_md_is_read_as_project_instructions(tmp_path, monkeypatch):
    from rlmagent import instructions

    (tmp_path / "CLAUDE.md").write_text("Use tabs.")
    paths = [p.name for p in instructions._agents_md_search_paths(str(tmp_path))]
    assert "CLAUDE.md" in paths
