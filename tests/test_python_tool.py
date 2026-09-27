import pytest

from rlmagent_app.kernel import CellResult, KernelSession
from rlmagent_app.tools import build_tool_registry
from rlmagent_app.tools.kernel import format_cell, make_python_tool



def test_registry_has_python_and_file_tools_but_no_bash(tmp_path):
    names = [t.name for t in build_tool_registry(KernelSession(cwd=str(tmp_path)))]
    assert "python" in names
    assert {"Read", "Write", "Edit"} <= set(names)
    assert "Bash" not in names


def test_registry_without_kernel_has_no_python():
    assert "python" not in [t.name for t in build_tool_registry()]


def test_format_cell_variants():
    assert format_cell(CellResult(output="")) == "(no output)"
    assert "Traceback" in format_cell(CellResult(output="", error="Traceback: boom"))
    assert "timed out" in format_cell(CellResult(output="x", timed_out=True))
    assert "variables from before are gone" in format_cell(CellResult(output="x", restarted=True))


async def test_tool_runs_code_in_the_kernel(tmp_path):
    kernel = KernelSession(cwd=str(tmp_path), timeout=20)
    tool = make_python_tool(kernel)
    try:
        await tool.execute("c1", {"code": "z = 5"})
        outcome = await tool.execute("c2", {"code": "print(z + 1)"})
        assert outcome.text.strip() == "6"
    finally:
        await kernel.shutdown()


def test_plan_mode_removes_the_python_tool(tmp_path):
    from rlmagent_app.planning import restrict_tools

    tools = build_tool_registry(KernelSession(cwd=str(tmp_path)))
    assert "python" not in [t.name for t in restrict_tools(tools)]


def test_python_calls_count_as_mutating_for_approval():
    from rlmagent_app.cli.main import _approval_context
    from rlmagent_harness.contracts.transcript import CallBlock

    context = _approval_context(CallBlock(name="python", id="c1", arguments={"code": "open('x','w')"}))
    assert context.is_mutating
    assert context.command == "open('x','w')"
