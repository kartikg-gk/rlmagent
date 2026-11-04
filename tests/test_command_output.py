import shlex
import sys

from rlmagent.agents.approval import Approver
from rlmagent.agents.output import BYTE_LIMIT, LINE_LIMIT, OutputTail
from rlmagent.agents.workspace import Workspace


def test_collector_retained_memory_and_report_are_bounded():
    tail = OutputTail()
    for _ in range(500):
        tail.feed(b"a" * 8192)
        assert len(tail.data) < BYTE_LIMIT
    assert len(tail.text().encode()) <= BYTE_LIMIT
    assert tail.total == 500 * 8192
    assert tail.truncated


def test_line_limit_and_unicode_boundary():
    tail = OutputTail()
    tail.feed(("unicode \u03bb\n" * 10000).encode())
    assert len(tail.text().splitlines()) <= LINE_LIMIT
    assert len(tail.text().encode()) <= BYTE_LIMIT
    assert "\ufffd" not in tail.text()


async def test_large_command_drains_to_completion_and_keeps_tail(tmp_path):
    script = tmp_path / "produce.py"
    script.write_text("print('x' * 2000000)\nprint('last-line')\n")
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    result = await Workspace(tmp_path, Approver("auto")).run(command, timeout=15)
    assert result["exit"] == 0
    assert result["output"].splitlines()[-1] == "last-line"
    assert result["truncated"] and result["output_bytes"] > 2000000
    assert len(result["output"].encode()) <= BYTE_LIMIT
