import json

import pytest

from rlmagent.history import agent_steps, conversation, make_history


def test_history_reads_the_whole_record(tmp_path):
    vault = tmp_path / "s.jsonl"
    rows = [
        {"type": "session_info"},
        {"type": "message", "timestamp": 1, "message": {"role": "user", "content": "fix it"}},
        {"type": "compaction", "timestamp": 2, "summary": "earlier work"},
        {"type": "message", "timestamp": 3, "message": {"role": "assistant", "content": [
            {"type": "text", "text": "done"}, {"type": "toolCall", "name": "python",
                                                "arguments": {"code": "x"}}]}},
    ]
    vault.write_text("\n".join(json.dumps(r) for r in rows) + "\nbroken\n")
    found = conversation(vault)
    assert [r["role"] for r in found] == ["user", "compaction", "assistant"]
    assert found[2]["text"].startswith("done\n[calls python")
    events = tmp_path / "e.jsonl"
    events.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "code_generated", "run_id": "c1", "step": 1, "code": "a = 1"},
        {"kind": "output_received", "run_id": "c1", "step": 1, "output": "ok"},
        {"kind": "code_generated", "run_id": "root", "step": 1, "code": "b = 2"},
    ]))
    assert agent_steps(events, "c1") == [{"step": 1, "code": "a = 1", "output": "ok"}]


async def test_history_function_pages_and_handles_no_session(tmp_path):
    vault = tmp_path / "s.jsonl"
    vault.write_text("\n".join(json.dumps({"type": "message", "message": {
        "role": "user", "content": str(n)}}) for n in range(5)))
    history = make_history(lambda: vault, lambda: None)
    page = await history(1, 3)
    assert page["total"] == 5 and [i["text"] for i in page["items"]] == ["1", "2"]
    assert (await history(agent="x"))["total"] == 0
    assert (await make_history(lambda: None, lambda: None)())["items"] == []


def test_a_dead_kernel_is_replaced_and_the_run_goes_on():
    from rlmness.runtime import CellOutcome, RuntimeGone

    from rlmagent.backend import RESTARTED, shape_cells

    class Kernel:
        built = 0

        def __init__(self):
            Kernel.built += 1
            self.ran = []
            self.alive = True

        def execute(self, code):
            if not self.alive:
                raise RuntimeGone("runtime exited")
            self.ran.append(code)
            return CellOutcome(stdout=f"ran {code}")

    kernel = Kernel()
    shape_cells(kernel, None, rebuild=Kernel)
    kernel.execute("helpers")
    kernel.execute("context")
    kernel.execute("x = 1")
    kernel.alive = False
    outcome = kernel.execute("print(x)")
    assert outcome.error == RESTARTED and Kernel.built == 2
    assert kernel.ran == ["helpers", "context"]  # the new kernel, set up again
    assert kernel.execute("y = 2").stdout == "ran y = 2"
    for _ in range(2):  # three restarts in all are allowed
        kernel.alive = False
        kernel.execute("again")
    kernel.alive = False
    with pytest.raises(RuntimeGone):
        kernel.execute("too many")
