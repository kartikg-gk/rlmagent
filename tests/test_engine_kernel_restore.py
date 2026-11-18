"""With restore on, the default engine's kernel comes back from its last checkpoint."""

from rlmness.runtime import SubprocessRuntime

from rlmagent.backend import RESTARTED, shape_cells


def _start(checkpoint=None, restore=True):
    def build(cp=None):
        return SubprocessRuntime(None, {}, 30, session=(dict(cp or {}) if restore else None))
    runtime = build(checkpoint)
    shape_cells(runtime, None, rebuild=build if restore else (lambda: build()), restore=restore)
    return runtime


def test_crash_restores_variables_functions_and_imports():
    rt = _start()
    try:
        rt.execute("helper_from_setup = 1")
        rt.execute("pass")
        assert rt.execute("import json as j\nrows = [1, 2, 3]\ndef twice(x):\n    return 2 * x").error is None
        crashed = rt.execute("import os; os._exit(1)")
        assert "last checkpoint" in crashed.error and "rows" in crashed.error
        out = rt.execute("print(rows, twice(4), j.dumps(1))")
        assert out.error is None and "[1, 2, 3] 8 1" in out.stdout
    finally:
        rt.close()


def test_without_restore_a_crash_starts_empty():
    rt = _start(restore=False)
    try:
        rt.execute("pass")
        rt.execute("pass")
        rt.execute("rows = [1]")
        assert rt.execute("import os; os._exit(1)").error == RESTARTED
        assert "NameError" in (rt.execute("rows").error or "")
    finally:
        rt.close()
