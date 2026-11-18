"""The kernel started at launch is the one the first cell runs in."""

import time

from rlmagent import prewarm


def test_started_kernel_is_adopted_and_runs_cells():
    prewarm.start()
    deadline = time.time() + 20
    while prewarm._ready is None and time.time() < deadline:
        time.sleep(0.05)
    started = prewarm._ready
    assert started is not None
    runtime = prewarm.runtime_class()(None, {}, 30)
    try:
        assert runtime.channel.process is started if hasattr(runtime, "channel") else True
        assert prewarm._ready is None
        out = runtime.execute("x = 6 * 7\nprint(x)")
        assert out.error is None and out.stdout.strip() == "42"
    finally:
        runtime.close()


def test_without_a_started_kernel_it_starts_its_own():
    assert prewarm.take() is None
    runtime = prewarm.runtime_class()(None, {}, 30)
    try:
        assert runtime.execute("print(1)").stdout.strip() == "1"
    finally:
        runtime.close()


def test_runtime_class_is_stable_for_reuse_checks():
    assert prewarm.runtime_class() is prewarm.runtime_class()


def test_wanted_only_for_local_one_shot_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    monkeypatch.delenv("RLM_AGENT_KERNEL", raising=False)
    monkeypatch.delenv("RLM_AGENT_CONFIG_DIR", raising=False)
    assert prewarm.wanted(["-p", "hi"])
    assert not prewarm.wanted(["hi"])
    assert not prewarm.wanted(["-p", "--sandbox", "hi"])
    assert not prewarm.wanted(["-p", "--kernel", "wasm", "hi"])
    (tmp_path / "config.toml").write_text('kernel = "wasm"\n')
    assert not prewarm.wanted(["-p", "hi"])
    assert prewarm.wanted(["-p", "--no-sandbox", "hi"])


def test_version_answers_without_loading_the_cli(capsys):
    import sys

    from rlmagent.cli.entry import main

    sys.modules.pop("rlmagent.cli.main", None)
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("rlm-agent ")
    assert "rlmagent.cli.main" not in sys.modules
