"""Real tasks for a real model, checked by what the agent leaves in its workspace.

These cost money and need a model, so they run only when asked:

    RLM_AGENT_E2E=1 pytest tests/e2e -m e2e

RLM_AGENT_E2E_PROVIDER and RLM_AGENT_E2E_MODEL choose the model (default: the Claude Code
CLI with haiku, which needs that CLI installed and signed in). Any provider rlm-agent
supports works; its key comes from the usual environment variable.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("RLM_AGENT_E2E") != "1",
                       reason="end-to-end tasks run only with RLM_AGENT_E2E=1"),
]

SRC = str(Path(__file__).resolve().parents[2] / "src")
PROVIDER = os.environ.get("RLM_AGENT_E2E_PROVIDER", "cli")
MODEL = os.environ.get("RLM_AGENT_E2E_MODEL", "cli:claude-code?model=haiku")
SECONDS = 900

BUGGY = "def add(a, b):\n    return a - b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _env(home: Path, **extra) -> dict:
    return {**os.environ, "RLM_AGENT_HOME": str(home), "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", ""), **extra}


def _agent(cwd: Path, home: Path, prompt: str, *flags: str, env=None) -> subprocess.CompletedProcess:
    command = [sys.executable, "-c", "from rlmagent.cli.main import main; raise SystemExit(main())",
               "--provider", PROVIDER, "--model", MODEL, "--auto-approve", *flags, "-p", prompt]
    return subprocess.run(command, cwd=cwd, env=env or _env(home), capture_output=True,
                          text=True, timeout=SECONDS, encoding="utf-8", errors="replace")


def _project(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "calc.py").write_text(BUGGY)
    (root / "test_calc.py").write_text(TEST)
    return root


def _tests_pass(root: Path) -> bool:
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=root, capture_output=True, text=True, timeout=300)
    return done.returncode == 0


FIX = ("The test in test_calc.py fails. Find the bug, fix it with edit_file, run the "
       "tests with run('python -m pytest -q') and FINAL a one-line summary.")


def test_fix_a_bug_in_the_local_workspace(tmp_path):
    work = _project(tmp_path / "w")
    done = _agent(work, tmp_path / "home", FIX)
    assert done.returncode == 0, done.stderr[-2000:]
    assert _tests_pass(work), (work / "calc.py").read_text()


def test_build_a_file_from_a_description(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    done = _agent(work, tmp_path / "home",
                  "Create slug.py with a function slug(text) that lowercases text and joins "
                  "words with single hyphens, dropping anything that is not a letter, digit "
                  "or space. Check it with a quick run() and FINAL done.")
    assert done.returncode == 0, done.stderr[-2000:]
    namespace: dict = {}
    exec((work / "slug.py").read_text(), namespace)
    assert namespace["slug"]("Hello,  World 2!") == "hello-world-2"


def test_fix_a_bug_through_a_tool_host_over_stdio(tmp_path):
    work = _project(tmp_path / "remote")
    here = tmp_path / "here"
    here.mkdir()
    host = f'"{sys.executable}" -m rlmagent.toolhost --root "{work}"'
    done = _agent(here, tmp_path / "home", FIX, "--tools-at", host)
    assert done.returncode == 0, done.stderr[-2000:]
    # Only the harness's own run logs belong here; the work happened in the host's root.
    assert _tests_pass(work) and {p.name for p in here.iterdir()} <= {".logs"}


def test_fix_a_bug_through_a_tool_host_over_tcp(tmp_path):
    work = _project(tmp_path / "remote")
    here = tmp_path / "here"
    here.mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = _env(tmp_path / "home", RLM_AGENT_TOOLS_TOKEN="e2e-token")
    host = subprocess.Popen([sys.executable, "-m", "rlmagent.toolhost", "--root", str(work),
                             "--tcp", f"127.0.0.1:{port}"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        done = _agent(here, tmp_path / "home", FIX, "--tools-at", f"tcp://127.0.0.1:{port}",
                      env=env)
        assert done.returncode == 0, done.stderr[-2000:]
        assert _tests_pass(work)
    finally:
        host.kill()
        host.wait()


def test_a_background_job_and_the_git_guard(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    done = _agent(work, tmp_path / "home",
                  "Start `python -c \"import time; time.sleep(2); print('job-finished')\"` with "
                  "job_start, wait for it with job_wait, read its output with job_read, and "
                  "write that output to out.txt. Then try run('git push') and write the error "
                  "you get to guard.txt. FINAL done.", "--deny-git")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "job-finished" in (work / "out.txt").read_text()
    assert "beyond the current branch" in (work / "guard.txt").read_text()
