"""Run the eval cases headless and report PASS/FAIL.

Each case is evals/cases/<name>/: a fixture/ folder copied into a fresh temp
workspace, task.txt given to `rlm-agent -p`, and verify.py run afterwards in that
workspace. verify.py never enters the workspace, so the agent cannot read it.

    python scripts/run_evals.py -m cli:claude-code?model=haiku --provider cli
    python scripts/run_evals.py off-by-one -- --max-cost 0.2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CASES = Path(__file__).resolve().parent.parent / "evals" / "cases"


def _agent() -> str:
    """rlm-agent from PATH, else the one next to this Python."""
    beside = Path(sys.executable).with_name("rlm-agent.exe" if sys.platform == "win32" else "rlm-agent")
    return shutil.which("rlm-agent") or str(beside)


def run_case(case: Path, agent_args: list[str], timeout: float) -> dict:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(case / "fixture", work)
        task = (case / "task.txt").read_text(encoding="utf-8")
        started = time.monotonic()
        try:
            agent = subprocess.run(
                [_agent(), "-p", "--no-auto-improve", *agent_args, task], cwd=work,
                capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
            )
            agent_error = "" if agent.returncode == 0 else agent.stderr[-400:]
        except subprocess.TimeoutExpired:
            agent_error = f"timed out after {timeout:.0f}s"
        seconds = time.monotonic() - started
        check = subprocess.run([sys.executable, str(case / "verify.py")], cwd=work,
                               capture_output=True, text=True)
    passed = check.returncode == 0
    detail = "" if passed else (check.stderr.strip().splitlines() or ["failed"])[-1]
    return {"case": case.name, "passed": passed, "seconds": round(seconds, 1),
            "detail": detail or agent_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    parser.add_argument("-m", "--model")
    parser.add_argument("--provider")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json", dest="json_out", help="write results here")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args, extra = parser.parse_args(argv[:split]), argv[split + 1:]
    agent_args = ([f"--provider={args.provider}"] if args.provider else []) + \
                 ([f"--model={args.model}"] if args.model else []) + extra
    names = args.cases or sorted(p.name for p in CASES.iterdir() if p.is_dir())
    results = []
    for name in names:
        result = run_case(CASES / name, agent_args, args.timeout)
        results.append(result)
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"{mark}  {name:<20} {result['seconds']:>6}s  {result['detail']}", flush=True)
    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} passed")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
