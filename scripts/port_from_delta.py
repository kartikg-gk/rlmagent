"""Copy delta's source into this repo under the rlm-agent names. Run once."""

import re
import shutil
import sys
from pathlib import Path

SOURCE = Path(r"E:\AI projects\delta")
TARGET = Path(__file__).resolve().parents[1]

PACKAGES = {
    "delta_harness": "rlmagent_harness",
    "delta_model": "rlmagent_model",
    "delta_app": "rlmagent_app",
}
# Order matters: longest, most specific first.
TEXT = [
    (r"\bdelta_harness\b", "rlmagent_harness"),
    (r"\bdelta_model\b", "rlmagent_model"),
    (r"\bdelta_app\b", "rlmagent_app"),
    (r"DELTA_HOME", "RLM_AGENT_HOME"),
    (r"DELTA_EXTENSIONS_DIR", "RLM_AGENT_EXTENSIONS_DIR"),
    (r"DELTA_PROVIDER", "RLM_AGENT_PROVIDER"),
    (r"DELTA_MODEL", "RLM_AGENT_MODEL"),
    (r"DELTA_API_KEY", "RLM_AGENT_API_KEY"),
    (r"~/\.delta\b", "~/.rlm-agent"),
    (r"\"\.delta\"", "\".rlm-agent\""),
    (r"'\.delta'", "'.rlm-agent'"),
    (r"\bdeltaa\b", "rlmagent"),
]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", "dist", "benchmark", "delta_bench", "tests_local"}


def rewrite(text: str) -> str:
    for pattern, repl in TEXT:
        text = re.sub(pattern, repl, text)
    return text


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(src)
        parts = [PACKAGES.get(p, p) for p in rel.parts]
        out = dst.joinpath(*parts)
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".py", ".toml", ".md", ".txt", ".tcss", ".cfg"}:
            out.write_text(rewrite(path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(path, out)


def main() -> int:
    for name in ("src", "tests"):
        copy_tree(SOURCE / name, TARGET / name)
    for name in ("pyproject.toml", "LICENSE", ".gitignore"):
        src = SOURCE / name
        if src.exists():
            (TARGET / name).write_text(rewrite(src.read_text(encoding="utf-8")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
