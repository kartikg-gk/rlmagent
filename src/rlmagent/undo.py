"""/undo: a hidden git repository that snapshots the workspace before every turn.

The repository lives under ~/.rlm-agent/shadow/, never inside the project, so the
project's own git history is untouched. It honours the project's .gitignore and
also skips dependency folders and files that usually hold secrets. Kernel state
is not restored; only files and the conversation are.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

EXCLUDES = """.git/
node_modules/
.venv/
venv/
__pycache__/
.rlm-agent/
.env
.env.*
*.pem
*.key
id_rsa*
*.p12
"""
MAX_FILE_BYTES = 5_000_000


class Shadow:
    def __init__(self, workspace: str | Path, home: Path) -> None:
        self.workspace = Path(workspace).resolve()
        key = hashlib.sha1(str(self.workspace).encode()).hexdigest()[:16]
        self.git_dir = home / "shadow" / f"{key}.git"
        self.available = shutil.which("git") is not None

    def _git(self, *args: str) -> str:
        out = subprocess.run(
            ["git", f"--git-dir={self.git_dir}", f"--work-tree={self.workspace}", *args],
            capture_output=True, text=True, cwd=self.workspace, timeout=120,
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or f"git {args[0]} failed")
        return out.stdout.strip()

    def _init(self) -> None:
        if (self.git_dir / "HEAD").exists():
            return
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q")
        self._git("config", "user.name", "rlm-agent")
        self._git("config", "user.email", "rlm-agent@localhost")
        self._git("config", "core.autocrlf", "false")
        (self.git_dir / "info").mkdir(exist_ok=True)
        (self.git_dir / "info" / "exclude").write_text(EXCLUDES, encoding="utf-8")

    def _skip_large(self) -> None:
        listed = self._git("ls-files", "--others", "--modified", "--exclude-standard", "-z")
        for rel in filter(None, listed.split("\0")):
            path = self.workspace / rel
            if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
                with open(self.git_dir / "info" / "exclude", "a", encoding="utf-8") as f:
                    f.write(f"/{rel}\n")

    def snapshot(self, label: str) -> str:
        """Commit the workspace as it is now; returns the commit id."""
        self._init()
        self._skip_large()
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "--no-verify", "-m", label[:200])
        return self._git("rev-parse", "HEAD")

    def restore(self, commit: str) -> list[str]:
        """Put the workspace back to `commit`; returns the paths that changed."""
        self.snapshot("before undo")
        changed = self._git("diff", "--name-only", commit, "HEAD").splitlines()
        self._git("read-tree", "-u", "--reset", commit)
        self._git("commit", "-q", "--allow-empty", "--no-verify", "-m", f"undo to {commit[:12]}")
        return changed
