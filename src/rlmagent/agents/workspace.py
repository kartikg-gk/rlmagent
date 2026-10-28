"""Host functions the sandboxed kernel uses to reach the project."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rlmagent.agents.approval import Approver, Request, check

MAX_READ = 2_000_000
MAX_OUTPUT = 50_000
MAX_LIST = 2_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".rlm-agent"}


class Workspace:
    def __init__(self, root: str | Path, approver: Approver) -> None:
        self.root = Path(root).resolve()
        self.approver = approver

    def _path(self, raw: str) -> Path:
        path = (self.root / raw).resolve()  # symlinks followed, so a link out is caught
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"{raw} is outside the workspace")
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    async def read_file(self, path: str) -> str:
        target = self._path(path)
        if target.stat().st_size > MAX_READ:
            raise ValueError(f"{path} is larger than {MAX_READ} bytes")
        return target.read_text(encoding="utf-8")

    async def list_files(self, pattern: str = "**/*") -> list[str]:
        found = []
        for path in sorted(self.root.glob(pattern)):
            rel = path.relative_to(self.root)
            if path.is_file() and not SKIP_DIRS.intersection(rel.parts):
                found.append(rel.as_posix())
                if len(found) >= MAX_LIST:
                    break
        return found

    async def write_file(self, path: str, content: str) -> str:
        target = self._path(path)
        request = Request("write", self._rel(target), (content,))
        approved = await self.approver.approve(request)
        check(approved, Request("write", self._rel(target), (content,)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {self._rel(target)} ({len(content)} characters)"

    async def edit_file(self, path: str, old: str, new: str) -> str:
        target = self._path(path)
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"old text found {count} times in {path}; it must be exactly once")
        request = Request("edit", self._rel(target), (old, new))
        approved = await self.approver.approve(request)
        check(approved, Request("edit", self._rel(target), (old, new)))
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {self._rel(target)}"

    async def run(self, cmd: str, timeout: float = 120.0) -> dict:
        approved = await self.approver.approve(Request("run", cmd))
        check(approved, Request("run", cmd))
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=str(self.root), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            from rlmagent.model.external.cli import kill_tree

            kill_tree(proc)
            await proc.wait()
            return {"exit": None, "output": f"timed out after {timeout:.0f}s"}
        text = out.decode("utf-8", errors="replace")
        if len(text) > MAX_OUTPUT:
            text = f"... ({len(text) - MAX_OUTPUT} characters cut)\n" + text[-MAX_OUTPUT:]
        return {"exit": proc.returncode, "output": text}

    def functions(self) -> dict:
        return {"read_file": self.read_file, "list_files": self.list_files,
                "write_file": self.write_file, "edit_file": self.edit_file, "run": self.run}

