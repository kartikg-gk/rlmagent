"""Host functions the sandboxed kernel uses to reach the project."""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import re
import shutil
import time
from pathlib import Path

from rlmagent.agents.approval import Approver, Request, Refused, check
from rlmagent.agents.patch import PatchError, parse, patched
from rlmagent.model.external.cli import own_group

MAX_READ = 2_000_000
MAX_OUTPUT = 50 * 1024
MAX_LIST = 2_000
MAX_MATCHES = 500
MAX_MATCH_TEXT = 300
MAX_INSTRUCTION = 64 * 1024
MAX_INSTRUCTIONS = 200
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".rlm-agent"}


def _load(path: Path) -> str:
    """A file's exact text: line endings are not translated."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _save(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


class Workspace:
    def __init__(self, root: str | Path, approver: Approver) -> None:
        self.root = Path(root).resolve()
        self.approver = approver
        self.on_output = None  # called with command output as it arrives
        self.policy = None  # rlmagent.policy.Policy, when the run sets one

    def _path(self, raw: str) -> Path:
        path = (self.root / raw).resolve()  # symlinks followed, so a link out is caught
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"{raw} is outside the workspace")
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    async def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """The file's text, or only lines start..end (1-based, both included)."""
        target = self._path(path)
        ranged = start is not None or end is not None
        if not ranged and target.stat().st_size > MAX_READ:
            raise ValueError(
                f"{path} is larger than {MAX_READ} bytes; read it in parts with start and end"
            )
        try:
            if not ranged:
                return target.read_text(encoding="utf-8")
            first = max(1, int(start or 1))
            last = None if end is None else int(end)
            picked = []
            with open(target, encoding="utf-8") as f:
                for number, line in enumerate(f, 1):
                    if last is not None and number > last:
                        break
                    if number >= first:
                        picked.append(line)
            return "".join(picked)
        except UnicodeDecodeError:
            raise ValueError(f"{path} is a binary file, not UTF-8 text") from None

    async def search_files(self, pattern: str, path: str = ".", glob: str | None = None,
                           ignore_case: bool = False, limit: int = MAX_MATCHES) -> list[dict]:
        """Lines matching a regular expression: [{"path", "line", "text"}], capped at limit."""
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        base = self._path(path)
        limit = max(1, min(int(limit), MAX_MATCHES))
        files = [base] if base.is_file() else self._walk(base, glob)
        found: list[dict] = []
        for file in files:
            try:
                if file.stat().st_size > MAX_READ:
                    continue
                with open(file, encoding="utf-8", newline="") as f:
                    for number, line in enumerate(f, 1):
                        if regex.search(line):
                            text = line.rstrip("\r\n")
                            found.append({
                                "path": self._rel(file),
                                "line": number,
                                "text": text[:MAX_MATCH_TEXT],
                            })
                            if len(found) >= limit:
                                found.append({"note": f"stopped at {limit} matches; narrow the search"})
                                return found
            except (UnicodeDecodeError, OSError):
                continue
        return found

    def _walk(self, base: Path, glob: str | None):
        for folder, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(names):
                file = Path(folder) / name
                if glob and not (fnmatch.fnmatch(name, glob)
                                 or fnmatch.fnmatch(self._rel(file), glob)):
                    continue
                yield file

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
        before = _load(target) if target.exists() else None
        preview = self._diff(path, before or "", content)
        request = Request("write", self._rel(target), (content,), preview=preview)
        approved = await self.approver.approve(request)
        check(approved, Request("write", self._rel(target), (content,)))
        current = _load(target) if target.exists() else None
        if current != before:
            raise Refused(f"{path} changed while approval was pending; read it again")
        target.parent.mkdir(parents=True, exist_ok=True)
        _save(target, content)
        return f"wrote {self._rel(target)} ({len(content)} characters)"

    async def edit_file(self, path: str, old: str, new: str) -> str:
        target = self._path(path)
        raw = _load(target)
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n") if crlf else raw
        if crlf:
            old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"old text found {count} times in {path}; it must be exactly once")
        request = Request(
            "edit",
            self._rel(target),
            (old, new),
            preview=self._diff(path, text, text.replace(old, new, 1)),
        )
        approved = await self.approver.approve(request)
        check(approved, Request("edit", self._rel(target), (old, new)))
        if _load(target) != raw:
            raise Refused(f"{path} changed while approval was pending; read it again")
        changed = text.replace(old, new, 1)
        _save(target, changed.replace("\n", "\r\n") if crlf else changed)
        return f"edited {self._rel(target)}"

    async def apply_patch(self, diff: str) -> str:
        """Apply a unified diff to any number of files, all of them or none."""
        before: dict[Path, str | None] = {}
        after: dict[Path, str | None] = {}
        for change in parse(diff):
            target = self._path(change.path)
            exists = target.exists()
            if change.old_path is None and exists:
                raise PatchError(f"{change.path} already exists")
            if change.old_path is not None and not exists:
                raise PatchError(f"{change.path} does not exist")
            text = _load(target) if exists else ""
            before[target] = text if exists else None
            after[target] = None if change.new_path is None else patched(text, change)
        names = ", ".join(self._rel(t) for t in after)
        approved = await self.approver.approve(Request("patch", names, (diff,), preview=diff))
        check(approved, Request("patch", names, (diff,)))
        for target, text in before.items():
            current = _load(target) if target.exists() else None
            if current != text:
                raise Refused(f"{self._rel(target)} changed while approval was pending; read it again")
        done: list[Path] = []
        try:
            for target, text in after.items():
                done.append(target)
                if text is None:
                    target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _save(target, text)
        except OSError:
            for target in done:
                if before[target] is None:
                    target.unlink(missing_ok=True)
                else:
                    _save(target, before[target])
            raise
        return f"patched {names}"

    async def ask_user(self, question: str, choices: list[str] | None = None,
                       multiple: bool = False) -> str | list[str]:
        """Put a question to the user and wait; text starting with Error: if nobody answers."""
        if self.approver.question is None:
            return "Error: nobody can answer here. Decide yourself and say what you assumed."
        async with self.approver.one_question:
            options = list(choices) if choices else None
            answer = await (self.approver.question(str(question), options, multiple=True)
                            if multiple and options else
                            self.approver.question(str(question), options))
        return answer or "Error: the user did not answer. Decide yourself and say what you assumed."

    async def run(self, cmd: str, timeout: float = 120.0) -> dict:
        if self.policy is not None:
            from rlmagent.policy import git_refusal

            if reason := git_refusal(cmd, self.policy):
                raise ValueError(reason)
        approved = await self.approver.approve(Request("run", cmd))
        check(approved, Request("run", cmd))
        bash = shutil.which("bash")
        if bash is None and os.name == "nt":
            candidate = (
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            )
            if candidate.is_file():
                bash = str(candidate)
        if bash is None:
            raise OSError("Bash is required for workspace commands")
        timeout = max(0.1, min(float(timeout), 1800.0))
        proc = await asyncio.create_subprocess_exec(
            bash,
            "--noprofile",
            "--norc",
            "-c",
            cmd,
            cwd=str(self.root),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **own_group(),
        )
        from rlmagent.agents.output import OutputTail

        tail = OutputTail()

        async def drain():
            shown, unsent = 0.0, bytearray()
            while chunk := await proc.stdout.read(8192):
                tail.feed(chunk)
                if self.on_output is not None:
                    unsent.extend(chunk)
                    if time.monotonic() - shown >= 0.25:
                        self.on_output(bytes(unsent[-4096:]).decode("utf-8", errors="replace"))
                        shown = time.monotonic()
                        unsent.clear()
            if self.on_output is not None and unsent:
                self.on_output(bytes(unsent[-4096:]).decode("utf-8", errors="replace"))
            await proc.wait()

        reader = asyncio.create_task(drain())
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout)
        except asyncio.CancelledError:
            from rlmagent.model.external.cli import kill_tree

            kill_tree(proc)
            await proc.wait()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            raise
        except asyncio.TimeoutError:
            from rlmagent.model.external.cli import kill_tree

            kill_tree(proc)
            await proc.wait()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            return {
                "exit": None,
                "output": tail.text(),
                "timed_out": True,
                "message": f"timed out after {timeout:.0f}s",
            }
        return {
            "exit": proc.returncode,
            "output": tail.text(),
            "truncated": tail.truncated,
            "output_bytes": tail.total,
        }

    @staticmethod
    def _diff(path: str, before: str, after: str) -> str:
        return "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )

    async def instruction_docs(self, path: str | None = None):
        """Discover project instruction files, or read a selected file."""
        names = {"AGENTS.md", "AGENTS.override.md", "CLAUDE.md", "SKILL.md"}
        if path is not None:
            target = self._path(path)
            if target.name not in names:
                raise ValueError("choose an instruction file from instruction_docs()")
            if target.stat().st_size > MAX_INSTRUCTION:
                raise ValueError(f"{path} is larger than {MAX_INSTRUCTION} bytes")
            return await self.read_file(path)
        import os

        found = []
        for folder, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs if d not in SKIP_DIRS | {"dist", "build", ".hg", ".svn"}
            )
            for name in sorted(names.intersection(files)):
                target = self._path(str(Path(folder) / name))
                found.append(self._rel(target))
                if len(found) >= MAX_INSTRUCTIONS:
                    return found
        return found

    def functions(self) -> dict:
        return {
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_files": self.search_files,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "apply_patch": self.apply_patch,
            "ask_user": self.ask_user,
            "run": self.run,
            "instruction_docs": self.instruction_docs,
        }
