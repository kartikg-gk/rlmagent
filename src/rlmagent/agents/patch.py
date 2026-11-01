"""Apply a unified diff to several files: everything is checked before anything is written."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


class PatchError(ValueError):
    pass


@dataclass
class Hunk:
    start: int  # 1-based line in the old file; 0 when it has no old lines
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)


@dataclass
class FilePatch:
    old_path: str | None  # None: the file is created
    new_path: str | None  # None: the file is deleted
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path


def _name(raw: str) -> str | None:
    name = raw.split("\t")[0].strip()
    if name == "/dev/null":
        return None
    return name[2:] if name.startswith(("a/", "b/")) else name


def parse(diff: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            raise PatchError(f"line {i + 1}: '---' without a following '+++'")
        current = FilePatch(_name(lines[i][4:]), _name(lines[i + 1][4:]))
        if current.old_path is None and current.new_path is None:
            raise PatchError(f"line {i + 1}: both sides are /dev/null")
        files.append(current)
        i += 2
        while i < len(lines) and lines[i].startswith("@@"):
            found = _HUNK.match(lines[i])
            if found is None:
                raise PatchError(f"line {i + 1}: bad hunk header")
            hunk = Hunk(int(found.group(1)))
            current.hunks.append(hunk)
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ")):
                line = lines[i]
                if line.startswith("\\"):
                    pass  # "No newline at end of file": the file keeps its own ending
                elif line.startswith("-"):
                    hunk.old.append(line[1:])
                elif line.startswith("+"):
                    hunk.new.append(line[1:])
                elif line.startswith(" ") or not line:
                    hunk.old.append(line[1:])
                    hunk.new.append(line[1:])
                else:
                    break  # the text after the last hunk, such as a "diff --git" line
                i += 1
    if not files:
        raise PatchError("no file changes found; expected '---' and '+++' lines")
    return files


def _locate(lines: list[str], hunk: Hunk, floor: int, path: str) -> int:
    size = len(hunk.old)
    stated = max(hunk.start - 1, 0)
    if stated >= floor and lines[stated:stated + size] == hunk.old:
        return stated
    places = [at for at in range(floor, len(lines) - size + 1) if lines[at:at + size] == hunk.old]
    if len(places) == 1:
        return places[0]
    what = "not found" if not places else f"found {len(places)} times"
    raise PatchError(f"{path}: the hunk at line {hunk.start} was {what}; read the file again")


def patched(text: str, patch: FilePatch) -> str:
    """The file's new text; raises PatchError when a hunk does not fit."""
    newline = "\r\n" if "\r\n" in text else "\n"
    ends_open = bool(text) and not text.endswith("\n")
    lines = text.splitlines()
    result: list[str] = []
    position = 0
    for hunk in patch.hunks:
        at = _locate(lines, hunk, position, patch.path)
        result += lines[position:at] + hunk.new
        position = at + len(hunk.old)
    result += lines[position:]
    if not result:
        return ""
    return newline.join(result) + ("" if ends_open else newline)
