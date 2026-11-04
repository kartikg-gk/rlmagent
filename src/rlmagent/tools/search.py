"""Search tools: Grep for file contents, Glob for file names. Both only read."""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Mapping
from pathlib import Path

from rlmagent.tools._shared import (
    CancelToken,
    JValue,
    ProgressNotifier,
    ToolOutcome,
    ToolSpec,
    fault,
    optional_bool,
    optional_int,
    require_str,
    text_outcome,
)

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".rlm-agent"}
MAX_FILE_BYTES = 2_000_000
MAX_LINE_CHARS = 300
DEFAULT_MATCHES = 200
MAX_MATCHES = 1_000
MAX_NAMES = 1_000


def _start(arguments: Mapping[str, JValue]) -> Path:
    raw = arguments.get("path")
    path = Path(raw) if isinstance(raw, str) and raw.strip() else Path.cwd()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    return path


def _files(base: Path, glob: str | None):
    if base.is_file():
        yield base
        return
    for folder, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            file = Path(folder) / name
            if glob:
                rel = file.relative_to(base).as_posix()
                if not (fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(rel, glob)):
                    continue
            yield file


# ═══════════════════════════════════════════════════════════════════════════
# Grep tool
# ═══════════════════════════════════════════════════════════════════════════

_GREP_PARAMS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regular expression to look for."},
        "path": {
            "type": "string",
            "description": "File or folder to search. Defaults to the working directory.",
        },
        "glob": {
            "type": "string",
            "description": "Only search files whose name or relative path matches, e.g. *.py.",
        },
        "ignore_case": {"type": "boolean", "description": "Match without regard to case."},
        "limit": {
            "type": "integer",
            "description": f"Most matching lines to return (default {DEFAULT_MATCHES}).",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


async def _run_grep(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: ProgressNotifier | None = None,
) -> ToolOutcome:
    try:
        regex = re.compile(
            require_str(arguments, "pattern"),
            re.IGNORECASE if optional_bool(arguments, "ignore_case") else 0,
        )
        base = _start(arguments)
    except (ValueError, re.error) as exc:
        return fault(str(exc))
    glob = arguments.get("glob") if isinstance(arguments.get("glob"), str) else None
    limit = max(1, min(optional_int(arguments, "limit") or DEFAULT_MATCHES, MAX_MATCHES))
    lines: list[str] = []
    for file in _files(base, glob):
        try:
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            with open(file, encoding="utf-8") as f:
                for number, line in enumerate(f, 1):
                    if regex.search(line):
                        lines.append(f"{file}:{number}: {line.rstrip()[:MAX_LINE_CHARS]}")
                        if len(lines) >= limit:
                            lines.append(f"(stopped at {limit} matches; narrow the search)")
                            return text_outcome("\n".join(lines))
        except (UnicodeDecodeError, OSError):
            continue
    return text_outcome("\n".join(lines) if lines else "No matches.")


def make_grep_tool() -> ToolSpec:
    return ToolSpec(
        name="Grep",
        label="Search Files",
        description=(
            "Search file contents with a regular expression. Returns path:line: text "
            "for each matching line. Skips .git, virtual environments and binary files."
        ),
        parameters=_GREP_PARAMS,
        run=_run_grep,
        parallelism="parallel",
        format_call=lambda a: f"{a.get('pattern', '?')} in {a.get('path') or '.'}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Glob tool
# ═══════════════════════════════════════════════════════════════════════════

_GLOB_PARAMS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "File name pattern, e.g. **/*.py."},
        "path": {
            "type": "string",
            "description": "Folder to search in. Defaults to the working directory.",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


async def _run_glob(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: ProgressNotifier | None = None,
) -> ToolOutcome:
    try:
        pattern = require_str(arguments, "pattern")
        base = _start(arguments)
    except ValueError as exc:
        return fault(str(exc))
    names: list[str] = []
    for path in sorted(base.glob(pattern)):
        rel = path.relative_to(base)
        if path.is_file() and not SKIP_DIRS.intersection(rel.parts):
            names.append(str(path))
            if len(names) >= MAX_NAMES:
                names.append(f"(stopped at {MAX_NAMES} files; narrow the pattern)")
                break
    return text_outcome("\n".join(names) if names else "No files found.")


def make_glob_tool() -> ToolSpec:
    return ToolSpec(
        name="Glob",
        label="Find Files",
        description="Find files by name pattern, such as **/*.py or src/**/test_*.py.",
        parameters=_GLOB_PARAMS,
        run=_run_glob,
        parallelism="parallel",
        format_call=lambda a: f"{a.get('pattern', '?')} in {a.get('path') or '.'}",
    )


def search_tools() -> list[ToolSpec]:
    return [make_grep_tool(), make_glob_tool()]


__all__ = ["make_glob_tool", "make_grep_tool", "search_tools"]
