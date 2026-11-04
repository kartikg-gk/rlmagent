"""Short notes added to a cell's output when it failed in a way models often repeat."""

from __future__ import annotations

import re

_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"f-string: (expecting|unmatched|single '\}')|f-string: .*quote", re.I),
     "Inside an f-string, use the other kind of quote for strings in the braces, or build "
     "the value in a variable first."),
    (re.compile(r"unterminated (triple-quoted )?string literal", re.I),
     "A string was not closed. For text with quotes or many lines, use triple quotes or "
     "write it to a variable in pieces."),
    (re.compile(r"coroutine .* was never awaited|<coroutine object", re.I),
     "A host function was called without await. Write `await read_file(...)`, not "
     "`read_file(...)`."),
    (re.compile(r"object (dict|list|str|NoneType) can't be used in 'await' expression", re.I),
     "That function is not a coroutine; call it without await."),
    (re.compile(r"'await' outside (async )?function", re.I),
     "Top-level await works in a cell, but not inside a plain def; make it `async def` and "
     "await the call."),
    (re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'"),
     "The module is not installed in this kernel. In the sandbox, try "
     "`import micropip; await micropip.install(...)` when the network is on, or do the work "
     "with run() on the host."),
    (re.compile(r"no response within \d+s|timed out after", re.I),
     "The cell ran past its time limit. Start long commands with job_start(command) and "
     "read them with job_read, or split the work into smaller cells."),
    (re.compile(r"NameError: name 'FINAL' is not defined"),
     "FINAL is only defined at the top level of the kernel; call it outside functions."),
]

EMPTY_CELL = ("The reply had no code to run. Answer with a Python cell; call FINAL(answer) "
              "when the task is done.")


def hint_for(code: str | None, output: str) -> str | None:
    """One hint for what went wrong, or None when nothing familiar shows."""
    if code is not None and not code.strip():
        return EMPTY_CELL
    for pattern, hint in _RULES:
        if pattern.search(output):
            return hint
    return None


__all__ = ["EMPTY_CELL", "hint_for"]
