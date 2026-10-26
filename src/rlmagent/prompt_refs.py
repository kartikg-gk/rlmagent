"""Files and links a prompt mentions, loaded before the turn so the agent starts with them."""

from __future__ import annotations

import re
from pathlib import Path

MAX_FILES = 10
MAX_BYTES = 200_000

_LINK = re.compile(r"https?://[^\s)>\]'\"`]+")
_CANDIDATES = (
    re.compile(r"`([^`\n]+)`"),          # `path`
    re.compile(r"@([^\s,;]+)"),            # @path
    re.compile(r"\]\(([^)\s]+)\)"),        # [text](path)
    re.compile(r"(?<![\w/.@])((?:[\w.-]+[/\\])*[\w.-]+\.\w{1,8})(?![\w/])"),  # plain a/b.py
)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def references(text: str, root: str | Path) -> dict:
    """`{"prompt", "links", "files": [{"path", "content"}]}`; missing, binary, large and
    outside-the-workspace files are left out."""
    root = Path(root).resolve()
    links = list(dict.fromkeys(m.rstrip(".,") for m in _LINK.findall(text)))
    seen, files = set(), []
    for pattern in _CANDIDATES:
        for raw in pattern.findall(text):
            raw = raw.strip().rstrip(".,:;")
            if not raw or raw.startswith(("http://", "https://")) or len(files) >= MAX_FILES:
                continue
            path = (root / raw).resolve()
            if path in seen or not _inside(path, root) or not path.is_file():
                continue
            seen.add(path)
            if path.stat().st_size > MAX_BYTES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            files.append({"path": path.relative_to(root).as_posix(), "content": content})
    return {"prompt": text, "links": links, "files": files}


def note(refs: dict) -> str:
    """The line added to the prompt so the agent knows what is waiting in its kernel."""
    parts = []
    if refs["files"]:
        parts.append("PROMPT['files'] holds " + ", ".join(f["path"] for f in refs["files"]))
    if refs["links"]:
        parts.append("PROMPT['links'] holds the links")
    return f"\n\n[Preloaded in the python kernel: {'; '.join(parts)}.]" if parts else ""
