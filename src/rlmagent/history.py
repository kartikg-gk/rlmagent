"""The whole record of a session, readable from a cell after compaction has shortened it."""

from __future__ import annotations

import json
from pathlib import Path

MAX_ITEMS = 200
MAX_TEXT = 20_000


def _text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append(part.get("text", ""))
        elif kind == "toolCall":
            parts.append(f"[calls {part.get('name')}: {json.dumps(part.get('arguments'))[:2000]}]")
        elif kind == "image":
            parts.append("[image]")
    return "\n".join(parts)


def _rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def conversation(vault: Path) -> list[dict]:
    """Every message the session saved, in order, including ones compaction summarized away."""
    found = []
    for row in _rows(vault):
        if row.get("type") == "message":
            message = row.get("message") or {}
            found.append({"role": message.get("role", "?"),
                          "text": _text(message.get("content"))[:MAX_TEXT],
                          "time": row.get("timestamp")})
        elif row.get("type") == "compaction":
            found.append({"role": "compaction", "text": str(row.get("summary", ""))[:MAX_TEXT],
                          "time": row.get("timestamp")})
    return found


def agent_steps(events: Path, agent: str) -> list[dict]:
    """One agent's steps from the run log: its code and what the code printed."""
    steps: dict[int, dict] = {}
    for row in _rows(events):
        if row.get("run_id") != agent or not isinstance(row.get("step"), int):
            continue
        step = steps.setdefault(row["step"], {"step": row["step"], "code": "", "output": ""})
        if row.get("kind") == "code_generated":
            step["code"] = str(row.get("code") or "")[:MAX_TEXT]
        elif row.get("kind") == "output_received":
            step["output"] = str(row.get("output") or "")[:MAX_TEXT]
    return [steps[n] for n in sorted(steps)]


def make_history(vault_path, events_path):
    """history() for agents; the two callables give the current files, or None."""

    async def history(start: int = 0, end: int | None = None, agent: str | None = None) -> dict:
        """The full record, with nothing lost to compaction: messages[start:end] of this
        session, or with agent=RUN_ID that agent's steps (code and output). At most 200 items
        per call; "total" tells you how many there are."""
        if agent is None:
            path = vault_path()
            items = conversation(Path(path)) if path else []
        else:
            path = events_path()
            items = agent_steps(Path(path), str(agent)) if path else []
        stop = len(items) if end is None else int(end)
        picked = items[int(start):stop][:MAX_ITEMS]
        return {"total": len(items), "start": int(start), "items": picked}

    return history


__all__ = ["agent_steps", "conversation", "make_history"]
