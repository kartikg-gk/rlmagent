"""What is known about a saved change: that it was saved, reached the model, and what followed."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

#: How many tasks either side of a change are compared.
WINDOW = 20
VERDICTS = ("better", "same", "worse")
#: A task with this many failed tool calls is worth a look for something to keep.
TROUBLE = 3


class EvidenceLog:
    """An append-only record per project. Nothing here changes what the agent does."""

    def __init__(self, path: Path | None):
        self.path = path
        self.rows: list[dict] = []
        self.last_failed = 0
        self._seen = {"transcript": 0, "events": 0}
        if path is not None:
            try:
                for line in Path(path).read_text(encoding="utf-8").splitlines():
                    try:
                        self.rows.append(json.loads(line))
                    except ValueError:
                        continue
            except OSError:
                pass

    def _add(self, row: dict) -> None:
        row = {"time": time.time(), **row}
        self.rows.append(row)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass  # a record about the run must never stop the run

    def saved(self, change: str, source: str, entries: list[str], summary: str = "") -> None:
        self._add({"type": "saved", "change": change, "source": source,
                   "entries": entries, "summary": summary})

    def rate(self, change: str, verdict: str) -> str:
        if verdict not in VERDICTS:
            return f"A verdict is one of {', '.join(VERDICTS)}."
        if not any(r["type"] == "saved" and r["change"] == change for r in self.rows):
            return f"No saved change {change}."
        self._add({"type": "verdict", "change": change, "verdict": verdict})
        return f"Change {change} rated {verdict}."

    def task_finished(self, session) -> None:
        """One finished task: which saved entries the model could read, and how the tools went."""
        calls, failed = self._tool_counts(session)
        self.last_failed = failed
        self._add({"type": "task", "calls": calls, "failed": failed,
                   "reached": sorted(self._reached(session))})

    def _tool_counts(self, session) -> tuple[int, int]:
        calls = failed = 0
        transcript = list(session.transcript)
        for message in transcript[self._seen["transcript"]:]:
            if hasattr(message, "is_error") and hasattr(message, "tool_call_id"):
                calls += 1
                failed += bool(message.is_error)
        self._seen["transcript"] = len(transcript)
        events = list(getattr(getattr(session, "backend", None), "events", None) or [])
        for event in events[self._seen["events"]:]:
            if event.get("kind") == "host_result":
                calls += 1
                failed += str(event.get("result", "")).startswith("Error:")
            elif event.get("kind") == "step" and event.get("step", 0) > 0 and event.get("code"):
                calls += 1
                failed += bool(event.get("error"))
        self._seen["events"] = len(events)
        return calls, failed

    def _reached(self, session) -> set[str]:
        """Entries actually in what the model is sent, read back rather than assumed."""
        names = {name for r in self.rows if r["type"] == "saved" for name in r["entries"]}
        if not names:
            return set()
        backend = getattr(session, "backend", None)
        sent = [getattr(session, "_system", "") or "", getattr(backend, "instructions", "") or ""]
        sent += [getattr(m, "text", "") or "" for m in session.transcript]
        text = "\n".join(sent)
        return {
            name for name in names
            if re.search(rf"(?:\] |Workflow |prompt |subagent ){re.escape(name)}\b", text)
        }

    def report(self) -> str:
        changes = [(n, r) for n, r in enumerate(self.rows) if r["type"] == "saved"]
        if not changes:
            return "No saved changes have been recorded for this project."
        # By position: clocks can give two neighbouring rows the same time.
        tasks = [(n, r) for n, r in enumerate(self.rows) if r["type"] == "task"]
        lines = []
        for position, change in changes[-10:]:
            before = [t for n, t in tasks if n < position][-WINDOW:]
            later = [t for n, t in tasks if n > position]
            after = later[:WINDOW]
            names = set(change["entries"])
            reached = sum(1 for t in later if names & set(t["reached"]))
            verdicts = [r["verdict"] for r in self.rows
                        if r["type"] == "verdict" and r["change"] == change["change"]]
            lines += [
                f"{change['change']} [{change['source']}] {', '.join(change['entries'])}",
                "  saved: " + time.strftime("%Y-%m-%d %H:%M", time.localtime(change["time"])),
                f"  reached the model: in {reached} of {len(later)} later tasks",
                f"  failed tool calls per task: {_rate(before)} before, {_rate(after)} after",
                f"  your verdict: {verdicts[-1] if verdicts else 'none yet'}",
            ]
        lines.append(
            "These counts are signals, not proof: tasks differ, and nothing here was a "
            "controlled test. Record a judgement with /refine rate <change> better|same|worse."
        )
        return "\n".join(lines)


def _rate(tasks: list[dict]) -> str:
    if not tasks:
        return "no data"
    return f"{sum(t['failed'] for t in tasks) / len(tasks):.2f} over {len(tasks)} tasks"
