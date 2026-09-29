"""Check proposed playbook edits and apply the valid ones; build the edits that undo a change."""

from __future__ import annotations

import ast
import time
import uuid
from dataclasses import asdict

from rlmagent_app.playbook.store import KINDS, Entry, Playbook

ACTIONS = ("create", "update", "delete")


def validate(edit: dict) -> str | None:
    action, kind = edit.get("action"), edit.get("kind")
    if action not in ACTIONS:
        return f"action must be one of {ACTIONS}"
    if not str(edit.get("id") or "").strip():
        return "id is required"
    if action == "delete":
        return None
    if kind not in KINDS:
        return f"kind must be one of {KINDS}"
    if not str(edit.get("title") or "").strip() or not str(edit.get("content") or "").strip():
        return "title and content are required"
    if kind == "skill":
        name = edit.get("callable")
        if not name:
            return "a skill needs `callable`: the function its code defines"
        try:
            tree = ast.parse(edit["content"])
        except SyntaxError as exc:
            return f"skill content is not valid Python: {exc.msg}"
        defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if name not in defined:
            return f"skill content never defines {name}()"
    return None


def apply_edits(pb: Playbook, edits: list[dict], *, baseline: dict[str, int], scope: str,
                trigger: str, summary: str, rationale: str = "",
                read_only: Playbook | None = None) -> dict:
    """Apply what is valid; record every edit, applied or not, in `pb.history`."""
    changes = []
    for edit in edits:
        eid = str(edit.get("id") or "").removeprefix("local:")
        before = pb.entries.get(eid)
        error = validate(edit)
        if error is None:
            error = _conflict(edit, eid, before, baseline, read_only)
        change = {"action": edit.get("action"), "id": eid, "kind": edit.get("kind"),
                  "applied": error is None, "error": error,
                  "before": asdict(before) if before else None, "after": None}
        if error is None:
            if edit["action"] == "delete":
                del pb.entries[eid]
            else:
                entry = Entry(
                    id=eid, kind=edit["kind"], title=edit["title"], content=edit["content"],
                    scope=scope, callable=edit.get("callable"),
                    version=(before.version + 1) if before else 1, updated=time.time(),
                )
                pb.entries[eid] = entry
                change["after"] = asdict(entry)
        changes.append(change)
    event = {"id": uuid.uuid4().hex[:8], "trigger": trigger, "scope": scope, "summary": summary,
             "rationale": rationale, "time": time.time(), "changes": changes}
    pb.history.append(event)
    return event


def _conflict(edit, eid, before, baseline, read_only) -> str | None:
    action = edit["action"]
    if read_only is not None and eid in read_only.entries and eid not in baseline and before is None:
        return "shared entries are read-only here; create a local entry instead"
    if action == "create" and before is not None:
        return "an entry with this id already exists"
    if action != "create" and before is None:
        return "no entry with this id"
    if before is not None and eid in baseline and baseline[eid] != before.version:
        return "entry changed while this edit was being planned"
    return None


def reverse_edits(event: dict) -> list[dict]:
    """Edits that undo the applied changes of `event`, newest first."""
    out = []
    for change in reversed(event["changes"]):
        if not change["applied"]:
            continue
        if change["action"] == "create":
            out.append({"action": "delete", "id": change["id"]})
        elif change["action"] == "update":
            out.append({"action": "update", **_as_edit(change["before"])})
        else:
            out.append({"action": "create", **_as_edit(change["before"])})
    return out


def _as_edit(entry: dict) -> dict:
    return {k: entry[k] for k in ("id", "kind", "title", "content", "callable")}
