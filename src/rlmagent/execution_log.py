"""Portable step records for execution-log viewers."""

from datetime import UTC, datetime


def records(event, history):
    kind = event["kind"]
    mapping = {
        "run_started": "agent_start",
        "run_failed": "agent_end",
        "run_completed": "agent_end",
        "code_generated": "code_generated",
        "step": "execution_result",
    }
    if kind not in mapping:
        return []
    agent = event.get("run_id", "root")
    start = next(
        (r for r in reversed(history) if r["kind"] == "run_started" and r.get("run_id") == agent),
        {},
    )
    row = {
        "level": 30,
        "time": datetime.fromtimestamp(event["timestamp"], UTC).isoformat(),
        "event_type": mapping[kind],
        "run_id": agent,
        "depth": start.get("depth", 0),
    }
    if start.get("parent_run_id"):
        row["parent_run_id"] = start["parent_run_id"]
    # Keep separate root executions distinct when a saved session appends turns.
    start_index = next(
        (i for i in range(len(history) - 1, -1, -1) if history[i] is start), len(history) - 1
    )
    root = next(
        (
            item
            for item in reversed(history[: start_index + 1])
            if item["kind"] == "run_started" and not item.get("parent_run_id")
        ),
        None,
    )
    if root is not None and "timestamp" in root:
        prefix = f"run-{int(root['timestamp'] * 1_000_000)}:"
        row["run_id"] = prefix + agent
        if row.get("parent_run_id"):
            row["parent_run_id"] = prefix + row["parent_run_id"]
    for field in ("step", "code", "output", "usage", "reasoning", "timestamps"):
        if field in event:
            row[field] = event[field]
    if kind == "step":
        row["hasError"] = bool(event.get("error", False))
    if kind == "run_failed":
        row["hasError"] = True
        row["error"] = event.get("error")
    result = []
    if kind == "run_completed":
        result.append(
            {
                **row,
                "event_type": "final_result",
                "result": event.get("result", event.get("output")),
            }
        )
    result.append(row)
    return result
