"""Portable step records for execution-log viewers."""

from datetime import UTC, datetime


def records(event, history):
    kind = event["kind"]
    mapping = {
        "run_started": "agent_start",
        "run_failed": "agent_end",
        "run_completed": "agent_end",
        "code_generated": "code_generated",
        "output_received": "execution_result",
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
    for field in ("step", "code", "output", "usage", "reasoning", "timestamps"):
        if field in event:
            row[field] = event[field]
    if kind == "output_received":
        row["hasError"] = bool(event.get("error", False))
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
