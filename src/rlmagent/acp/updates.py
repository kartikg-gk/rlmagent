"""Our events turned into ACP session/update payloads."""

from __future__ import annotations

from rlmagent.harness.contracts.stream import (
    MessageEndEvent,
    MessageUpdateEvent,
    RetryEvent,
    ToolRunEndEvent,
    ToolRunStartEvent,
    ToolRunUpdateEvent,
)
from rlmagent.harness.contracts.transcript.entries import ModelEntry
from rlmagent.harness.contracts.transcript.segments import TextSegment, ThoughtSegment

KINDS = {"python": "execute", "Read": "read", "Write": "edit", "Edit": "edit"}
PLAN_STATUS = {"starting": "pending", "running": "in_progress", "idle": "in_progress"}


def chunk(kind: str, text: str) -> dict:
    return {"sessionUpdate": kind, "content": {"type": "text", "text": text}}


def _title(name: str, args: dict) -> str:
    if name == "python":
        first = str(args.get("code", "")).strip().splitlines()
        return first[0][:80] if first else "python"
    path = args.get("path") or args.get("file_path")
    return f"{name} {path}" if path else name


def _text_content(text: str) -> list[dict]:
    return [{"type": "content", "content": {"type": "text", "text": text}}] if text else []


class Translator:
    """Keeps how much of the current reply was already sent, so updates carry only deltas."""

    def __init__(self) -> None:
        self.sent_text = 0
        self.sent_thought = 0

    def _deltas(self, msg: ModelEntry) -> list[dict]:
        text = "".join(b.text for b in msg.content if isinstance(b, TextSegment))
        thought = "".join(b.thinking for b in msg.content if isinstance(b, ThoughtSegment))
        out = []
        if len(thought) > self.sent_thought:
            out.append(chunk("agent_thought_chunk", thought[self.sent_thought:]))
            self.sent_thought = len(thought)
        if len(text) > self.sent_text:
            out.append(chunk("agent_message_chunk", text[self.sent_text:]))
            self.sent_text = len(text)
        return out

    def updates(self, event) -> list[dict]:
        if isinstance(event, MessageUpdateEvent) and isinstance(event.message, ModelEntry):
            return self._deltas(event.message)
        if isinstance(event, MessageEndEvent) and isinstance(event.message, ModelEntry):
            out = self._deltas(event.message)
            if event.message.stop_reason == "error" and event.message.error_message:
                out.append(chunk("agent_message_chunk", f"\n\nError: {event.message.error_message}"))
            self.sent_text = self.sent_thought = 0
            return out
        if isinstance(event, ToolRunStartEvent):
            update = {"sessionUpdate": "tool_call", "toolCallId": event.tool_call_id,
                      "title": _title(event.tool_name, event.args),
                      "kind": KINDS.get(event.tool_name, "other"), "status": "in_progress",
                      "rawInput": event.args}
            path = event.args.get("path") or event.args.get("file_path")
            if path:
                update["locations"] = [{"path": str(path)}]
            return [update]
        if isinstance(event, ToolRunUpdateEvent):
            return [{"sessionUpdate": "tool_call_update", "toolCallId": event.tool_call_id,
                     "content": _text_content(event.partial_result.text)}]
        if isinstance(event, ToolRunEndEvent):
            return [{"sessionUpdate": "tool_call_update", "toolCallId": event.tool_call_id,
                     "status": "failed" if event.is_error else "completed",
                     "content": _text_content(event.result.text)}]
        if isinstance(event, RetryEvent):
            return [chunk("agent_thought_chunk", f"[retry] {event.message}\n")]
        return []


def stop_reason(messages: list) -> str:
    last = next((m for m in reversed(messages) if isinstance(m, ModelEntry)), None)
    if last is None:
        return "end_turn"
    if last.stop_reason == "aborted":
        return "cancelled"
    if last.stop_reason == "length":
        return "max_tokens"
    if last.stop_reason == "error" and any(
            w in (last.error_message or "").lower() for w in ("budget", "limit", "allowance")):
        return "max_turn_requests"
    return "end_turn"


def plan(tree) -> list[dict]:
    """Every sub-agent as one plan entry, nested ones indented by depth."""
    entries = []
    for rec in tree.records.values():
        node = rec.node
        status = PLAN_STATUS.get(rec.status, "completed")
        suffix = f" ({rec.status})" if rec.status in ("failed", "cancelled") else ""
        task = (node.task.strip().splitlines() or [node.id])[0][:120]
        entries.append({"content": "  " * max(node.depth - 1, 0) + task + suffix,
                        "priority": "medium", "status": status})
    return entries


def replay(transcript) -> list[dict]:
    """A loaded session's history as message updates."""
    out = []
    for msg in transcript:
        role = getattr(msg, "role", "")
        if role == "user":
            out.append(chunk("user_message_chunk", msg.text))
        elif role == "assistant" and msg.text:
            out.append(chunk("agent_message_chunk", msg.text))
    return out
