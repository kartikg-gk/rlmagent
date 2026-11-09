"""Why a run stopped, as one word a training or evaluation pipeline can act on.

done              the agent answered
max_calls         the tree used up its model calls
max_cost          the tree used up its spend
max_tokens        the tree used up its completion or prompt tokens
max_seconds       the run used up its time
agent_tokens      one agent used up its completion tokens
compaction_failed the context was full and could not be compacted
no_answer         the run ended without a usable answer (an empty or failed final reply)
cancelled         the user or client stopped it
error             anything else, with the message beside it
"""

from __future__ import annotations

_LIMITS = (
    ("calls reaches the limit", "max_calls"),
    ("cost ", "max_cost"),
    ("completion tokens", "max_tokens"),
    ("prompt tokens", "max_tokens"),
    ("tokens", "max_tokens"),
    ("s reaches the limit", "max_seconds"),
)


def classify(refusal: str | None, final_text: str, error: str | None,
             cancelled: bool = False) -> str:
    if cancelled or (error or "").lower() in ("cancelled", "aborted"):
        return "cancelled"
    text = f"{refusal or ''} {error or ''}"
    if "reached its limit of" in text and "completion tokens" in text:
        return "agent_tokens"
    if refusal:
        for needle, reason in _LIMITS:
            if needle in refusal:
                return reason
        return "max_calls"
    if "compaction" in (error or "").lower():
        return "compaction_failed"
    if error:
        return "error"
    if not final_text.strip():
        return "no_answer"
    return "done"


def of_session(session, refusal: str | None) -> dict:
    """The stop reason of the session's last turn, with the message that explains it."""
    last = next((m for m in reversed(list(session.transcript))
                 if getattr(m, "role", "") == "assistant"), None)
    text = getattr(last, "text", "") or ""
    error = getattr(last, "error_message", None)
    stop = getattr(last, "stop_reason", None)
    reason = classify(refusal, text, error, cancelled=stop == "aborted")
    typed = getattr(getattr(session, "backend", None), "last_stop_reason", None)
    if typed and stop != "aborted":
        reason = typed
    return {"stop_reason": reason, "stop_detail": refusal or error or ""}


__all__ = ["classify", "of_session"]
