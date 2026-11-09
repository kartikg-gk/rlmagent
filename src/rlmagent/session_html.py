"""A self-contained, escaped transcript for sharing as a local file."""

import json
from html import escape

from rlmagent.harness.contracts.transcript import surface_text


def render(messages, title="Session", *, records=None) -> str:
    cards = []
    if records is not None:
        messages = []
        from types import SimpleNamespace

        for record in records:
            if record.type == "message":
                messages.append(record.message)
            elif record.type in ("compaction", "branch_summary"):
                messages.append(
                    SimpleNamespace(
                        role="Context summary" if record.type == "compaction" else "Branch summary",
                        text=record.summary,
                    )
                )
    for message in messages:
        text = getattr(message, "text", None)
        if text is None:
            text = surface_text(message)
        for call in getattr(message, "tool_calls", []):
            text += (
                "\n\nTool: "
                + call.name
                + "\n"
                + json.dumps(call.arguments, ensure_ascii=False, indent=2)
            )
        cards.append(
            "<article><h2>"
            + escape(message.role)
            + "</h2><pre>"
            + escape(text)
            + "</pre></article>"
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>" + escape(title) + "</title><style>"
        "body{max-width:900px;margin:2rem auto;padding:0 1rem;font:16px system-ui;"
        "background:#111827;color:#e5e7eb}article{border-top:1px solid #374151;"
        "padding:1rem 0}h1,h2{font-size:1.2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "font:14px ui-monospace,monospace}</style><h1>"
        + escape(title)
        + "</h1>"
        + "".join(cards)
        + "</html>"
    )
