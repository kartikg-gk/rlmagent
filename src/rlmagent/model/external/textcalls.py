"""The python tool through plain text, for agents that cannot call our tools."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rlmagent.harness.contracts.transcript import TranscriptEntry, gather_text

PROTOCOL = """## Running code
You have no tools of your own here; ignore any you think you have. Running a python block is how you use the python tool: the host runs it in a persistent kernel that has the project files and every function named above. To run code, reply with one block:

```python
print("hello")
```

Its output comes back in the next message. Put at most one block in a reply; text before it \
is shown to the user. A reply with no block is your final answer."""

_BLOCK = re.compile(r"```(?:python|py|repl)[ \t]*\r?\n(.*?)```", re.S)


def parse_reply(text: str) -> tuple[str, str | None]:
    """Split a reply into the text before its first python block and the block's code."""
    match = _BLOCK.search(text)
    if match is None:
        return text.strip(), None
    return text[: match.start()].strip(), match.group(1)


def _assistant(entry) -> str:
    parts = []
    for block in entry.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "toolCall":
            parts.append(f"```python\n{block.arguments.get('code', '')}\n```")
    return "\n".join(parts)


def render(messages: Sequence[TranscriptEntry]) -> str:
    """The conversation as one prompt, oldest first."""
    parts = []
    for entry in messages:
        role = getattr(entry, "role", "")
        if role == "user":
            parts.append(f"[user]\n{entry.text}")
        elif role == "assistant":
            parts.append(f"[assistant]\n{_assistant(entry)}")
        elif role == "toolResult":
            label = "output" if not entry.is_error else "error"
            parts.append(f"[{entry.tool_name} {label}]\n{gather_text(entry.content)}")
        elif role == "custom":
            parts.append(f"[note]\n{entry.text}")
        elif role in ("compactionSummary", "branchSummary"):
            parts.append(f"[summary of earlier conversation]\n{entry.summary}")
    return "\n\n".join(parts)
