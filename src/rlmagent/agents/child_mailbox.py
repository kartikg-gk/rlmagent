"""Read-state bookkeeping for parent-owned child notes."""

from time import time
from uuid import uuid4


def record_notes(notes, previous, current):
    """Fold a mailbox snapshot. Each runtime serializes its appends and full drains."""
    if previous == current:
        return notes
    if not current:
        pending = len(previous)
        return notes[:-pending] + [{**note, "status": "read"} for note in notes[-pending:]]
    return notes + [{"id": uuid4().hex, "text": text, "status": "unread", "time": time()}
                    for text in current[len(previous):]]


def notes_metadata(notes, unread_only):
    return [{k: v for k, v in note.items() if k != "text"}
            for note in notes if not unread_only or note["status"] == "unread"]


def read_note(notes, identity):
    for index, note in enumerate(notes):
        if note["id"] == identity:
            if note["status"] == "read":
                return notes, dict(note)
            read = {**note, "status": "read"}
            return notes[:index] + [read] + notes[index + 1:], dict(read)
    return notes, None
