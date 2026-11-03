"""The task list an agent keeps for itself while it works, saved with the session."""

from __future__ import annotations

import json
from pathlib import Path

MAX_TASKS = 100
MAX_TEXT = 300


class TaskList:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.items: list[dict] = []
        self.on_change = None  # called after every change, e.g. to redraw a panel
        self.load()

    def attach(self, path: Path | None) -> None:
        """Keep the list in `path` from now on, reading what is already there."""
        self.path = path
        self.load()

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, list):
            self.items = [
                {"text": str(item.get("text", ""))[:MAX_TEXT], "done": bool(item.get("done"))}
                for item in data if isinstance(item, dict)
            ][:MAX_TASKS]

    def _save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items, indent=1), encoding="utf-8")
        if self.on_change is not None:
            self.on_change(self)

    def _index(self, number) -> int:
        index = int(number) - 1
        if not 0 <= index < len(self.items):
            raise ValueError(f"there is no task {number}; tasks are numbered 1 to {len(self.items)}")
        return index

    def add(self, text: str) -> int:
        text = " ".join(str(text).split())[:MAX_TEXT]
        if not text:
            raise ValueError("a task needs some text")
        if len(self.items) >= MAX_TASKS:
            raise ValueError(f"the list already has {MAX_TASKS} tasks; clear finished ones")
        self.items.append({"text": text, "done": False})
        self._save()
        return len(self.items)

    def done(self, number) -> None:
        self.items[self._index(number)]["done"] = True
        self._save()

    def undo(self, number) -> None:
        self.items[self._index(number)]["done"] = False
        self._save()

    def remove(self, number) -> None:
        del self.items[self._index(number)]
        self._save()

    def clear(self, finished_only: bool = True) -> None:
        self.items = [item for item in self.items if finished_only and not item["done"]]
        self._save()

    def render(self) -> str:
        if not self.items:
            return "No tasks."
        return "\n".join(f"{n}. [{'x' if item['done'] else ' '}] {item['text']}"
                         for n, item in enumerate(self.items, 1))

    async def todo(self, action: str = "list", value=None):
        """Keep your own task list: todo() lists it; todo("add", text);
        todo("done", n); todo("undo", n); todo("remove", n); todo("clear") drops finished ones."""
        if action == "list":
            pass
        elif action == "add":
            self.add(value)
        elif action in ("done", "undo", "remove"):
            getattr(self, action)(value)
        elif action == "clear":
            self.clear()
        else:
            raise ValueError("action is one of list, add, done, undo, remove, clear")
        return self.render()


__all__ = ["TaskList"]
