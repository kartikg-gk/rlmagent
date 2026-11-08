"""User-owned model cycles and per-model request settings, without credentials."""

import json
from uuid import uuid4

from rlmagent.discovery import _resolve_home
from rlmagent.refinement.store import _lock


def read(provider):
    try:
        data = json.loads((_resolve_home() / "model-preferences.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    return data.get(provider, {"cycle": [], "overrides": {}})


def change(provider, operation):
    path = _resolve_home() / "model-preferences.json"
    with _lock(path.with_suffix(".lock")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {}
        selected = data.setdefault(provider, {"cycle": [], "overrides": {}})
        operation(selected)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return selected


def tier(provider, model, value):
    change(
        provider, lambda selected: selected["overrides"].setdefault(model, {}).update(tier=value)
    )


def cycle(provider, operation, model=None):
    def update(selected):
        models = selected["cycle"]
        if operation == "add":
            if not model or len(model) > 256:
                raise ValueError("Supply a model identifier")
            if model not in models:
                if len(models) >= 100:
                    raise ValueError("Model cycle limit reached")
                models.append(model)
        elif operation == "remove":
            if model in models:
                models.remove(model)

    return change(provider, update)["cycle"]
