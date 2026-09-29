"""A run contract: every setting for a run in one JSON document, nothing implied."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

FORMAT = "rlmagent.run/1"
BUILTIN_TOOLS = ("python", "Read", "Write", "Edit")
_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}
_KEYLESS = {"ollama"}


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: str
    api_key_env: str | None
    base_url: str


@dataclass(frozen=True)
class Limits:
    max_depth: int
    max_calls: int
    max_cost: float
    max_live: int
    max_agents: int
    max_seconds: float | None
    cell_timeout: float


@dataclass(frozen=True)
class PromptSpec:
    system_file: str | None
    append: str | None
    append_sub_agent: str | None
    append_leaf: str | None


@dataclass(frozen=True)
class Sessions:
    save: bool
    dir: str | None


@dataclass(frozen=True)
class Trace:
    file: str


@dataclass(frozen=True)
class Contract:
    model: ModelSpec
    limits: Limits
    prompt: PromptSpec
    tools: tuple[str, ...]
    workdir: str
    sessions: Sessions
    trace: Trace
    raw: dict

    def api_key(self) -> str:
        if self.model.provider in _KEYLESS and self.model.api_key_env is None:
            return self.model.provider
        name = self.model.api_key_env
        value = os.environ.get(name or "")
        if not value:
            raise ContractError(f"model.api_key_env: environment variable {name} is not set")
        return value


def _fields(doc: object, where: str, spec: dict) -> dict:
    if not isinstance(doc, dict):
        raise ContractError(f"{where or 'contract'}: expected an object")
    for key in doc:
        if key not in spec:
            raise ContractError(f"{_join(where, key)}: unknown field")
    out = {}
    for key, kinds in spec.items():
        path = _join(where, key)
        if key not in doc:
            raise ContractError(f"{path}: missing (write null if it is not set)")
        out[key] = _check(doc[key], kinds, path)
    return out


def _check(value, kinds, path):
    if isinstance(value, bool) and bool not in kinds:
        raise ContractError(f"{path}: expected {_names(kinds)}, got a boolean")
    if value is None and type(None) in kinds:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and float in kinds:
        return float(value)
    if not isinstance(value, tuple(k for k in kinds if k is not type(None))):
        raise ContractError(f"{path}: expected {_names(kinds)}, got {type(value).__name__}")
    return value


def _names(kinds) -> str:
    return " or ".join("null" if k is type(None) else k.__name__ for k in kinds)


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


N = type(None)


def parse_contract(doc: object) -> Contract:
    top = _fields(doc, "", {
        "format": (str,), "model": (dict,), "limits": (dict,), "prompt": (dict,),
        "tools": (list,), "workdir": (str,), "sessions": (dict,), "trace": (dict,),
    })
    if top["format"] != FORMAT:
        raise ContractError(f"format: {top['format']!r} is not supported (expected {FORMAT!r})")
    m = _fields(top["model"], "model", {
        "provider": (str,), "name": (str,), "api_key_env": (str, N), "base_url": (str, N),
    })
    if m["provider"] not in _BASE_URLS:
        raise ContractError(f"model.provider: {m['provider']!r} is not one of {sorted(_BASE_URLS)}")
    if m["api_key_env"] is None and m["provider"] not in _KEYLESS:
        raise ContractError("model.api_key_env: required for this provider")
    lim = _fields(top["limits"], "limits", {
        "max_depth": (int,), "max_calls": (int,), "max_cost": (float,), "max_live": (int,),
        "max_agents": (int,), "max_seconds": (float, N), "cell_timeout": (float,),
    })
    pr = _fields(top["prompt"], "prompt", {
        "system_file": (str, N), "append": (str, N), "append_sub_agent": (str, N),
        "append_leaf": (str, N),
    })
    tools = top["tools"]
    bad = [t for t in tools if t not in BUILTIN_TOOLS]
    if bad or len(set(tools)) != len(tools):
        raise ContractError(f"tools: each must be one of {list(BUILTIN_TOOLS)}, once")
    ses = _fields(top["sessions"], "sessions", {"save": (bool,), "dir": (str, N)})
    tr = _fields(top["trace"], "trace", {"file": (str,)})
    return Contract(
        model=ModelSpec(m["provider"], m["name"], m["api_key_env"],
                        m["base_url"] or _BASE_URLS[m["provider"]]),
        limits=Limits(**lim),
        prompt=PromptSpec(**pr),
        tools=tuple(tools),
        workdir=top["workdir"],
        sessions=Sessions(**ses),
        trace=Trace(**tr),
        raw=doc,
    )


def load_contract(path: str | Path) -> Contract:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path} is not valid JSON: {exc}") from None
    return parse_contract(doc)
