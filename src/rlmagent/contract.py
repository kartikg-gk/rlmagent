"""A run contract: every setting for a run in one JSON document, nothing implied."""

from __future__ import annotations

import json
import copy
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

FORMAT = "rlmagent.run/1"
BUILTIN_TOOLS = ("python", "Read", "Write", "Edit")
_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "mistral": "https://api.mistral.ai/v1",
    "azure": "",
    "xai": "https://api.x.ai/v1",
    "codex": "https://api.openai.com/v1",
    "claude-subscription": "",
}
ACCOUNT_PROVIDERS = frozenset({"codex", "claude-subscription"})
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
    request_timeout: float = 300.0
    max_new_tokens: int | None = None
    strict_new_tokens: bool = False


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
class KernelSpec:
    kind: str = "local"
    approve: str = "deny"
    network: bool = False
    restore: bool = False


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
    kernel: KernelSpec = KernelSpec()
    playbook: dict | None = None
    engine: str = "session"
    policy: dict | None = None

    def api_key(self) -> str | None:
        if self.model.provider in ACCOUNT_PROVIDERS:
            return None
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
    doc = copy.deepcopy(doc)
    original = doc
    engine = "session"
    if isinstance(doc, dict) and "engine" in doc:
        engine = _check(doc["engine"], (str,), "engine")
        if engine not in ("session", "rlmness"):
            raise ContractError("engine: expected session or rlmness")
        doc = {key: value for key, value in doc.items() if key != "engine"}
    playbook = None
    if isinstance(doc, dict) and "playbook" in doc:
        playbook = _fields(doc["playbook"], "playbook", {"file": (str,), "update": (bool,)})
        if not playbook["file"].strip():
            raise ContractError("playbook.file: must not be empty")
        doc = {key: value for key, value in doc.items() if key != "playbook"}
    policy = None
    if isinstance(doc, dict) and "policy" in doc:
        policy_doc = doc["policy"]
        shell_access, shell_network, allowed_tools = "workspace", False, None
        if isinstance(policy_doc, dict):
            shell_access = _check(policy_doc.get("shell_access", "workspace"),
                                  (str,), "policy.shell_access")
            shell_network = _check(policy_doc.get("shell_network", False),
                                   (bool,), "policy.shell_network")
            if shell_access not in ("workspace", "host"):
                raise ContractError("policy.shell_access: expected workspace or host")
            allowed_tools = _check(policy_doc.get("allowed_tools"), (list, N),
                                   "policy.allowed_tools")
            if allowed_tools is not None and (
                any(not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z_0-9]*", name)
                    for name in allowed_tools)
                or len(set(allowed_tools)) != len(allowed_tools)
            ):
                raise ContractError("policy.allowed_tools: expected unique tool names")
            policy_doc = {key: value for key, value in policy_doc.items()
                          if key not in ("shell_access", "shell_network", "allowed_tools")}
        # Optional: every field must be given, null where the default applies.
        policy = _fields(policy_doc, "policy", {
            "allow_git": (bool,), "max_tool_output_bytes": (int, N),
            "max_compactions": (int, N), "agent_max_tokens": (int, N),
            "search_block": (list,),
        })
        if not all(isinstance(rule, str) for rule in policy["search_block"]):
            raise ContractError("policy.search_block: expected a list of strings")
        if "shell_access" in doc["policy"]:
            policy["shell_access"] = shell_access
        if "shell_network" in doc["policy"]:
            policy["shell_network"] = shell_network
        if allowed_tools is not None:
            policy["allowed_tools"] = allowed_tools
        doc = {key: value for key, value in doc.items() if key != "policy"}
    kernel = KernelSpec()
    if isinstance(doc, dict) and "kernel" in doc:
        # Optional, so contracts written before it still run the same way.
        kernel_doc = doc["kernel"]
        restore = False
        if isinstance(kernel_doc, dict) and "restore" in kernel_doc:
            restore = _check(kernel_doc["restore"], (bool,), "kernel.restore")
            kernel_doc = {key: value for key, value in kernel_doc.items() if key != "restore"}
        k = _fields(kernel_doc, "kernel", {"kind": (str,), "approve": (str,), "network": (bool,)})
        if k["kind"] not in ("wasm", "local"):
            raise ContractError(f"kernel.kind: {k['kind']!r} is not 'wasm' or 'local'")
        if k["approve"] not in ("deny", "auto"):
            raise ContractError(f"kernel.approve: {k['approve']!r} is not 'deny' or 'auto'")
        if restore and k["kind"] != "wasm":
            raise ContractError("kernel.restore: requires kernel.kind 'wasm'")
        kernel = KernelSpec(**k, restore=restore)
        doc = {key: value for key, value in doc.items() if key != "kernel"}
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
    account = m["provider"] in ACCOUNT_PROVIDERS
    if account and (m["api_key_env"] is not None or m["base_url"] is not None):
        raise ContractError("model: account routes require null api_key_env and base_url")
    if m["api_key_env"] is None and m["provider"] not in _KEYLESS and not account:
        raise ContractError("model.api_key_env: required for this provider")
    if m["provider"] == "azure" and not m["base_url"]:
        raise ContractError("model.base_url: required for the deployed Azure endpoint")
    address = m["base_url"] or _BASE_URLS[m["provider"]]
    parsed = urlsplit(address)
    if not account and (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ContractError("model.base_url: use an HTTP(S) endpoint without credentials, query or fragment")
    if m["provider"] == "claude-subscription":
        if re.fullmatch(r"cli:claude-code(?:\?model=[A-Za-z0-9][A-Za-z0-9._/-]{0,127})?", m["name"]) is None:
            raise ContractError("model.name: use cli:claude-code or cli:claude-code?model=MODEL")
    elif m["name"].startswith(("cli:", "acp:")):
        raise ContractError("model.name: this native-provider contract requires a native model identifier")
    limit_input = top["limits"]
    if isinstance(limit_input, dict):
        limit_input = {"request_timeout": 300.0, "max_new_tokens": None,
                       "strict_new_tokens": False, **limit_input}
    lim = _fields(limit_input, "limits", {
        "max_depth": (int,), "max_calls": (int,), "max_cost": (float,), "max_live": (int,),
        "max_agents": (int,), "max_seconds": (float, N), "cell_timeout": (float,),
        "request_timeout": (float,), "max_new_tokens": (int, N), "strict_new_tokens": (bool,),
    })
    if lim["strict_new_tokens"] and lim["max_new_tokens"] is None:
        raise ContractError("limits.strict_new_tokens: requires max_new_tokens")
    for name, value in lim.items():
        if value is None:
            continue
        if not math.isfinite(value) or value < 0:
            raise ContractError(f"limits.{name}: must be finite and nonnegative")
        if name in ("max_live", "cell_timeout", "request_timeout") and value == 0:
            raise ContractError(f"limits.{name}: must be positive")
    if policy is not None:
        for name in ("max_tool_output_bytes", "max_compactions", "agent_max_tokens"):
            if policy[name] is not None and policy[name] < 0:
                raise ContractError(f"policy.{name}: must be nonnegative or null")
    for name in ("name", "base_url"):
        if m[name] is not None and not m[name].strip():
            raise ContractError(f"model.{name}: must not be blank")
    pr = _fields(top["prompt"], "prompt", {
        "system_file": (str, N), "append": (str, N), "append_sub_agent": (str, N),
        "append_leaf": (str, N),
    })
    tools = top["tools"]
    bad = [t for t in tools if t not in BUILTIN_TOOLS]
    if bad or len(set(tools)) != len(tools):
        raise ContractError(f"tools: each must be one of {list(BUILTIN_TOOLS)}, once")
    if kernel.kind == "wasm" and {"Write", "Edit"} & set(tools):
        raise ContractError("tools: Write and Edit are not offered with kernel.kind 'wasm'; "
                            "the sandbox changes files through write_file and edit_file")
    if policy is not None and "allowed_tools" in policy and kernel.kind != "wasm":
        if "python" in tools and "python" in policy["allowed_tools"]:
            raise ContractError("policy.allowed_tools: restricted Python requires a Wasm kernel")
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
        raw=original,
        kernel=kernel,
        playbook=playbook,
        engine=engine,
        policy=policy,
    )


def load_contract(path: str | Path) -> Contract:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path} is not valid JSON: {exc}") from None
    return parse_contract(doc)
