import copy
import json

import pytest

from rlmagent_app.contract import ContractError, load_contract, parse_contract

VALID = {
    "format": "rlmagent.run/1",
    "model": {"provider": "openrouter", "name": "deepseek/deepseek-v4-flash",
              "api_key_env": "TEST_KEY", "base_url": None},
    "limits": {"max_depth": 2, "max_calls": 200, "max_cost": 2.0, "max_live": 8,
               "max_agents": 50, "max_seconds": None, "cell_timeout": 300},
    "prompt": {"system_file": None, "append": None, "append_sub_agent": None,
               "append_leaf": None},
    "tools": ["python", "Read", "Write", "Edit"],
    "workdir": ".",
    "sessions": {"save": False, "dir": None},
    "trace": {"file": "trace.jsonl"},
}


def _with(path, value):
    doc = copy.deepcopy(VALID)
    *parents, last = path.split(".")
    target = doc
    for key in parents:
        target = target[key]
    if value is _DROP:
        del target[last]
    else:
        target[last] = value
    return doc


_DROP = object()


def test_a_valid_contract_is_accepted():
    c = parse_contract(VALID)
    assert c.model.name == "deepseek/deepseek-v4-flash"
    assert c.limits.max_cost == 2.0 and c.limits.max_seconds is None
    assert c.tools == ("python", "Read", "Write", "Edit")
    assert c.model.base_url == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize("path, value, named", [
    ("limits.max_calls", _DROP, "limits.max_calls"),
    ("model.api_key_env", _DROP, "model.api_key_env"),
    ("limits.max_cost", "2", "limits.max_cost"),
    ("limits.max_depth", 1.5, "limits.max_depth"),
    ("sessions.save", "no", "sessions.save"),
    ("tools", ["python", "Bash"], "tools"),
    ("model.provider", "acme", "model.provider"),
    ("format", "rlmagent.run/2", "format"),
])
def test_bad_fields_are_refused_by_name(path, value, named):
    with pytest.raises(ContractError, match=named.replace(".", r"\.")):
        parse_contract(_with(path, value))


def test_unknown_fields_are_refused_by_name():
    doc = _with("limits.max_tokens", 5)
    with pytest.raises(ContractError, match=r"limits\.max_tokens"):
        parse_contract(doc)
    with pytest.raises(ContractError, match="extra"):
        parse_contract({**VALID, "extra": 1})


def test_booleans_are_not_numbers():
    with pytest.raises(ContractError, match=r"limits\.max_calls"):
        parse_contract(_with("limits.max_calls", True))


def test_load_reads_a_file_and_reports_bad_json(tmp_path):
    good = tmp_path / "run.json"
    good.write_text(json.dumps(VALID))
    assert load_contract(good).trace.file == "trace.jsonl"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ContractError, match="not valid JSON"):
        load_contract(bad)


def test_the_key_comes_from_the_named_variable(monkeypatch):
    c = parse_contract(VALID)
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ContractError, match="TEST_KEY"):
        c.api_key()
    monkeypatch.setenv("TEST_KEY", "k")
    assert c.api_key() == "k"


def test_ollama_needs_no_key(monkeypatch):
    c = parse_contract(_with("model", {"provider": "ollama", "name": "llama3.3",
                                       "api_key_env": None, "base_url": None}))
    assert c.api_key() == "ollama"
