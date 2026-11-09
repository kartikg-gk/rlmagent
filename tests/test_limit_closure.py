import json
import os
import time
from types import SimpleNamespace

import pytest

from rlmagent import packages
from rlmagent.cli import update


def test_package_prune_preserves_active_recent_loaded_and_unowned_directories(
    tmp_path, monkeypatch
):
    active, expired, recent, loaded = (character * 32 for character in "abcd")
    for name in (active, expired, recent, loaded, "unrelated"):
        path = tmp_path / name
        path.mkdir()
        (path / "resource").write_text("keep or collect")
        if name != recent:
            old = time.time() - 40 * 86400
            os.utime(path, (old, old))
    monkeypatch.syspath_prepend(str(tmp_path / loaded))
    records = {"current": {"directory": active}}
    assert packages.prune(tmp_path, records) == [expired]
    assert (tmp_path / expired).exists()
    assert packages.prune(tmp_path, records, apply=True) == [expired]
    assert not (tmp_path / expired).exists()
    assert all((tmp_path / name).exists() for name in (active, recent, loaded, "unrelated"))


def test_package_prune_cli_previews_and_rejects_invalid_retention(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    target = packages.root() / ("e" * 32)
    target.mkdir(parents=True)
    assert packages.run(["prune", "--older-than", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == [target.name]
    assert target.exists()
    assert packages.run(["prune", "--older-than", "nan", "--apply"]) == 1
    assert target.exists()
    assert packages.run(["prune", "--older-than", "0", "--apply"]) == 0
    assert not target.exists()


def test_package_mutation_collects_expired_cache_after_committing_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    store = packages.root()
    active = store / ("f" * 32)
    expired = store / ("e" * 32)
    active.mkdir(parents=True)
    expired.mkdir()
    old = time.time() - 40 * 86400
    os.utime(expired, (old, old))
    packages._save({"fixture": {"directory": active.name}}, store)
    assert packages.run(["remove", "fixture"]) == 0
    assert packages._records(store) == {}
    assert active.exists() and not expired.exists()


def test_package_prune_never_follows_junctions(tmp_path, monkeypatch):
    from pathlib import Path

    linked = tmp_path / ("a" * 32)
    linked.mkdir()
    old = time.time() - 40 * 86400
    os.utime(linked, (old, old))
    monkeypatch.setattr(Path, "is_junction", lambda path: path == linked)
    assert packages.prune(tmp_path, {}, apply=True) == []
    assert linked.exists()


def catalog():
    return {
        "releases": {
            "1.0": [{}],
            "1.1rc1": [{}],
            "1.2.dev20261105": [{}],
            "1.2.dev20261106": [{"yanked": True}],
            "9.0.dev1": [{"requires_python": ">=99"}],
            "10.0.dev1": [],
            "invalid": [{}],
        }
    }


@pytest.mark.parametrize(
    "channel,version",
    [
        ("stable", "1.0"),
        ("preview", "1.1rc1"),
        ("nightly", "1.2.dev20261105"),
    ],
)
def test_release_channels_select_only_compatible_published_artifacts(channel, version):
    assert update.select_release(catalog(), channel) == version


def test_nightly_update_pins_dev_release_and_never_substitutes_an_rc(tmp_path, monkeypatch):
    from rlmagent.model.transport import client

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    monkeypatch.setattr(update.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        update.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            version="1.0",
            read_text={"INSTALLER": "pip"}.get,
        ),
    )
    monkeypatch.setattr(client, "fetch_json", lambda *args, **kw: catalog())
    calls = []
    monkeypatch.setattr(update.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        update.subprocess,
        "run",
        lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert update.run(["--channel", "nightly", "--apply"]) == 0
    assert "rlmagent==1.2.dev20261105" in calls[0]
    with pytest.raises(ValueError, match="No compatible"):
        update.select_release({"releases": {"1.1rc1": [{}]}}, "nightly")


def test_offline_nightly_update_does_not_query_the_index(tmp_path, monkeypatch):
    from rlmagent.model.transport import client

    monkeypatch.setenv("RLM_AGENT_OFFLINE", "1")
    monkeypatch.setattr(
        update.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            version="1.0",
            read_text=lambda name: None,
        ),
    )
    monkeypatch.setattr(client, "fetch_json", lambda *args, **kw: pytest.fail("Unexpected network"))
    with pytest.raises(SystemExit) as stopped:
        update.run(["--channel", "nightly", "--apply"])
    assert stopped.value.code == 1
