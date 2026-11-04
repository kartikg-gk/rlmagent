from types import SimpleNamespace

import pytest

from rlmagent.cli import update


def test_update_requires_apply(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(update.sys, "prefix", str(tmp_path))
    package = SimpleNamespace(version="0.2.0", read_text={"INSTALLER": "pip"}.get)
    monkeypatch.setattr(update.importlib.metadata, "distribution", lambda _: package)
    calls = []

    def install(args, *, check):
        calls.append(args)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(update.subprocess, "run", install)
    monkeypatch.setattr(update.shutil, "which", lambda name: name)
    assert update.run([]) == 0
    assert "--apply" in capsys.readouterr().out
    assert calls == []
    assert update.run(["--apply"]) == 7
    assert calls == [[update.sys.executable, "-m", "pip", "install", "--upgrade", "rlmagent"]]


def test_update_refuses_a_direct_install(monkeypatch):
    package = SimpleNamespace(read_text={"direct_url.json": "{}"}.get)
    monkeypatch.setattr(update.importlib.metadata, "distribution", lambda _: package)
    with pytest.raises(SystemExit) as stopped:
        update.run(["--apply"])
    assert stopped.value.code == 1
