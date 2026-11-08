import base64
import csv
import hashlib
import io
import json
from types import SimpleNamespace
from zipfile import ZipFile

import httpx
import pytest


def wheel(path, *, version="1.0", text="First resource"):
    metadata = f"sample_resources-{version}.dist-info"
    files = {
        "rlmagent-resources.json": json.dumps({"prompts": ["prompts"]}),
        "prompts/check.md": text,
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: sample-resources\nVersion: {version}\n",
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = io.StringIO()
    writer = csv.writer(record)
    for name, value in files.items():
        data = value.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, f"sha256={digest}", len(data)])
    writer.writerow([f"{metadata}/RECORD", "", ""])
    files[f"{metadata}/RECORD"] = record.getvalue()
    with ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return path


def test_real_capability_wheel_install_update_remove(tmp_path, monkeypatch):
    from rlmagent import packages

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    first = wheel(tmp_path / "sample_resources-1.0-py3-none-any.whl")
    second = wheel(
        tmp_path / "sample_resources-2.0-py3-none-any.whl", version="2.0", text="Updated resource"
    )
    assert packages.run(["install", "sample", str(first)]) == 0
    original = packages.resource_dirs("prompts")[0]
    assert (original / "check.md").read_text() == "First resource"
    assert packages._records()["sample"]["distributions"] == [
        {"name": "sample-resources", "version": "1.0"}
    ]
    assert packages.run(["update", "sample", "--source", str(second)]) == 0
    updated = packages.resource_dirs("prompts")[0]
    assert updated != original and (updated / "check.md").read_text() == "Updated resource"
    assert (
        packages._records()["sample"]["artifact_sha256"]
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )
    assert packages.run(["remove", "sample"]) == 0
    assert packages.resource_dirs("prompts") == []
    assert (original / "check.md").exists()  # cached old version remains recoverable


async def test_share_requires_review_digest_and_never_retries_a_publication():
    from rlmagent.session_share import prepare, publish

    body, digest = prepare("session", "<html>reviewed</html>", "{}\n")
    calls = []

    def serve(request):
        calls.append(request)
        assert json.loads(request.content)["public"] is False
        return httpx.Response(201, json={"html_url": "https://gist.github.com/fixture"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        with pytest.raises(ValueError):
            await publish(body, expected_digest="wrong", token="fixture", client=client)
        assert calls == []
        assert await publish(body, expected_digest=digest, token="fixture", client=client)
        assert len(calls) == 1
        body["files"]["session.html"]["content"] = "changed after review"
        with pytest.raises(ValueError):
            await publish(body, expected_digest=digest, token="fixture", client=client)
        assert len(calls) == 1


def test_remote_clipboard_uses_tmux_osc52_without_claiming_confirmation(monkeypatch):
    from rlmagent import clipboard

    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        clipboard.subprocess, "run", lambda *a, **kw: pytest.fail("Remote copy used host clipboard")
    )
    writes = []
    status = clipboard.copy("answer\nwith unicode \u03bb", writes.append)
    assert "unconfirmed" in status and writes[0].startswith("\x1bPtmux;\x1b\x1b]52;c;")
    assert writes[0].endswith("\x1b\\")


def test_native_clipboard_confirms_only_a_successful_write(monkeypatch):
    from rlmagent import clipboard

    for key in ("TMUX", "SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(clipboard.sys, "platform", "win32")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(
        clipboard.subprocess, "run", lambda command, **kw: SimpleNamespace(returncode=0)
    )
    writes = []
    assert clipboard.copy("fixture", writes.append) == "Copied to the native clipboard."
    assert writes == []
    monkeypatch.setattr(
        clipboard.subprocess, "run", lambda command, **kw: SimpleNamespace(returncode=1)
    )
    assert "unconfirmed" in clipboard.copy("fixture", writes.append)
    assert writes


def test_managed_release_verifies_artifacts_before_activation_and_rolls_back(tmp_path, monkeypatch):
    from rlmagent import managed_release as release

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    artifact = wheel(tmp_path / "sample_resources-1.0-py3-none-any.whl")
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "rlmagent.release/1",
                "version": "0.2.0",
                "channel": "stable",
                "artifacts": [
                    {
                        "filename": artifact.name,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        release.venv,
        "EnvBuilder",
        lambda **kw: SimpleNamespace(
            create=lambda location: location.mkdir(parents=True, exist_ok=True)
        ),
    )
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    first = release.install(manifest)
    assert release._read(release.root() / "current.json") == first
    second = release.install(manifest)
    assert second["slot"] != first["slot"]
    assert release.rollback() == first
    snapshot = release._read(release.root() / "current.json")
    artifact.write_bytes(b"corrupt replacement")
    with pytest.raises(ValueError, match="checksum"):
        release.install(manifest)
    assert release._read(release.root() / "current.json") == snapshot


def test_failed_post_activation_health_restores_prior_installation(tmp_path, monkeypatch):
    from rlmagent import managed_release as release

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    previous = {"slot": "a" * 32, "version": "0.1.0", "channel": "stable"}
    candidate = {"slot": "b" * 32, "version": "0.2.0", "channel": "preview"}
    release._write(release.root() / "current.json", candidate)
    release._write(
        release.root() / "transaction.json",
        {"phase": "activating", "previous": previous, "candidate": candidate},
    )
    release._recover()
    assert release._read(release.root() / "current.json") == previous
    assert release._read(release.root() / "transaction.json")["phase"] == "recovered"


def test_busy_worker_refuses_update_without_drain(tmp_path, monkeypatch):
    from rlmagent import managed_release as release

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    monkeypatch.setattr(release, "manifest", lambda path: {"version": "0.2.0"})
    monkeypatch.setattr(release, "_fleet", lambda: [{"name": "busy", "alive": True}])
    with pytest.raises(ValueError, match="Workers are running"):
        release.install(tmp_path / "release.json")


def test_activation_failure_restores_pointer_and_releases_admission_gate(tmp_path, monkeypatch):
    from rlmagent import managed_release as release
    from rlmagent import worker

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    previous = {"slot": "a" * 32, "version": "0.1.0", "channel": "stable"}
    release._write(release.root() / "current.json", previous)
    monkeypatch.setattr(release, "manifest", lambda path: {"version": "0.2.0", "channel": "stable"})
    monkeypatch.setattr(release, "_stage", lambda *args: None)
    monkeypatch.setattr(release, "_fleet", lambda: [])
    monkeypatch.setattr(
        release.venv, "EnvBuilder", lambda **kw: SimpleNamespace(create=lambda path: None)
    )
    monkeypatch.setattr(
        release.subprocess, "run", lambda *args, **kw: SimpleNamespace(returncode=0)
    )
    checked = []

    def health(record):
        checked.append(record)
        if len(checked) == 2:
            raise ValueError("Replacement installation failed its health check")

    monkeypatch.setattr(release, "_health", health)
    with pytest.raises(ValueError, match="health check"):
        release.install(tmp_path / "release.json")
    assert release._read(release.root() / "current.json") == previous
    assert release._read(release.root() / "transaction.json")["phase"] == "recovered"
    assert not (worker.root() / "admission-paused").exists()


def test_drain_timeout_preserves_running_process_and_installation(tmp_path, monkeypatch):
    from rlmagent import managed_release as release
    from rlmagent import worker

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    worker.folder("busy").mkdir(parents=True)
    state = {"name": "busy", "alive": True}
    monkeypatch.setattr(release, "_fleet", lambda: [state])
    with pytest.raises(ValueError, match="no workers were killed"):
        release._drain([state], 0)
    assert (worker.folder("busy") / "stop").exists()
    assert state["alive"]


def test_portable_branch_compaction_roundtrip_keeps_records_and_escapes_html(tmp_path):
    from rlmagent.harness.contracts.transcript import HumanEntry
    from rlmagent.harness.session.records import (
        ForkSummaryRecord,
        PruneRecord,
        SessionMetaRecord,
        TipRecord,
        TranscriptRecord,
    )
    from rlmagent.harness.session.store import deserialize_records, serialize_record
    from rlmagent.session_html import render
    from rlmagent.session_import import import_session

    root = SessionMetaRecord(cwd=str(tmp_path))
    message = TranscriptRecord(
        parent_id=root.id, message=HumanEntry(content="line\u2028<script>x</script>")
    )
    branch = ForkSummaryRecord(
        parent_id=message.id, branch_root_id=message.id, summary="Branch <b>"
    )
    compact = PruneRecord(
        parent_id=branch.id, summary="Condensed <c>", replaces_entry_ids=[message.id]
    )
    tip = TipRecord(parent_id=compact.id, entry_id=compact.id)
    records = [root, message, branch, compact, tip]
    source = tmp_path / "portable.jsonl"
    source.write_text("".join(serialize_record(record) for record in records), encoding="utf-8")
    destination = tmp_path / "imported"
    identity = import_session(source, destination)
    restored = deserialize_records(
        (destination / f"{identity}.jsonl").read_text(encoding="utf-8").split("\n")
    )
    assert restored == records
    html = render([], records=restored)
    assert "Branch &lt;b&gt;" in html and "Condensed &lt;c&gt;" in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html and "<script>" not in html
    assert "line\u2028" in html


def test_managed_worker_restart_uses_selected_slot_and_saved_session(tmp_path, monkeypatch):
    from rlmagent import managed_release as release
    from rlmagent import worker

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "unrelated-source")
    monkeypatch.setenv("PYTHONHOME", "unrelated-interpreter")
    release._write(
        worker.folder("one") / "launch.json",
        {
            "cwd": str(tmp_path),
            "argv": ["--resume", "old", "--no-approve"],
        },
    )
    calls = []
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda command, **kw: calls.append((command, kw)) or SimpleNamespace(returncode=0),
    )
    record = {"slot": "b" * 32, "version": "0.2.0", "channel": "stable"}
    release._restart([{"name": "one", "session": "latest"}], record)
    command, options = calls[0]
    assert command[0] == str(release.python_for(record))
    assert command[-2:] == ["--resume", "latest"] and "old" not in command
    assert "PYTHONPATH" not in options["env"] and "PYTHONHOME" not in options["env"]
    assert options["cwd"] == str(tmp_path)
