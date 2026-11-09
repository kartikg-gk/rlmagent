from pathlib import Path
from subprocess import CompletedProcess
from xml.etree.ElementTree import fromstring

from rlmagent import worker, worker_service


def test_windows_task_uses_the_current_user_and_exact_argv(tmp_path, monkeypatch):
    commands = []
    home = tmp_path / "home with spaces"
    sid = "S-1-5-21-42"

    def command(*args, check=True):
        commands.append(args)
        if args[0] == "whoami.exe":
            return CompletedProcess(args, 0, f'"host\\user","{sid}"\n', "")
        if args[1] == "/Create":
            task = fromstring(Path(args[args.index("/XML") + 1]).read_bytes())
            ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
            assert task.findtext("t:Triggers/t:LogonTrigger/t:UserId", namespaces=ns) == sid
            principal = task.find("t:Principals/t:Principal", ns)
            assert principal.findtext("t:LogonType", namespaces=ns) == "InteractiveToken"
            assert principal.findtext("t:RunLevel", namespaces=ns) == "LeastPrivilege"
            assert task.findtext("t:Actions/t:Exec/t:Command", namespaces=ns) == "C:/My Env/python.exe"
            assert f'"{home}"' in task.findtext("t:Actions/t:Exec/t:Arguments", namespaces=ns)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(worker_service, "_command", command)
    argv = ["C:/My Env/python.exe", "-m", "rlmagent.worker_service", str(home), str(home / "sessions")]
    assert worker_service._windows("install", "test", argv, home) == 0
    assert worker_service._windows("status", "test", argv, home) == 0
    assert worker_service._windows("uninstall", "test", argv, home) == 0
    assert [args[1] for args in commands[1:]] == ["/Create", "/Run", "/Query", "/End", "/Delete"]
    assert all(args[3] == "RlmAgentFleet-test" for args in commands[1:])


def test_systemd_unit_lifecycle_uses_the_user_manager(tmp_path, monkeypatch):
    commands = []

    def command(*args, check=True):
        commands.append(args)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(worker_service, "_command", command)
    argv = ["/tmp/My Env/python", "-m", "rlmagent.worker_service", "/tmp/%u/$HOME", "/tmp/sessions"]
    assert worker_service._linux("install", "test", argv) == 0
    unit = tmp_path / "systemd/user/rlm-agent-fleet-test.service"
    definition = unit.read_text(encoding="utf-8")
    assert 'ExecStart="/tmp/My Env/python"' in definition
    assert '"/tmp/%%u/$$HOME"' in definition
    assert "KillMode=process" in definition
    assert worker_service._linux("uninstall", "test", argv) == 0
    assert not unit.exists()
    assert [args[:3] for args in commands] == [
        ("systemctl", "--user", operation)
        for operation in ("daemon-reload", "enable", "restart", "disable", "daemon-reload")
    ]


def test_worker_service_dispatch_keeps_the_manager_exit_status(monkeypatch):
    def status(action):
        assert action == "status"
        return 3

    monkeypatch.setattr(worker_service, "manage", status)
    assert worker.handle(["service", "status"]) == 3


def test_launchd_agent_lifecycle_uses_the_gui_domain(tmp_path, monkeypatch):
    import plistlib
    from pathlib import Path

    commands = []

    def command(*args, check=True):
        commands.append(args)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(worker_service, "_command", command)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(worker_service.os, "getuid", lambda: 501, raising=False)
    argv = ["/Users/me/My Env/python", "-m", "rlmagent.worker_service", "/h", "/s"]
    home = tmp_path / "home"
    assert worker_service._macos("install", "test", argv, home) == 0
    plist = tmp_path / "Library/LaunchAgents/com.rlmagent.fleet.test.plist"
    definition = plistlib.loads(plist.read_bytes())
    assert definition["ProgramArguments"] == argv
    assert definition["KeepAlive"] == {"SuccessfulExit": False}
    assert definition["StandardErrorPath"] == str(home / "fleet-service.log")
    assert worker_service._macos("status", "test", argv, home) == 0
    assert worker_service._macos("uninstall", "test", argv, home) == 0
    assert not plist.exists()
    target = "gui/501/com.rlmagent.fleet.test"
    assert commands == [
        ("launchctl", "bootout", target),
        ("launchctl", "bootstrap", "gui/501", str(plist)),
        ("launchctl", "kickstart", "-k", target),
        ("launchctl", "print", target),
        ("launchctl", "bootout", target),
    ]
