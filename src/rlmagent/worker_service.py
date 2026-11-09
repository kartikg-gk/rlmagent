"""Register the fleet watcher with the current user's service manager."""

import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from rlmagent.discovery import default_paths


def _command(*args: str, check=True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{args[0]} timed out") from exc
    if check and result.returncode:
        raise ValueError((result.stderr or result.stdout).strip() or f"{args[0]} failed")
    return result


def _print_result(result: subprocess.CompletedProcess) -> int:
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def _task_xml(name: str, sid: str, argv: list[str]) -> bytes:
    task = Element("Task", {"version": "1.2", "xmlns": "http://schemas.microsoft.com/windows/2004/02/mit/task"})
    registration = SubElement(task, "RegistrationInfo")
    SubElement(registration, "Description").text = "Recover saved RLM Agent workers when they exit."
    SubElement(registration, "URI").text = f"\\{name}"
    trigger = SubElement(SubElement(task, "Triggers"), "LogonTrigger")
    SubElement(trigger, "UserId").text = sid
    principal = SubElement(SubElement(task, "Principals"), "Principal", id="Watcher")
    SubElement(principal, "UserId").text = sid
    SubElement(principal, "LogonType").text = "InteractiveToken"
    SubElement(principal, "RunLevel").text = "LeastPrivilege"
    settings = SubElement(task, "Settings")
    for key, value in (
        ("MultipleInstancesPolicy", "IgnoreNew"), ("DisallowStartIfOnBatteries", "false"),
        ("StopIfGoingOnBatteries", "false"), ("ExecutionTimeLimit", "PT0S"),
    ):
        SubElement(settings, key).text = value
    retry = SubElement(settings, "RestartOnFailure")
    SubElement(retry, "Interval").text = "PT1M"
    SubElement(retry, "Count").text = "10"
    action = SubElement(SubElement(task, "Actions", Context="Watcher"), "Exec")
    SubElement(action, "Command").text = argv[0]
    SubElement(action, "Arguments").text = subprocess.list2cmdline(argv[1:])
    return tostring(task, encoding="utf-16", xml_declaration=True)


def _windows(action: str, name: str, argv: list[str], home: Path) -> int:
    name = f"RlmAgentFleet-{name}"
    if action == "install":
        identity = _command("whoami.exe", "/user", "/fo", "csv", "/nh")
        rows = list(csv.reader(identity.stdout.splitlines()))
        sid = rows[0][-1] if rows and rows[0] else ""
        if not re.fullmatch(r"S-1(?:-\d+)+", sid):
            raise ValueError("Could not identify the current Windows user")
        home.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=home) as directory:
            definition = Path(directory) / "fleet.xml"
            definition.write_bytes(_task_xml(name, sid, argv))
            _command("schtasks.exe", "/Create", "/TN", name, "/XML", str(definition), "/F")
        action = "start"
    if action == "uninstall":
        _command("schtasks.exe", "/End", "/TN", name, check=False)
    native = {"start": "/Run", "stop": "/End", "status": "/Query", "uninstall": "/Delete"}
    extra = {"status": ["/FO", "LIST", "/V"], "uninstall": ["/F"]}.get(action, [])
    return _print_result(_command("schtasks.exe", native[action], "/TN", name, *extra, check=False))


def _systemd_unit(argv: list[str]) -> str:
    # systemd expands percent specifiers and dollar variables even in quoted arguments.
    if any(any(ord(char) < 32 for char in arg) for arg in argv):
        raise ValueError("Service paths may not contain control characters")
    quoted = [
        '"' + arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'
        for arg in argv
    ]
    return (
        "[Unit]\nDescription=RLM Agent worker fleet watcher\n\n"
        "[Service]\nType=exec\nExecStart=" + " ".join(quoted) + "\n"
        "Restart=on-failure\nRestartSec=5s\nKillMode=process\n\n"
        "[Install]\nWantedBy=default.target\n"
    )


def _linux(action: str, name: str, argv: list[str]) -> int:
    name = f"rlm-agent-fleet-{name}.service"
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    unit = config / "systemd" / "user" / name
    manager = ("systemctl", "--user")
    if action == "install":
        definition = _systemd_unit(argv)
        unit.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=unit.parent) as directory:
            staged = Path(directory) / unit.name
            staged.write_text(definition, encoding="utf-8")
            staged.replace(unit)
        _command(*manager, "daemon-reload")
        _command(*manager, "enable", name)
        action = "restart"
    if action == "uninstall":
        _command(*manager, "disable", "--now", name)
        unit.unlink(missing_ok=True)
        return _print_result(_command(*manager, "daemon-reload", check=False))
    return _print_result(_command(*manager, action, "--no-pager", name, check=False))


def manage(action: str) -> int:
    paths = default_paths()
    home = paths.home.resolve()
    name = hashlib.sha256(os.path.normcase(str(home)).encode()).hexdigest()[:16]
    argv = [sys.executable, "-m", "rlmagent.worker_service", str(home), str(paths.sessions.resolve())]
    if sys.platform == "win32":
        windowless = Path(sys.executable).with_name("pythonw.exe")
        if windowless.is_file():
            argv[0] = str(windowless)
        return _windows(action, name, argv, home)
    if sys.platform == "linux":
        return _linux(action, name, argv)
    raise ValueError("Worker service management requires Windows Task Scheduler or Linux systemd")


def _run(home: str, sessions: str) -> int:
    from rlmagent.worker import _watch_fleet

    os.environ["RLM_AGENT_HOME"] = home
    os.environ["RLM_AGENT_SESSIONS_DIR"] = sessions
    if sys.platform != "win32":
        return _watch_fleet()
    executable = Path(sys.executable)
    console = executable.with_name("python.exe")
    if executable.name.lower() == "pythonw.exe" and console.is_file():
        # Workers keep using the console interpreter with their existing detached flags.
        sys.executable = str(console)
    with (Path(home) / "fleet-service.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            return _watch_fleet()


if __name__ == "__main__":
    raise SystemExit(_run(*sys.argv[1:]))
