"""Native clipboard writes with explicit terminal-delivery fallback."""

import base64
import os
import shutil
import subprocess
import sys


def sequence(text: str, *, tmux=False) -> str:
    encoded = base64.b64encode(text.encode()).decode()
    value = f"\x1b]52;c;{encoded}\x07"
    return "\x1bPtmux;" + value.replace("\x1b", "\x1b\x1b") + "\x1b\\" if tmux else value


def copy(text: str, terminal_write) -> str:
    if len(text.encode()) > 1024 * 1024:
        raise ValueError("Clipboard content exceeds the 1 MiB limit")
    remote = bool(
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY") or os.environ.get("TMUX")
    )
    command = None
    if not remote:
        if sys.platform == "win32":
            shell = shutil.which("powershell.exe")
            if shell:
                command = [
                    shell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "[Console]::InputEncoding = [Text.UTF8Encoding]::new(); "
                    "Set-Clipboard -Value ([Console]::In.ReadToEnd())",
                ]
        else:
            for candidate in (
                ("pbcopy",) if sys.platform == "darwin" else ("wl-copy", "xclip", "xsel")
            ):
                executable = shutil.which(candidate)
                if executable:
                    command = [executable]
                    if candidate == "xclip":
                        command += ["-selection", "clipboard"]
                    elif candidate == "xsel":
                        command += ["--clipboard", "--input"]
                    break
    if command is not None:
        try:
            result = subprocess.run(
                command,
                input=text.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            if result.returncode == 0:
                return "Copied to the native clipboard."
        except (OSError, subprocess.TimeoutExpired):
            pass
    terminal_write(sequence(text, tmux=bool(os.environ.get("TMUX"))))
    return "Clipboard delivery requested from the terminal; delivery is unconfirmed."
