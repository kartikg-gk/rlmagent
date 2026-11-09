"""Run policy: limits and guards set for a session, from the command line or a contract."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEFAULT_OUTPUT_BYTES = 20_000


@dataclass
class Policy:
    allow_git: bool = True
    max_tool_output_bytes: int = DEFAULT_OUTPUT_BYTES  # zero disables this output cap
    max_compactions: int | None = None
    agent_max_tokens: int | None = None  # completion tokens one agent may spend
    search_block: list[str] = field(default_factory=list)
    shell_access: str = "workspace"
    shell_network: bool = False
    allowed_tools: frozenset[str] | None = None

    def allows_tool(self, name: str) -> bool:
        return self.allowed_tools is None or name in self.allowed_tools


# Git commands that only look at or change the current branch's working state.
_GIT_LOCAL = {
    "status", "diff", "add", "commit", "restore", "rm", "mv", "show", "log", "blame",
    "stash", "apply", "grep", "ls-files", "rev-parse", "describe", "shortlog", "mergetool",
}
# Subcommands that touch other refs, other branches, remotes or history.
_GIT_REACHING = {
    "push", "pull", "fetch", "clone", "remote", "checkout", "switch", "branch", "tag",
    "merge", "rebase", "reset", "cherry-pick", "revert", "worktree", "reflog", "gc",
    "filter-branch", "update-ref", "symbolic-ref", "for-each-ref", "notes", "submodule",
    "bisect", "replace", "prune", "am", "format-patch", "send-email", "config",
}


def _git_calls(command: str):
    """The git subcommands in a shell line, split on ; && || | and newlines."""
    for piece in re.split(r"&&|\|\||[;|\n]", command):
        try:
            words = shlex.split(piece, posix=True)
        except ValueError:
            words = piece.split()
        while words and "=" in words[0] and not words[0].startswith("-"):
            words = words[1:]  # leading VAR=value assignments
        if not words or words[0].rsplit("/", 1)[-1] not in ("git", "git.exe"):
            continue
        rest = words[1:]
        while rest and rest[0].startswith("-"):
            # -C dir and -c key=value take a value; other options stand alone.
            rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
        yield (rest[0] if rest else ""), rest[1:]


def git_refusal(command: str, policy: Policy) -> str | None:
    """Why a shell command may not run under this policy, or None."""
    if policy.allow_git:
        return None
    for sub, args in _git_calls(command):
        if sub in _GIT_REACHING:
            return (f"git {sub} touches other branches, remotes or history, which this run does not allow; "
                    "run; work with status, diff, add, commit and log")
        if sub == "log" and any(a in ("--all", "--branches", "--remotes", "--tags") for a in args):
            return "git log over other refs is turned off for this run"
        if sub not in _GIT_LOCAL and sub:
            return f"git {sub} is not on this run's list of allowed git commands"
    return None


def cap_output(text: str, limit: int) -> str:
    """Keep the start and end of long output, saying how much was left out."""
    data = text.encode("utf-8", errors="replace")
    if limit <= 0 or len(data) <= limit:
        return text
    head = data[: limit * 2 // 3].decode("utf-8", errors="ignore")
    tail = data[-(limit // 3):].decode("utf-8", errors="ignore")
    cut = len(data) - len(head.encode()) - len(tail.encode())
    return (f"{head}\n... [{cut} bytes of output left out; print less, or slice what you "
            f"need] ...\n{tail}")


def blocked_url(url: str, policy: Policy) -> bool:
    """Whether a URL is on the run's block list (a domain, *.domain, or an origin)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    origin = f"{parsed.scheme}://{parsed.netloc}".lower()
    for rule in policy.search_block:
        rule = rule.lower().rstrip("/")
        if "://" in rule:
            if origin == rule:
                return True
        elif rule.startswith("*."):
            if host.endswith(rule[1:]):
                return True
        elif host == rule or host.endswith("." + rule):
            return True
    return False


__all__ = ["DEFAULT_OUTPUT_BYTES", "Policy", "blocked_url", "cap_output", "git_refusal"]
