"""Approval for what the sandbox asks the host to change, bound to exactly what runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rlmagent.safety import is_dangerous_command, is_safe_command

ONCE, ALWAYS, DENY = "once", "always", "deny"
AUTO = "auto"  # allow this one and stop asking for the rest of the session


@dataclass(frozen=True)
class Request:
    kind: str            # "write", "edit" or "run"
    target: str          # the path, or the command for "run"
    payload: tuple = ()  # what will be written: content, or (old, new)

    @property
    def digest(self) -> str:
        blob = json.dumps([self.kind, self.target, list(self.payload)], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def describe(self, limit: int = 2000) -> str:
        if self.kind == "run":
            return f"Run: {self.target}"
        body = "\n".join(str(p) for p in self.payload)
        if len(body) > limit:
            body = body[:limit] + f"\n... ({len(body) - limit} more characters)"
        return f"{self.kind.capitalize()} {self.target}\n{body}"


class Refused(Exception):
    pass


Ask = Callable[[Request], Awaitable[str]]


class Approver:
    """`ask`, `auto` or `deny`; `ask` needs a function that shows the request to the user."""

    def __init__(self, policy: str = "deny", ask: Ask | None = None) -> None:
        self.policy = policy
        self.ask = ask
        self._always: set[tuple[str, str]] = set()

    async def approve(self, request: Request) -> str:
        """The digest that may run, or Refused."""
        if request.kind == "run" and is_dangerous_command(request.target):
            raise Refused(f"refused: {request.target!r} is a destructive command")
        if request.kind == "run" and is_safe_command(request.target):
            return request.digest
        if self.policy == "auto" or (request.kind, request.target) in self._always:
            return request.digest
        if self.policy != "ask" or self.ask is None:
            raise Refused(f"refused: {request.kind} {request.target} needs approval "
                          "(run with --auto-approve, or interactively)")
        answer = await self.ask(request)
        if answer == DENY:
            raise Refused(f"the user refused: {request.kind} {request.target}")
        if answer == ALWAYS:
            self._always.add((request.kind, request.target))
        if answer == AUTO:
            self.policy = "auto"
        return request.digest

    def toggle_auto(self) -> bool:
        """Switch between asking and auto-accepting; True when auto-accept is now on."""
        if self.policy == "ask":
            self.policy = "auto"
        elif self.policy == "auto" and self.ask is not None:
            self.policy = "ask"
        return self.policy == "auto"


def check(approved: str, request: Request) -> None:
    """Refuse when what is about to run is not what was approved."""
    if approved != request.digest:
        raise Refused(f"refused: {request.kind} {request.target} changed after approval")
