"""Coding agents as models: Claude Code and Codex through their own CLIs, others through ACP."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from rlmagent.harness.contracts.tooling import CancelToken, ToolSpec
from rlmagent.harness.contracts.transcript import (
    CallBlock,
    CostBreakdown,
    ModelEntry,
    TextSegment,
    TranscriptEntry,
    UsageStats,
)
from rlmagent.harness.provider.wire import (
    CallCloseEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    WireEvent,
)
from rlmagent.model.external import acp
from rlmagent.model.external.acp import Cancelled as _Cancelled
from rlmagent.model.external.textcalls import PROTOCOL, parse_reply, render

PREFIXES = ("cli:", "acp:")


def is_external(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIXES)


def parse_model(model: str) -> tuple[str, str, str | None]:
    """`cli:codex?model=gpt-5.5` -> ("cli", "codex", "gpt-5.5")."""
    route, _, rest = model.partition(":")
    name, _, query = rest.partition("?")
    sub = dict(p.partition("=")[::2] for p in query.split("&") if p).get("model") or None
    known = PRESETS if route == "cli" else acp.agents()
    if name not in known:
        raise ValueError(f"unknown agent {name!r} in {model!r}; known: {', '.join(known)}")
    return route, name, sub


def kill_tree(proc) -> None:
    """Stop the process and anything it started (npx starts node, for one)."""
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def _exe(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise ValueError(f"{name} not found on PATH")
    return found


def _claude(sub: str | None, system: str, prompt: str, cwd: Path) -> tuple[list[str], str]:
    (cwd / "system.md").write_text(system, encoding="utf-8")
    argv = [_exe("claude"), "-p", "--output-format", "json",
            "--system-prompt-file", str(cwd / "system.md"),
            "--tools", "", "--strict-mcp-config", "--setting-sources", "project",
            "--no-session-persistence"]
    return argv + (["--model", sub] if sub else []), prompt


_CODEX_TOOLS = ("shell_tool", "unified_exec", "code_mode_host", "view_image", "browser_use",
                "computer_use", "apps", "tool_suggest", "sleep_tool", "multi_agent", "goals",
                "image_generation")


def _codex(sub: str | None, system: str, prompt: str, cwd: Path) -> tuple[list[str], str]:
    argv = [_exe("codex"), "exec", "--json", "--sandbox", "read-only",
            "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "-C", str(cwd),
            "-c", "web_search=disabled"]
    for feature in _CODEX_TOOLS:
        argv += ["--disable", feature]
    return argv + (["-m", sub] if sub else []) + ["-"], f"{system}\n\n{prompt}"


def _usage(**kw) -> UsageStats:
    total = sum(kw.get(k, 0) for k in ("input", "output", "cache_read", "cache_write"))
    return UsageStats(total_tokens=total, **kw)


def parse_claude(out: str) -> tuple[str, UsageStats]:
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        raise ValueError("claude printed nothing")
    data = json.loads(lines[-1])
    if data.get("is_error") or data.get("subtype") != "success":
        raise ValueError(f"claude failed: {data.get('result') or data.get('subtype')}")
    u = data.get("usage", {})
    usage = _usage(input=u.get("input_tokens", 0), output=u.get("output_tokens", 0),
                   cache_read=u.get("cache_read_input_tokens", 0),
                   cache_write=u.get("cache_creation_input_tokens", 0),
                   cost=CostBreakdown(total=float(data.get("total_cost_usd") or 0.0)))
    return str(data.get("result", "")), usage


def parse_codex(out: str) -> tuple[str, UsageStats]:
    texts, usage = [], UsageStats()
    for line in out.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "item.completed" and event.get("item", {}).get("type") == "agent_message":
            texts.append(event["item"].get("text", ""))
        elif kind == "turn.completed":
            u = event.get("usage", {})
            cached = u.get("cached_input_tokens", 0)
            usage = _usage(input=u.get("input_tokens", 0) - cached, cache_read=cached,
                           output=u.get("output_tokens", 0))
        elif kind in ("error", "turn.failed"):
            message = event.get("message") or event.get("error", {}).get("message", "")
            raise ValueError(f"codex failed: {message}")
    if not texts:
        raise ValueError("codex gave no answer")
    return "\n\n".join(texts), usage


PRESETS = {"claude-code": (_claude, parse_claude), "codex": (_codex, parse_codex)}


async def _run(argv: list[str], stdin: str, cwd: Path, signal: CancelToken | None) -> str:
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    task = asyncio.ensure_future(proc.communicate(stdin.encode("utf-8")))
    try:
        while not task.done():
            if signal is not None and signal.is_cancelled():
                raise _Cancelled
            await asyncio.wait({task}, timeout=0.2)
    finally:
        if not task.done():
            kill_tree(proc)
            task.cancel()
            await proc.wait()
    out, err = task.result()
    if proc.returncode != 0 and not out.strip():
        raise ValueError(f"{Path(argv[0]).stem} exited {proc.returncode}: {err.decode(errors='replace')[-500:]}")
    return out.decode("utf-8", errors="replace")


def _fault(model: str, reason: str, message: str) -> StreamFaultEvent:
    entry = ModelEntry(model=model, provider="cli", api="cli", stop_reason=reason,
                       error_message=message)
    return StreamFaultEvent(reason=reason, error=entry)


class ExternalAgentProvider:
    """Answers `cli:` models with an agent CLI; anything else goes to `fallback`."""

    def __init__(self, fallback=None) -> None:
        self.fallback = fallback

    def __getattr__(self, name: str):
        if self.fallback is None:
            raise AttributeError(name)
        return getattr(self.fallback, name)

    def stream_response(self, *, model: str, system: str, messages: Sequence[TranscriptEntry],
                        tools: Sequence[ToolSpec], signal: CancelToken | None = None,
                        cache_key: str | None = None) -> AsyncIterator[WireEvent]:
        if not is_external(model):
            if self.fallback is None:
                raise ValueError(f"the cli provider needs a cli: or acp: model, got {model!r}")
            return self.fallback.stream_response(model=model, system=system, messages=messages,
                                                 tools=tools, signal=signal, cache_key=cache_key)
        return self._answer(model, system, messages, tools, signal)

    async def _answer(self, model, system, messages, tools, signal) -> AsyncIterator[WireEvent]:
        coding = any(t.name == "python" for t in tools)
        try:
            route, name, sub = parse_model(model)
            system = f"{system}\n\n{PROTOCOL}" if coding else system
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                if route == "acp":
                    text, usage = await acp.answer(name, sub, f"{system}\n\n{render(messages)}",
                                                   Path(tmp), signal)
                else:
                    build, parse = PRESETS[name]
                    argv, stdin = build(sub, system, render(messages), Path(tmp))
                    text, usage = parse(await _run(argv, stdin, Path(tmp), signal))
        except _Cancelled:
            yield _fault(model, "aborted", "cancelled")
            return
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            yield _fault(model, "error", str(exc))
            return
        said, code = parse_reply(text) if coding else (text.strip(), None)
        content: list = [TextSegment(text=said)] if said else []
        stop = "stop"
        if code is not None:
            call = CallBlock(name="python", id=f"call_{uuid.uuid4().hex[:12]}", arguments={"code": code})
            content.append(call)
            stop = "toolUse"
        message = ModelEntry(content=content, model=model, provider="cli", api="cli",
                             stop_reason=stop, usage=usage)
        if code is not None:
            yield CallCloseEvent(content_index=len(content) - 1, tool_call=call, partial=message)
        yield StreamCloseEvent(reason=stop, message=message)
