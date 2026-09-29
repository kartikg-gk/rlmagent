"""A record of every model call in an agent tree, and how the calls relate."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from rlmagent_app.agents.budget import REFUSED
from rlmagent_harness.provider.wire import StreamCloseEvent, StreamFaultEvent


class Tracer:
    def __init__(self, path: Path | str | None = None, contract: dict | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        self._last: dict[str, str] = {}
        self._pending: dict[str, list[tuple[str, str]]] = {}
        self._compacted: set[str] = set()
        self._returned: set[str] = set()
        self._totals: dict[str, dict] = {}
        self._open: dict[str, tuple] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        self._write("run", contract=contract, started=time.time())

    def _write(self, kind: str, **fields) -> None:
        record = {"kind": kind, **fields}
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def begin(self, agent: str, model: str) -> str:
        call = uuid.uuid4().hex
        previous = self._last.get(agent)
        compacted = agent in self._compacted
        pending = self._pending.pop(agent, [])
        links = [(source, how) for source, how in pending]
        if previous is not None:
            links.insert(0, (previous, "compact" if compacted else "next"))
        self._compacted.discard(agent)
        self._last[agent] = call
        self._open[call] = (agent, model, time.time(), previous, pending, compacted, links)
        return call

    def drop(self, call: str) -> None:
        """Forget a call that was never sent (refused by the budget)."""
        agent, _, _, previous, pending, compacted, _ = self._open.pop(call)
        if previous is None:
            self._last.pop(agent, None)
        else:
            self._last[agent] = previous
        if pending:
            self._pending.setdefault(agent, [])[:0] = pending
        if compacted:
            self._compacted.add(agent)

    def end(self, call: str, usage, stop: str | None) -> None:
        agent, model, started, _, _, _, links = self._open.pop(call)
        for source, how in links:
            self._write("link", **{"from": source, "to": call, "how": how})
        self._totals.setdefault(agent, {"calls": 0, "cost": 0.0, "input": 0, "output": 0})
        cost = usage.cost.total if usage is not None else 0.0
        tokens_in = usage.input if usage is not None else 0
        tokens_out = usage.output if usage is not None else 0
        t = self._totals[agent]
        t["calls"] += 1
        t["cost"] += cost
        t["input"] += tokens_in
        t["output"] += tokens_out
        self._write("call", id=call, agent=agent, model=model, stop=stop, cost=cost,
                    input=tokens_in, output=tokens_out, seconds=round(time.time() - started, 3))

    def spawned(self, parent: str, child: str) -> None:
        if parent in self._last:
            self._pending.setdefault(child, []).append((self._last[parent], "spawn"))

    def returned(self, parent: str, child: str) -> None:
        last = self._last.get(child)
        if last is not None and last not in self._returned:
            self._returned.add(last)
            self._pending.setdefault(parent, []).append((last, "return"))

    def compacted(self, agent: str) -> None:
        self._compacted.add(agent)

    def finish(self, agents: list[dict], stopped_by: str | None, complete: bool = True) -> dict:
        for agent in agents:
            self._write("agent", **agent)
        summary = {
            "calls": sum(t["calls"] for t in self._totals.values()),
            "cost": sum(t["cost"] for t in self._totals.values()),
            "input": sum(t["input"] for t in self._totals.values()),
            "output": sum(t["output"] for t in self._totals.values()),
            "agents": self._totals,
            "agents_started": max(len(agents) - 1, 0),
            "stopped_by": stopped_by,
            "complete": complete,
        }
        self._write("summary", **summary)
        return summary


class TracedProvider:
    """One agent's view of the tree's provider: every call is traced as that agent's."""

    follows_tree = True

    def __init__(self, tree, agent: str) -> None:
        self._tree = tree
        self._agent = agent

    def __getattr__(self, name):
        return getattr(self._tree.provider, name)

    async def stream_response(self, *, model, **kwargs):
        tracer = self._tree.tracer
        call = tracer.begin(self._agent, model)
        usage, stop, refused = None, None, False
        try:
            async for event in self._tree.provider.stream_response(model=model, **kwargs):
                if isinstance(event, StreamCloseEvent):
                    usage, stop = event.message.usage, event.reason
                elif isinstance(event, StreamFaultEvent):
                    usage, stop = event.error.usage, event.reason
                    refused = (event.error.error_message or "").startswith(REFUSED)
                yield event
        finally:
            if refused:
                tracer.drop(call)
            else:
                tracer.end(call, usage, stop)
