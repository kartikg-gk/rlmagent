"""A record of every model call in an agent tree, and how the calls relate."""

from __future__ import annotations

import json
import math
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

from rlmagent.agents.budget import REFUSED
from rlmagent.harness.provider.wire import StreamCloseEvent, StreamFaultEvent

request_id: ContextVar[str | None] = ContextVar("model_request_id", default=None)
attempt_observer: ContextVar[object] = ContextVar("model_attempt_observer", default=None)


def model_attempt(**fields):
    observer = attempt_observer.get()
    if observer is not None:
        observer(**fields)


def correlate_headers(headers: dict[str, str]) -> dict[str, str]:
    call = request_id.get()
    if call is None:
        return headers
    result = {key: value for key, value in headers.items()
              if key.lower() != "x-rlmagent-request-id"}
    result["X-Rlmagent-Request-Id"] = call
    return result


async def correlated(stream, call: str | None, tracer=None):
    """Give each provider read the same ID without leaking it to other streams."""
    iterator = aiter(stream)
    try:
        while True:
            token = request_id.set(call)
            observer = attempt_observer.set(
                (lambda **fields: tracer._write("model_attempt", request=call, **fields))
                if tracer is not None else None)
            try:
                event = await anext(iterator)
            except StopAsyncIteration:
                return
            finally:
                request_id.reset(token)
                attempt_observer.reset(observer)
            yield event
    finally:
        if close := getattr(iterator, "aclose", None):
            await close()


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
        self._historical_calls: dict[str, dict] = {}
        self._child_calls: dict[str, str] = {}
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

    def persist(self, path: Path) -> None:
        """Append this runtime's request ledger to the saved session."""
        if self.path is not None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Saved request ledger contains an invalid record")
                    if record.get("kind") == "call":
                        self._remember_call(self._historical_calls, record)
        with path.open("a", encoding="utf-8") as output:
            for record in self.records:
                output.write(json.dumps(record, default=str) + "\n")
        self.path = path

    @staticmethod
    def _remember_call(calls, record):
        identity = record.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Saved request ledger has a call without an identity")
        if identity in calls and calls[identity] != record:
            raise ValueError("Saved request ledger has conflicting records for one call")
        calls[identity] = record

    def lifetime_usage(self):
        """Idempotent persisted-plus-current usage, separate from admission counters."""
        calls = dict(self._historical_calls)
        for record in self.records:
            if record.get("kind") == "call":
                self._remember_call(calls, record)
        fields = ("input", "output", "cache_read", "cache_write", "reasoning", "cost")
        agents, incomplete, unreported, unpriced = {}, 0, 0, 0
        for record in calls.values():
            agent = record.get("agent")
            if not isinstance(agent, str):
                raise ValueError("Saved request ledger has an invalid agent")
            totals = agents.setdefault(agent, dict.fromkeys(("calls", *fields), 0))
            totals["calls"] += 1
            incomplete += any(field not in record for field in fields)
            unreported += not record.get("usage_reported", False)
            unpriced += record.get("cost", 0) == 0
            for field in fields:
                value = record.get(field, 0)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Saved request ledger has invalid usage")
                totals[field] += value
        return {
            "agents": agents,
            "totals": {field: sum(agent[field] for agent in agents.values())
                       for field in ("calls", *fields)},
            "coverage": {"incompleteUsageRecords": incomplete,
                         "unreportedUsageRecords": unreported, "unpricedRecords": unpriced},
        }

    def child_usage(self, child: str, *, compatibility: bool = False) -> dict:
        fields = ("input", "output", "cache_read", "cache_write", "reasoning", "cost")
        calls = {} if compatibility else dict(self._historical_calls)
        for record in self.records:
            if record.get("kind") == "call":
                self._remember_call(calls, record)
        selected = [record for record in calls.values()
                    if record.get("agent" if compatibility else "child") == child]
        return {"calls": len(selected),
                "unreported": sum(not record.get("usage_reported") for record in selected),
                "unpriced": sum(record.get("cost", 0) == 0 for record in selected),
                **{field: sum(record.get(field, 0) for record in selected) for field in fields}}

    def begin(self, agent: str, model: str, *, child: str | None = None) -> str:
        call = uuid.uuid4().hex
        if child is not None:
            self._child_calls[call] = child
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
        self._child_calls.pop(call, None)
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
        child = self._child_calls.pop(call, None)
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
        for name in ("cache_read", "cache_write", "reasoning"):
            t[name] = t.get(name, 0) + ((getattr(usage, name, 0) or 0) if usage is not None else 0)
        self._write("call", id=call, agent=agent, model=model, stop=stop, cost=cost,
                    input=tokens_in, output=tokens_out, usage_reported=usage is not None,
                    cache_read=(usage.cache_read or 0) if usage is not None else 0,
                    cache_write=(usage.cache_write or 0) if usage is not None else 0,
                    reasoning=(usage.reasoning or 0) if usage is not None else 0,
                    seconds=round(time.time() - started, 3),
                    **({"child": child} if child is not None else {}))

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
        limit = getattr(getattr(self._tree, "policy", None), "agent_max_tokens", None)
        if limit is not None and self._tree.tracer._totals.get(self._agent, {}).get("output", 0) >= limit:
            from rlmagent.harness.contracts.transcript import ModelEntry

            yield StreamFaultEvent(reason="error", error=ModelEntry(model=model, stop_reason="error",
                error_message=f"agent {self._agent} reached its limit of {limit} completion tokens"))
            return
        refinement = getattr(self._tree, "refinement", None)
        if refinement is not None:
            kwargs["system"] = refinement.filter_system(kwargs["system"])
        tracer = self._tree.tracer
        call = tracer.begin(self._agent, model)
        session = getattr(self._tree, "session", None)
        if session is None:
            session = getattr(self._tree, "_session_of", {}).get(self._agent)
        ledger = getattr(session, "steering", None)
        pending = []
        if ledger is not None:
            from rlmagent.harness.contracts.transcript import HumanEntry

            text, pending = ledger.context(self._agent, root=self._agent == self._tree.root_id())
            if text:
                kwargs["messages"] = [*kwargs["messages"], HumanEntry(content="User steering:\n" + text)]
        usage, stop, refused = None, None, False
        from rlmagent.model.deadline import bounded_stream
        try:
            async for event in correlated(
                bounded_stream(self._tree.provider.stream_response(model=model, **kwargs),
                               model, getattr(self._tree, "request_timeout", 300.0)), call, tracer
            ):
                if isinstance(event, StreamCloseEvent):
                    usage, stop = event.message.usage, event.reason
                    if ledger is not None:
                        ledger.consumed(pending, self._agent, call)
                        if pending:
                            tracer._write("steering", agent=self._agent, request=call, messages=pending)
                elif isinstance(event, StreamFaultEvent):
                    usage, stop = event.error.usage, event.reason
                    refused = (event.error.error_message or "").startswith(REFUSED)
                yield event
        finally:
            if refused:
                tracer.drop(call)
            else:
                tracer.end(call, usage, stop)
