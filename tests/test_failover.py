"""Retry budget and provider failover."""

from __future__ import annotations

import pytest

from rlmagent.harness.contracts.transcript import ModelEntry, TextSegment
from rlmagent.harness.provider.wire import (
    ContentChunkEvent,
    StreamCloseEvent,
    StreamFaultEvent,
    StreamOpenEvent,
)
from rlmagent.model.failover import FailoverProvider, parse_fallbacks, with_fallbacks
from rlmagent.model.transport.backoff import RetryBudget


def _entry(text="", model="m"):
    return ModelEntry(model=model, content=[TextSegment(text=text)] if text else [],
                      stop_reason="stop")


class Scripted:
    def __init__(self, *events):
        self.events, self.models = events, []

    async def stream_response(self, *, model, **_):
        self.models.append(model)
        for event in self.events:
            yield event


def _fault():
    entry = _entry()
    return StreamFaultEvent(reason="error", error=entry)


def _ok(text):
    entry = _entry(text)
    return [StreamOpenEvent(partial=entry),
            ContentChunkEvent(content_index=0, delta=text, partial=entry),
            StreamCloseEvent(reason="stop", message=entry)]


async def _run(provider, model="main"):
    return [e async for e in provider.stream_response(
        model=model, system="", messages=[], tools=[])]


def test_retry_budget_refills_after_the_window():
    now = [0.0]
    budget = RetryBudget(limit=2, window_seconds=10, clock=lambda: now[0])
    assert budget.take() and budget.take()
    assert not budget.take()
    now[0] = 10.0
    assert budget.take()


def test_parse_fallbacks_skips_bad_items():
    assert parse_fallbacks("a:m1, bad ,b:org/m2,:x") == [("a", "m1"), ("b", "org/m2")]


def test_no_fallbacks_means_no_wrapper():
    primary = Scripted()
    assert with_fallbacks(primary, lambda n: None, env={}) is primary


@pytest.mark.asyncio
async def test_outright_failure_moves_to_the_next_pair():
    primary = Scripted(StreamOpenEvent(partial=_entry()), _fault())
    backup = Scripted(*_ok("hello"))
    provider = FailoverProvider(primary, [("b", "backup-model")], lambda n: backup)

    events = await _run(provider)

    assert backup.models == ["backup-model"]
    assert not any(isinstance(e, StreamFaultEvent) for e in events)
    assert provider.last_route == ("b", "backup-model")


@pytest.mark.asyncio
async def test_failure_after_output_is_not_replayed_elsewhere():
    entry = _entry("par")
    primary = Scripted(StreamOpenEvent(partial=entry),
                       ContentChunkEvent(content_index=0, delta="par", partial=entry),
                       _fault())
    backup = Scripted(*_ok("hello"))
    provider = FailoverProvider(primary, [("b", "x")], lambda n: backup)

    events = await _run(provider)

    assert backup.models == []
    assert isinstance(events[-1], StreamFaultEvent)


@pytest.mark.asyncio
async def test_last_route_failure_is_reported():
    provider = FailoverProvider(Scripted(_fault()), [("b", "x")], lambda n: Scripted(_fault()))
    events = await _run(provider)
    assert isinstance(events[-1], StreamFaultEvent)


@pytest.mark.asyncio
async def test_fallback_without_credentials_is_skipped():
    def resolve(name):
        if name == "nokey":
            raise RuntimeError("no key")
        return Scripted(*_ok("ok"))

    provider = FailoverProvider(Scripted(_fault()), [("nokey", "a"), ("good", "b")], resolve)
    events = await _run(provider)
    assert provider.last_route == ("good", "b")
    assert isinstance(events[-1], StreamCloseEvent)
