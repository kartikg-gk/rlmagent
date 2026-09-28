"""With the switch on, every model call carries one idempotency key, kept across its retries.

Off by default: the official clients do not send this header on the endpoints
used here, so nothing shows the providers accept it.
"""

from __future__ import annotations

import httpx

from test_provider_resilience import (
    _A_START,
    _A_TEXT,
    _CHAT_OK,
    _OVERLOAD,
    _a_error,
    _anthropic,
    _anthropic_sse,
    _chat_sse,
    _openai,
)
from rlmagent_harness.contracts.transcript import HumanEntry


def _recording(bodies: list[str]) -> tuple[httpx.AsyncClient, list[str | None]]:
    keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers.get("Idempotency-Key"))
        return httpx.Response(200, text=bodies[min(len(keys), len(bodies)) - 1])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), keys


async def _call(provider) -> None:
    async for _ in provider.stream_response(
        model="m", system="s", messages=[HumanEntry(content="q")], tools=[],
    ):
        pass


async def test_openai_retry_keeps_the_key_and_a_new_call_gets_a_new_one(monkeypatch) -> None:
    monkeypatch.setenv("RLM_AGENT_IDEMPOTENCY", "1")
    provider, _ = _openai([])
    provider._client, keys = _recording(
        [_chat_sse(_OVERLOAD), _chat_sse(*_CHAT_OK), _chat_sse(*_CHAT_OK)]
    )
    await _call(provider)
    await _call(provider)
    await provider.close()
    assert len(keys) == 3
    assert keys[0] and keys[0] == keys[1]
    assert keys[2] and keys[2] != keys[0]


async def test_anthropic_retry_keeps_the_key(monkeypatch) -> None:
    monkeypatch.setenv("RLM_AGENT_IDEMPOTENCY", "1")
    provider, _ = _anthropic([])
    provider._client, keys = _recording(
        [_anthropic_sse(_A_START, _a_error()), _anthropic_sse(_A_START, *_A_TEXT)]
    )
    await _call(provider)
    await provider.close()
    assert len(keys) == 2
    assert keys[0] and keys[0] == keys[1]


async def test_no_key_is_sent_unless_asked_for(monkeypatch) -> None:
    monkeypatch.delenv("RLM_AGENT_IDEMPOTENCY", raising=False)
    provider, _ = _openai([])
    provider._client, keys = _recording([_chat_sse(*_CHAT_OK)])
    await _call(provider)
    await provider.close()
    anthropic, _ = _anthropic([])
    anthropic._client, more = _recording([_anthropic_sse(_A_START, *_A_TEXT)])
    await _call(anthropic)
    await anthropic.close()
    assert keys == [None] and more == [None]
