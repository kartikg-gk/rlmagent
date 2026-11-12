import json

import httpx
import pytest

from rlmagent.agents.budget import BudgetedProvider, ContractAllowance, NewTokenBudget
from rlmagent.harness.contracts.transcript import HumanEntry
from rlmagent.harness.provider.wire import StreamCloseEvent, StreamFaultEvent
from rlmagent.model.external import ExternalAgentProvider
from rlmagent.model.oai_compatible import OpenAIProvider
from rlmagent.model.settings import Credential, OpenAIProfile


BASE = "https://api.openai.com/v1"
MESSAGES = [HumanEntry(content="count this prompt exactly")]


def allowance(limit):
    return ContractAllowance(new_token_budget=NewTokenBudget(limit, strict=True))


async def collect(provider, model="gpt-5"):
    return [event async for event in provider.stream_response(
        model=model, system="same instructions", messages=MESSAGES, tools=[])]


async def make_provider(transport, name="openai"):
    provider = OpenAIProvider(OpenAIProfile(name=name, credential=Credential("fixture", BASE),
                                             endpoint="responses", transport="http"))
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return provider


def completed(input_tokens, output_tokens, cached_tokens=0):
    response = {"type": "response.completed", "response": {
        "id": "resp_fixture", "status": "completed", "output": [],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens,
                  "input_tokens_details": {"cached_tokens": cached_tokens}},
    }}
    return httpx.Response(200, text="event: response.completed\ndata: " + json.dumps(response) + "\n\n")


async def test_strict_responses_counts_exact_payload_and_caps_generation():
    requests = []

    def serve(request):
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 33})
        return completed(33, 19, cached_tokens=20)

    native = await make_provider(serve)
    budgeted = BudgetedProvider(native, allowance(80))
    try:
        assert isinstance((await collect(budgeted))[-1], StreamCloseEvent)
        assert budgeted.allowance.new_token_budget.used == 32
        assert [route for route, _ in requests] == ["/v1/responses/input_tokens", "/v1/responses"]
        counted, generated = (body for _, body in requests)
        assert generated["max_output_tokens"] == 47
        assert not generated["store"]
        assert {key: generated[key] for key in counted} == counted

        result = (await collect(budgeted))[-1]
        assert isinstance(result, StreamFaultEvent)
        assert "less than 16 tokens" in result.error.error_message
        assert [route for route, _ in requests].count("/v1/responses") == 1
    finally:
        await native.close()


async def test_strict_stream_does_not_retry_an_uncertain_attempt():
    paths = []

    def serve(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 10})
        return httpx.Response(503, json={"error": {"message": "fixture failure"}})

    native = await make_provider(serve)
    budgeted = BudgetedProvider(native, allowance(90))
    try:
        assert isinstance((await collect(budgeted))[-1], StreamFaultEvent)
        assert isinstance((await collect(budgeted))[-1], StreamFaultEvent)
        assert paths == ["/v1/responses/input_tokens", "/v1/responses"]
        assert budgeted.allowance.new_token_budget.unmeasured
    finally:
        await native.close()


@pytest.mark.parametrize(("name", "model", "reason"), [
    ("codex", "gpt-5", "Codex subscription does not accept"),
    ("openai", "cli:claude-code", "no verified input counter"),
])
async def test_strict_subscription_requests_never_start(name, model, reason):
    def unexpected(request):
        pytest.fail(f"strict mode must not send {request.url.path} on this route")

    native = await make_provider(unexpected, name)
    budgeted = BudgetedProvider(ExternalAgentProvider(fallback=native), allowance(100_000))
    try:
        fault = (await collect(budgeted, model))[-1]
        assert isinstance(fault, StreamFaultEvent)
        assert reason in fault.error.error_message
        assert budgeted.allowance.calls == 0
    finally:
        await native.close()


async def test_reported_usage_that_exceeds_a_reservation_stops_the_tree():
    def serve(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 20})
        return completed(20, 45)

    native = await make_provider(serve)
    budgeted = BudgetedProvider(native, allowance(60))
    try:
        fault = (await collect(budgeted))[-1]
        assert isinstance(fault, StreamFaultEvent)
        assert "did not match the strict reservation" in fault.error.error_message
        assert budgeted.allowance.new_token_budget.unmeasured
    finally:
        await native.close()
