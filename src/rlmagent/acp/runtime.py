"""Explicit, in-memory configuration for an editor-owned session."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rlmagent.contract import ACCOUNT_PROVIDERS, Contract, ContractError, parse_contract

KEY = "rlmagent/runtime-v1"


@dataclass(frozen=True)
class Runtime:
    contract: Contract
    api_key: str | None = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(repr=False)


def parse(value) -> Runtime:
    if not isinstance(value, dict) or value.keys() != {"contract", "credential"}:
        raise ContractError("runtime requires contract and credential objects")
    credential = value["credential"]
    contract = parse_contract(value["contract"])
    if contract.model.provider in ACCOUNT_PROVIDERS:
        if credential != {"account": contract.model.provider}:
            raise ContractError("Account runtime credential must explicitly name its provider")
        return Runtime(contract, None, ())
    if not isinstance(credential, dict) or credential.keys() != {"api_key", "headers"}:
        raise ContractError("runtime credential requires api_key and headers")
    key, headers = credential["api_key"], credential["headers"]
    if (
        not isinstance(key, str)
        or not key
        or any(char.isspace() for char in key)
        or not isinstance(headers, dict)
    ):
        raise ContractError("runtime credential has invalid types")
    reserved = {
        "authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "idempotency-key",
        "x-stainless-retry-count",
        "x-rlmagent-attempt-id",
        "x-rlmagent-request-id",
    }
    for name, content in headers.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name) is None
            or name.lower() in reserved
            or not isinstance(content, str)
            or any(c in name + content for c in "\r\n")
        ):
            raise ContractError("runtime contains an invalid or reserved provider header")
    return Runtime(contract, key, tuple(headers.items()))


async def build(runtime: Runtime):
    from rlmagent.contract_run import _provider, build_contract_session
    from rlmagent.model.settings import Credential

    spec = runtime.contract.model
    credential = (Credential(runtime.api_key, spec.base_url, runtime.headers)
                  if runtime.api_key is not None else None)
    provider = _provider(runtime.contract, credential)
    return await build_contract_session(
        runtime.contract, provider=provider, engine=runtime.contract.engine
    )
