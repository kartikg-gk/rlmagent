"""Gemini content streaming through the native generate-content endpoint."""

from __future__ import annotations

import json
import os
from urllib.parse import quote
from uuid import uuid4

import httpx

from rlmagent.harness.contracts.transcript import ModelEntry
from rlmagent.harness.provider.wire import StreamFaultEvent
from rlmagent.model._oai.normalize import EventAssembler
from rlmagent.model._oai.parsers import (
    CallBegin,
    CallResolved,
    Finished,
    Metadata,
    ReasoningChunk,
    TextChunk,
)
from rlmagent.model._oai.transport import HttpStreamError, open_event_stream
from rlmagent.model.settings import ConfigError, Credential
from rlmagent.model.transport.client import build_async_client


def content(messages) -> list[dict]:
    result = []
    for entry in messages:
        role = getattr(entry, "role", "")
        if role == "toolResult":
            parts = [
                {"functionResponse": {"name": entry.tool_name, "response": {"result": entry.text}}}
            ]
            sender = "user"
        elif role in ("user", "assistant"):
            sender = "model" if role == "assistant" else "user"
            blocks = entry.content
            parts = [{"text": blocks}] if isinstance(blocks, str) else []
            if not isinstance(blocks, str):
                for block in blocks:
                    if block.type == "text":
                        parts.append({"text": block.text})
                    elif block.type == "image":
                        parts.append(
                            {"inlineData": {"mimeType": block.mime_type, "data": block.data}}
                        )
                    elif block.type == "toolCall":
                        call = {"functionCall": {"name": block.name, "args": block.arguments}}
                        if block.thought_signature:
                            call["thoughtSignature"] = block.thought_signature
                        parts.append(call)
        elif hasattr(entry, "summary"):
            sender, parts = "user", [{"text": entry.summary}]
        else:
            continue
        if parts:
            if result and result[-1]["role"] == sender:
                result[-1]["parts"].extend(parts)
            else:
                result.append({"role": sender, "parts": parts})
    return result


class GoogleProvider:
    def __init__(self, credential: Credential):
        self.credential = credential
        self._client = build_async_client(timeout=httpx.Timeout(120))

    async def close(self):
        await self._client.aclose()

    async def stream_response(self, *, model, system, messages, tools, signal=None, cache_key=None):
        name = model.removeprefix("models/")
        payload = {
            "contents": content(messages),
            "systemInstruction": {"parts": [{"text": system}]},
        }
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in tools
                    ]
                }
            ]
        url = (
            self.credential.base_url.rstrip("/")
            + f"/models/{quote(name, safe='')}:streamGenerateContent?alt=sse"
        )
        headers = {
            "x-goog-api-key": self.credential.api_key,
            "Content-Type": "application/json",
            **dict(self.credential.extra_headers),
        }
        assembler = EventAssembler(model=model, provider="google", api="google-generate")
        signatures = {}
        usage, finish, index = {}, "stop", 0
        completed = False
        try:
            async for event in open_event_stream(
                self._client, url, payload, headers, signal=signal
            ):
                value = json.loads(event.data)
                signals = []
                if value.get("error") or value.get("promptFeedback", {}).get("blockReason"):
                    raise ValueError("Google declined the content request")
                if value.get("responseId"):
                    signals.append(Metadata(value["responseId"], value.get("modelVersion")))
                for candidate in value.get("candidates", [])[:1]:
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            signals.append(
                                ReasoningChunk(part["text"])
                                if part.get("thought")
                                else TextChunk(part["text"])
                            )
                        if "functionCall" in part:
                            call = part["functionCall"]
                            identity = call.get("id") or uuid4().hex
                            signatures[identity] = part.get("thoughtSignature")
                            signals.extend(
                                [
                                    CallBegin(index, identity, call["name"]),
                                    CallResolved(
                                        index,
                                        identity,
                                        call["name"],
                                        json.dumps(call.get("args", {})),
                                    ),
                                ]
                            )
                            index += 1
                            finish = "tool_calls"
                    reason = candidate.get("finishReason")
                    completed = completed or reason is not None
                    if reason == "MAX_TOKENS":
                        finish = "length"
                    elif reason not in (None, "STOP", "MAX_TOKENS"):
                        raise ValueError(f"Google generation ended with {reason}")
                if "usageMetadata" in value:
                    reported = value["usageMetadata"]
                    usage = {
                        "prompt_tokens": reported.get("promptTokenCount", 0),
                        "completion_tokens": reported.get("candidatesTokenCount", 0)
                        + reported.get("thoughtsTokenCount", 0),
                        "total_tokens": reported.get("totalTokenCount", 0),
                        "prompt_tokens_details": {
                            "cached_tokens": reported.get("cachedContentTokenCount", 0)
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": reported.get("thoughtsTokenCount", 0)
                        },
                    }
                for parsed in signals:
                    for output in assembler.accept(parsed):
                        yield output
            if signal is not None and signal.is_cancelled():
                yield StreamFaultEvent(
                    reason="aborted", error=ModelEntry(model=model, stop_reason="aborted")
                )
                return
            if not completed:
                raise ValueError("Google stream ended before a terminal candidate")
            assembler.accept(Finished(finish, usage))
            for output in assembler.finalize():
                if hasattr(output, "message"):
                    for call in output.message.tool_calls:
                        call.thought_signature = signatures.get(call.id)
                yield output
        except (httpx.HTTPError, HttpStreamError, ValueError, KeyError, TypeError) as exc:
            yield StreamFaultEvent(
                reason="error",
                error=ModelEntry(
                    model=model,
                    stop_reason="error",
                    error_message=f"Google request failed ({type(exc).__name__})",
                ),
            )


def build_google():
    from rlmagent.config.loader import load_api_key

    key = load_api_key("google")
    if not key:
        raise ConfigError("GOOGLE_API_KEY is not set")
    return GoogleProvider(
        Credential(
            key,
            os.environ.get("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        )
    )
