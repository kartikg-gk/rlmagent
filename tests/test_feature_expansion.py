import copy
import json
from types import SimpleNamespace

import pytest
from _scripted import text_turn
from test_acp_server import Pipe, _stop
from test_contract import VALID
from test_refinement import ENTRY, finish, good_turns, session_for

from rlmagent.acp.runtime import KEY, parse
from rlmagent.acp.server import Server
from rlmagent.contract import ContractError
from rlmagent.harness.contracts.transcript import HumanEntry
from rlmagent.model.scripted import ReplayProvider


def runtime_document(tmp_path):
    contract = copy.deepcopy(VALID)
    contract["workdir"] = str(tmp_path)
    contract["tools"] = []
    contract["trace"]["file"] = str(tmp_path / "trace.jsonl")
    return {"contract": contract, "credential": {"api_key": "fixture-private", "headers": {}}}


async def test_explicit_acp_runtime_ignores_environment_credentials(tmp_path, monkeypatch):
    document = runtime_document(tmp_path)
    monkeypatch.delenv("TEST_KEY", raising=False)
    model = ReplayProvider([text_turn("configured response")])
    profiles = []

    def provider(profile):
        profiles.append(profile)
        return model

    monkeypatch.setattr("rlmagent.model.oai_compatible.OpenAIProvider", provider)
    pipe = Pipe()
    server = Server(pipe.read_line, pipe.write)
    import asyncio

    task = asyncio.create_task(server.serve())
    try:
        pipe.send(1, "session/new", {"cwd": str(tmp_path), "_meta": {KEY: document}})
        sid = (await pipe.reply(1))["result"]["sessionId"]
        pipe.send(
            2, "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "hello"}]}
        )
        reply = (await pipe.reply(2))["result"]
        assert reply["_meta"]["rlmagent/session-v1"]["limits"]["max_calls"] == 200
        assert profiles[0].credential.api_key == "fixture-private"
        assert "fixture-private" not in json.dumps(pipe.sent)
        assert "fixture-private" not in (tmp_path / "trace.jsonl").read_text()
    finally:
        await _stop(pipe, task)


def test_explicit_runtime_rejects_reserved_headers(tmp_path):
    document = runtime_document(tmp_path)
    document["credential"]["headers"] = {"Idempotency-Key": "override"}
    with pytest.raises(ContractError):
        parse(document)


@pytest.mark.parametrize(
    "field,value",
    [("max_calls", -1), ("max_cost", float("nan")), ("max_live", 0), ("cell_timeout", 0)],
)
def test_explicit_runtime_rejects_invalid_limits(tmp_path, field, value):
    document = runtime_document(tmp_path)
    document["contract"]["limits"][field] = value
    with pytest.raises(ContractError):
        parse(document)


def test_explicit_runtime_owns_its_configuration(tmp_path):
    document = runtime_document(tmp_path)
    runtime = parse(document)
    document["contract"]["limits"]["max_calls"] = 1
    assert runtime.contract.limits.max_calls == 200
    assert runtime.contract.raw["limits"]["max_calls"] == 200


async def test_refinement_global_disable_and_rollback(tmp_path):
    session = await session_for(good_turns(), tmp_path / "state.json")
    session.refinement.home = tmp_path / "home"
    try:
        await session.handle_command("/refine keep the focused check")
        await finish(session)
        assert ENTRY["content"] in session._system
        await session.handle_command("/refine off")
        assert ENTRY["content"] not in session._system
        assert "disabled" in await session.handle_command("/refine another update")
        assert session.refinement.store.read().revision == 1
        await session.handle_command("/refine on")
        assert ENTRY["content"] in session._system
        assert "Restored" in await session.handle_command("/refine rollback 0")
        assert ENTRY["content"] not in session._system
        assert session.refinement.store.read().revision == 2
    finally:
        await session.shutdown()


def test_html_transcript_escapes_untrusted_content():
    from rlmagent.session_html import render

    exported = render([HumanEntry(content='<script>alert("x")</script>')])
    assert "<script>" not in exported and "&lt;script&gt;" in exported
    assert "Content-Security-Policy" in exported


def test_session_import_preserves_branch_and_unicode(tmp_path):
    from rlmagent.harness.session.records import SessionMetaRecord, TipRecord, TranscriptRecord
    from rlmagent.harness.session.store import deserialize_records, serialize_record
    from rlmagent.session_import import import_session

    root = SessionMetaRecord(cwd=str(tmp_path))
    message = TranscriptRecord(parent_id=root.id, message=HumanEntry(content="first\u2028second"))
    tip = TipRecord(parent_id=message.id, entry_id=message.id)
    source = tmp_path / "export.jsonl"
    source.write_text(
        "".join(serialize_record(row) for row in (root, message, tip)), encoding="utf-8"
    )
    identity = import_session(source, tmp_path / "sessions")
    imported = deserialize_records(
        (tmp_path / "sessions" / f"{identity}.jsonl").read_text(encoding="utf-8").split("\n")
    )
    assert imported[1].message.content == "first\u2028second"
    assert imported[-1].entry_id == message.id
    source.write_text(serialize_record(root) + serialize_record(root), encoding="utf-8")
    with pytest.raises(ValueError):
        import_session(source, tmp_path / "sessions")


def test_failed_package_resource_validation_keeps_prior_version(tmp_path, monkeypatch):
    from pathlib import Path

    from rlmagent import packages

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    valid = True

    def install(command, **kw):
        target = Path(command[command.index("--target") + 1])
        target.mkdir(parents=True)
        (target / "skills").mkdir()
        (target / "rlmagent-resources.json").write_text(
            json.dumps({"skills": ["skills" if valid else "../outside"]})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(packages.subprocess, "run", install)
    assert packages.run(["install", "sample", "sample-package"]) == 0
    previous = packages._records()
    valid = False
    assert packages.run(["update", "sample"]) == 1
    assert packages._records() == previous


def test_project_packages_obey_discovery_trust_and_home(tmp_path, monkeypatch):
    from pathlib import Path

    from rlmagent import packages
    from rlmagent.discovery import RlmAgentPaths, skill_search_paths

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "unused-home"))

    def install(command, **kw):
        location = Path(command[command.index("--target") + 1])
        (location / "skills").mkdir(parents=True)
        (location / "rlmagent-resources.json").write_text(json.dumps({"skills": ["skills"]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(packages.subprocess, "run", install)
    assert packages.run(["install", "sample", "sample-package", "--local"]) == 0
    trusted = RlmAgentPaths(home=tmp_path / "explicit-home", project=project, project_enabled=True)
    blocked = RlmAgentPaths(home=trusted.home, project=project, project_enabled=False)
    resources = packages.resource_dirs("skills", paths=trusted)
    assert len(resources) == 1 and resources[0].is_relative_to(project)
    assert resources[0] in skill_search_paths(trusted)
    assert resources[0] not in skill_search_paths(blocked)
    assert packages._records() == {}
    assert packages.run(["remove", "sample", "--local"]) == 0


async def test_tui_loads_user_keymap(tmp_path, monkeypatch):
    from rlmagent.tui.app import RlmAgentApp
    from rlmagent.tui.inspect import HelpScreen

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))
    (tmp_path / "keybindings.json").write_text(json.dumps({"help": "f6"}))
    async with RlmAgentApp().run_test() as pilot:
        await pilot.pause()
        assert pilot.app._keymap == {"help": "f6"}
        assert pilot.app.screen.active_bindings["f6"].binding.action == "show_help"
        await pilot.press("f6")
        assert isinstance(pilot.app.screen, HelpScreen)


def test_package_registry_roundtrip(tmp_path, monkeypatch):
    from rlmagent import packages

    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path))

    def install(command, *, check):
        target = command[command.index("--target") + 1]
        from pathlib import Path

        location = Path(target)
        (location / "skills").mkdir(parents=True)
        (location / "rlmagent-resources.json").write_text(json.dumps({"skills": ["skills"]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(packages.subprocess, "run", install)
    assert packages.run(["install", "sample", "example-skill"]) == 0
    assert len(packages.resource_dirs("skills")) == 1
    assert packages.run(["update", "sample"]) == 0
    assert packages.run(["remove", "sample"]) == 0
    assert packages.resource_dirs("skills") == []


async def test_native_google_stream_preserves_signature_and_usage(monkeypatch):
    import httpx

    from rlmagent.model import google
    from rlmagent.model.settings import Credential

    chunks = [
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking", "thought": True},
                            {
                                "functionCall": {"name": "python", "args": {"code": "print(42)"}},
                                "thoughtSignature": "fixture-signature",
                            },
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 2,
                "totalTokenCount": 15,
            },
        }
    ]

    def serve(request):
        assert request.headers["x-goog-api-key"] == "fixture"
        assert request.url.path.endswith("/models/gemini-test:streamGenerateContent")
        body = json.loads(request.content)
        assert body["systemInstruction"]["parts"][0]["text"] == "system"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(row) + "\n\n" for row in chunks),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    monkeypatch.setattr(google, "build_async_client", lambda **kw: client)
    provider = google.GoogleProvider(Credential("fixture", "https://model.test/v1beta"))
    try:
        events = [
            event
            async for event in provider.stream_response(
                model="gemini-test", system="system", messages=[HumanEntry(content="go")], tools=[]
            )
        ]
        result = events[-1]
        assert result.type == "done" and result.reason == "toolUse"
        assert result.message.tool_calls[0].thought_signature == "fixture-signature"
        assert result.message.usage.output == 5
        encoded = google.content([result.message])
        assert encoded[0]["parts"][-1]["thoughtSignature"] == "fixture-signature"
    finally:
        await provider.close()


async def test_azure_and_mistral_choose_explicit_endpoints(monkeypatch):
    from rlmagent.config.config import resolve_provider

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://azure.test/openai/v1")
    monkeypatch.setenv("MISTRAL_API_KEY", "fixture")
    for name, endpoint in (("azure", "responses"), ("mistral", "chat")):
        provider = resolve_provider(name)
        try:
            assert provider._profile.endpoint == endpoint
            if name == "azure":
                headers = provider._build_headers(provider._profile.credential)
                assert headers["api-key"] == "fixture" and "Authorization" not in headers
        finally:
            await provider.close()
