import json
from types import SimpleNamespace

import httpx
import pytest

from rlmagent.agents.approval import Approver, ONCE, Refused
from rlmagent.agents.workspace import Workspace


async def test_approval_shows_diff_and_refuses_stale_file(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n")

    async def approve(request):
        shown = request.describe()
        assert "-value = 1" in shown and "+value = 2" in shown
        target.write_text("user edit\n")
        return ONCE

    workspace = Workspace(tmp_path, Approver("ask", approve))
    with pytest.raises(Refused, match="changed while approval"):
        await workspace.edit_file("sample.py", "1", "2")
    assert target.read_text() == "user edit\n"


async def test_instruction_discovery_skips_build_trees_and_confines_reads(tmp_path):
    (tmp_path / "AGENTS.md").write_text("root instructions")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "SKILL.md").write_text("generated")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "SKILL.md").write_text("skill text")
    workspace = Workspace(tmp_path, Approver("deny"))
    assert await workspace.instruction_docs() == ["AGENTS.md", "skills/SKILL.md"]
    assert await workspace.instruction_docs("skills/SKILL.md") == "skill text"
    with pytest.raises(ValueError, match="outside"):
        await workspace.instruction_docs("../AGENTS.md")


async def test_fetch_bounds_content_and_rejects_non_http(monkeypatch):
    from rlmagent.network import fetch_text

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="abcdefghij")),
            **kw,
        ),
    )
    result = await fetch_text("https://example.test", max_chars=5, allow_private=True)
    assert result["text"] == "abcde" and result["truncated"] is True
    with pytest.raises(ValueError):
        await fetch_text("file:///private")


async def test_remote_discovery_checks_headers_and_custom_endpoint():
    from rlmagent.model.discovery import discover_models

    def reply(request):
        assert request.url == "https://custom.test/v1/models"
        assert request.headers["authorization"] == "Bearer test-placeholder"
        return httpx.Response(200, json={"data": [{"id": "z"}, {"id": "a"}, {"id": "a"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        assert await discover_models(
            "openai", api_key="test-placeholder", base_url="https://custom.test/v1", client=client
        ) == ["a", "z"]


async def test_backend_recovers_edit_and_logs_complete_usage(tmp_path):
    from _scripted import text_turn
    from rlmness import Allowance
    from rlmagent.backend import BackendProvider
    from rlmagent.harness.contracts.transcript import HumanEntry, UsageStats
    from rlmagent.harness.provider.wire import StreamCloseEvent
    from rlmagent.model.scripted import ReplayProvider

    (tmp_path / "test.txt").write_text("old")
    turn = text_turn(
        "```python\nr = await edit_file('test.txt', 'missing', 'new')\nawait write_file('result.txt', r)\nFINAL(r.startswith('Error:'))\n```"
    )
    turn[-1].message.usage = UsageStats(
        input=11,
        output=4,
        cache_read=8,
        cache_write=6,
        cache_write_1h=2,
        reasoning=3,
        total_tokens=29,
        cost={"total": 0.01, "cache_write": 0.002},
    )
    provider = ReplayProvider([turn])
    tree = SimpleNamespace(
        cwd=str(tmp_path),
        approver=Approver("auto"),
        network=False,
        cell_timeout=30,
        allowance=Allowance(max_calls=4, max_cost=2),
    )
    backend = BackendProvider(provider, provider, tree)
    backend.event_path = tmp_path / "events.jsonl"
    try:
        events = [
            e
            async for e in backend.stream_response(
                model="m",
                system="",
                messages=[HumanEntry(content="work")],
                tools=[SimpleNamespace(name="python")],
            )
        ]
        assert isinstance(events[-1], StreamCloseEvent), events
        assert events[-1].message.text == "True"
        assert events[-1].message.usage == turn[-1].message.usage
        assert "found 0 times" in (tmp_path / "result.txt").read_text()
        logged = [json.loads(line) for line in backend.event_path.read_text().splitlines()]
        assert any(row["kind"] == "code_generated" for row in logged)
        assert any(
            row["kind"] == "model_usage" and row["usage"]["cache_write"] == 6 for row in logged
        )
        assert any(
            row["kind"] == "host_result" and str(row["result"]).startswith("Error:")
            for row in logged
            if isinstance(row.get("result"), str)
        )
    finally:
        await backend.close()


async def test_cloud_credentials_refresh_and_project_resolution(monkeypatch):
    import sys
    from rlmagent.model.vertex import ApplicationCredentials

    credentials = SimpleNamespace(valid=False, token="test-placeholder")
    refreshes = []

    def refresh(request):
        refreshes.append(request)
        credentials.valid = True

    credentials.refresh = refresh
    monkeypatch.setitem(
        sys.modules,
        "google",
        SimpleNamespace(
            auth=SimpleNamespace(default=lambda **kw: (credentials, "detected-project"))
        ),
    )
    monkeypatch.setitem(sys.modules, "google.auth", sys.modules["google"].auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "google.auth.transport.requests", SimpleNamespace(Request=object)
    )
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    resolver = ApplicationCredentials()
    first = await resolver()
    await resolver()
    assert len(refreshes) == 1
    assert (
        first.base_url
        == "https://aiplatform.googleapis.com/v1beta1/projects/detected-project/locations/global/endpoints/openapi"
    )


async def test_picker_custom_model_and_endpoint():
    from textual.app import App
    from textual.widgets import Input, ListView
    from rlmagent.config.store import provider_info
    from rlmagent.tui.model_screen import ModelScreen

    class PickerApp(App):
        def on_mount(self):
            self.picker = ModelScreen()
            self.push_screen(self.picker)

    app = PickerApp()
    async with app.run_test() as pilot:
        screen = app.picker
        screen._show_models(provider_info("ollama"))
        screen.action_endpoint()
        field = screen.query_one("#model-key", Input)
        field.value = "http://localhost:1234/v1"
        await pilot.press("enter")
        await pilot.pause()
        assert screen._endpoint == "http://localhost:1234/v1"
        assert screen.query_one("#model-list", ListView).display
        screen._on_model_chosen("__custom__")
        field.value = "custom-model"
        await pilot.press("enter")
        await pilot.pause()
        assert screen._model == "custom-model"


async def test_credential_check_uses_private_endpoint_and_rejects_bad_key(monkeypatch):
    from rlmagent.model.discovery import check_provider

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")

    def reply(request):
        assert request.url == "https://openrouter.ai/api/v1/key"
        return httpx.Response(401, json={"error": "invalid credential"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await check_provider("openrouter", client=client)


async def test_approval_screen_shows_literal_diff_and_accepts_keyboard_shortcut():
    from textual.app import App
    from textual.widgets import Static
    from rlmagent.agents.approval import AUTO, Request
    from rlmagent.tui.approval_screen import ApprovalScreen

    class ApprovalApp(App):
        def on_mount(self):
            self.choice = None
            self.dialog = ApprovalScreen(
                Request(
                    "edit",
                    "file.py",
                    ("old", "new"),
                    preview="--- a/file.py\n+++ b/file.py\n-[old]\n+[new]",
                )
            )
            self.push_screen(self.dialog, lambda value: setattr(self, "choice", value))

    app = ApprovalApp()
    async with app.run_test() as pilot:
        assert "[old]" in app.dialog.query_one("#approval-body", Static).content.code
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.choice == AUTO


def test_snapshot_git_leaves_protocol_input_unread(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    import pytest
    if shutil.which('git') is None:
        pytest.skip('git not on PATH')
    work = tmp_path / 'workspace'
    work.mkdir()
    code = (
        'import sys,json; from pathlib import Path; from rlmagent.undo import Shadow; '
        'shadow=Shadow(sys.argv[1],Path(sys.argv[2])); shadow._init(); '
        'print(json.dumps([shadow._git("hash-object","--stdin"),sys.stdin.read()]))'
    )
    result = subprocess.run([sys.executable, '-c', code, str(work), str(tmp_path / 'home')],
                            input='protocol message', text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ['e69de29bb2d1d6434b8b29ae775ad8c2e48c5391', 'protocol message']
