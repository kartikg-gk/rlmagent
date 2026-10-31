from types import SimpleNamespace

from rlmagent.execution_log import records


def test_step_record_contains_code_output_error_usage_and_timing():
    start = {"kind": "run_started", "run_id": "child", "parent_run_id": "root", "depth": 1}
    event = {
        "kind": "step",
        "timestamp": 1,
        "run_id": "child",
        "step": 2,
        "code": "broken()",
        "output": "failed",
        "error": True,
        "usage": {"prompt_tokens": 4},
        "timestamps": {"execution_end": "now"},
    }
    (row,) = records(event, [start, event])
    assert row["event_type"] == "execution_result"
    assert row["parent_run_id"] == "root" and row["depth"] == 1
    assert row["hasError"] and row["usage"]["prompt_tokens"] == 4
    assert row["code"] == "broken()" and row["timestamps"]["execution_end"] == "now"
    final = {"kind": "run_completed", "timestamp": 2, "run_id": "child", "result": 42}
    assert [row["event_type"] for row in records(final, [start, final])] == [
        "final_result",
        "agent_end",
    ]


def test_cloud_cli_tokens_are_cached_and_errors_do_not_include_credentials(monkeypatch):
    from rlmagent.model import vertex

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fixture-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setattr(vertex.shutil, "which", lambda name: "gcloud")
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["timeout"] == 20
        return SimpleNamespace(returncode=0, stdout="synthetic-token", stderr="")

    monkeypatch.setattr(vertex.subprocess, "run", run)
    resolver = vertex.ApplicationCredentials()
    assert resolver._gcloud().api_key == "synthetic-token"
    assert resolver._gcloud().api_key == "synthetic-token"
    assert len(calls) == 1
    resolver._expires = 0
    resolver._gcloud()
    assert len(calls) == 2


async def test_default_backend_uses_integrated_manual_guidance(tmp_path, monkeypatch):
    import json
    from _scripted import text_turn
    import importlib

    cli = importlib.import_module("rlmagent.cli.main")
    from rlmagent.model.scripted import ReplayProvider
    from rlmagent.runtime import build_session
    from rlmagent.kernel.wasm import available

    if not available():
        import pytest

        pytest.skip("WASM unavailable")
    entry = {
        "id": "release-check",
        "kind": "prompt",
        "title": "Release check",
        "content": "Use the focused release checker.",
    }
    provider = ReplayProvider(
        [
            text_turn(json.dumps({"tool": "guidance_inspect", "arguments": {}})),
            text_turn(
                json.dumps(
                    {"tool": "guidance_edit", "arguments": {"action": "create", "entry": entry}}
                )
            ),
            text_turn(
                json.dumps(
                    {
                        "summary": "Keep check",
                        "evidence": "Synthetic correction",
                        "expected": "Reuse check",
                    }
                )
            ),
            text_turn(
                "```python\nitem = await guidance('release-check')\nFINAL(item['content'])\n```"
            ),
        ]
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_resolve_provider", lambda name: provider)
    args = cli._build_run_parser().parse_args(
        [
            "--provider",
            "scripted",
            "--model",
            "fixture",
            "--no-network",
            "--no-session",
            "--no-auto-improve",
            "--no-approve",
            "--max-calls",
            "10",
            "-p",
        ]
    )
    session = await build_session(args)
    try:
        await session.handle_command("/refine")
        await session.refinement.task
        assert session.refinement.phase == "completed", session.refinement.status()
        assert entry["content"] in session.backend.instructions
        async for _ in session.submit("Read the saved guidance through Python and FINAL it."):
            pass
        assert session.transcript[-1].text == entry["content"]
    finally:
        await session.shutdown()
