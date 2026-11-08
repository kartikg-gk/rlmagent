import asyncio
import copy
import json

import pytest
from test_contract import VALID

from rlmagent import factory_graph
from rlmagent.contract import ContractError


def manifest(tmp_path, dependencies):
    stages = {}
    for name, after in dependencies.items():
        contract = copy.deepcopy(VALID)
        contract["workdir"] = str(tmp_path / name)
        contract["trace"]["file"] = str(tmp_path / f"{name}.jsonl")
        stages[name] = {"contract": contract, "prompt": name, "after": after}
    path = tmp_path / "factory.json"
    path.write_text(json.dumps({"format": "rlmagent.factory/2", "parallel": 2, "stages": stages}))
    return path


async def test_graph_runs_independent_stages_concurrently_and_hands_results_to_join(tmp_path, monkeypatch):
    path = manifest(tmp_path, {"join": ["left", "right"], "left": [], "right": []})
    entered = set()
    branches = asyncio.Event()
    seen = {}

    async def stage_runner(stage, inputs, directory, name):
        seen[name] = inputs
        if not inputs:
            entered.add(name)
            if len(entered) == 2:
                branches.set()
            await asyncio.wait_for(branches.wait(), timeout=1)
        return {"status": "done", "output": stage.prompt}

    monkeypatch.setattr(factory_graph, "_run_stage", stage_runner)
    assert await factory_graph.execute_graph(*factory_graph.load_graph(path), tmp_path) == 0
    assert seen["join"] == {"left": {"status": "done", "output": "left"},
                            "right": {"status": "done", "output": "right"}}
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "done"
    seen.clear()
    assert await factory_graph.execute_graph(*factory_graph.load_graph(path), tmp_path) == 0
    assert seen == {}


def test_graph_rejects_cycles_before_starting(tmp_path):
    path = manifest(tmp_path, {"a": ["b"], "b": ["a"]})
    with pytest.raises(ContractError, match="cycle"):
        factory_graph.load_graph(path)


async def test_workflow_caps_live_agents_and_keeps_transition_count_across_retry(tmp_path, monkeypatch):
    path = manifest(tmp_path, {"a": [], "b": [], "c": []})
    doc = json.loads(path.read_text())
    doc["policy"] = {"max_live_agents": 9, "max_transitions": 2}
    path.write_text(json.dumps(doc))
    calls = []
    active = 0

    async def run_stage(stage, inputs, directory, name):
        nonlocal active
        active += 1
        assert active == 1
        calls.append(name)
        await asyncio.sleep(0)
        active -= 1
        return {"status": "done"}

    monkeypatch.setattr(factory_graph, "_run_stage", run_stage)
    graph = factory_graph.load_graph(path)
    assert await factory_graph.execute_graph(*graph, tmp_path) == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["transitions"] == 2
    assert state["stop_reason"] == "workflow transition limit reached"
    assert state["stages"]["c"]["status"] == "waiting"
    assert await factory_graph.execute_graph(*graph, tmp_path, retry=True) == 1
    assert calls == ["a", "b"]


async def test_workflow_failure_policy_drains_active_stages_without_admitting_new_work(tmp_path, monkeypatch):
    path = manifest(tmp_path, {"fail": [], "active": [], "unstarted": []})
    doc = json.loads(path.read_text())
    doc["policy"] = {"on_failure": "stop"}
    path.write_text(json.dumps(doc))
    calls = []

    async def run_stage(stage, inputs, directory, name):
        calls.append(name)
        if name == "fail":
            return {"status": "failed"}
        await asyncio.sleep(0.01)
        return {"status": "done"}

    monkeypatch.setattr(factory_graph, "_run_stage", run_stage)
    assert await factory_graph.execute_graph(*factory_graph.load_graph(path), tmp_path) == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert calls == ["fail", "active"]
    assert state["stages"]["active"]["status"] == "done"
    assert state["stages"]["unstarted"]["status"] == "waiting"
    assert state["stop_reason"] == "stage failed: fail"


def test_factory_launch_pins_topology_before_the_runner_starts(tmp_path, monkeypatch, capsys):
    path = manifest(tmp_path, {"source": [], "join": ["source"]})
    monkeypatch.setenv("TEST_KEY", "fixture-only")
    monkeypatch.setattr(factory_graph, "rlm_agent_home", lambda: tmp_path)
    monkeypatch.setattr(factory_graph.subprocess, "Popen", lambda *args, **kwargs: None)
    assert factory_graph.command("launch", [str(path)]) == 0
    run_id = capsys.readouterr().out.strip()
    directory = tmp_path / "factory-runs" / run_id
    assert factory_graph.run_status(directory)["graph"]["join"]["after"] == ["source"]
    doc = json.loads(path.read_text())
    doc["stages"]["join"]["after"] = []
    path.write_text(json.dumps(doc))
    with pytest.raises(ContractError, match="since this run was launched"):
        factory_graph.command("_execute", [run_id])
    assert factory_graph.run_status(directory)["status"] == "failed"


async def test_foreach_joins_in_order_and_retries_only_failed_items(tmp_path, monkeypatch):
    path = manifest(tmp_path, {"source": [], "map": ["source"], "join": ["map"]})
    doc = json.loads(path.read_text())
    doc["stages"]["map"]["foreach"] = "source"
    path.write_text(json.dumps(doc))
    calls = []

    async def stage_runner(stage, inputs, directory, name, *, item=None):
        calls.append(name)
        if name == "source":
            return {"status": "done", "output": '["one", null, "three"]'}
        if item is not None:
            if name == "map.item-1" and calls.count(name) == 1:
                return {"status": "failed", "output": "temporary failure"}
            return {"status": "done", "output": json.dumps(item["input"])}
        assert inputs == {"map": {"status": "done", "output": ['"one"', "null", '"three"']}}
        return {"status": "done", "output": "joined"}

    monkeypatch.setattr(factory_graph, "_run_stage", stage_runner)
    graph = factory_graph.load_graph(path)
    assert await factory_graph.execute_graph(*graph, tmp_path) == 1
    assert calls == ["source", "map.item-0", "map.item-1", "map.item-2"]
    with pytest.raises(ContractError, match="retry-stages"):
        await factory_graph.execute_graph(*graph, tmp_path)
    assert await factory_graph.execute_graph(*graph, tmp_path, retry=True) == 0
    assert calls == ["source", "map.item-0", "map.item-1", "map.item-2", "map.item-1", "join"]


async def test_factory_view_follows_checkpoint_and_shows_item_output(tmp_path):
    from textual.widgets import DataTable, Static

    from rlmagent.tui.factory_view import FactoryGraphApp

    graph = {"source": {"after": [], "when": "all_done"},
             "map": {"after": ["source"], "when": "all_done", "foreach": "source"}}
    (tmp_path / "run.json").write_text(json.dumps({"graph": graph}))
    app = FactoryGraphApp(tmp_path)
    async with app.run_test() as pilot:
        table = app.query_one(DataTable)
        assert table.row_count == 2
        assert [str(value) for value in table.get_row_at(1)] == [
            "map", "source", "all_done · foreach source", "waiting"]
        (tmp_path / "state.json").write_text(json.dumps({
            "id": tmp_path.name, "status": "done", "graph": graph,
            "stages": {"source": {"status": "done"}, "map": {
                "status": "done", "output": ["mapped"],
                "items": [{"status": "done", "output": "mapped"}],
            }},
        }))
        app.refresh_run()
        assert table.row_count == 3
        table.move_cursor(row=2)
        await pilot.pause()
        assert app.query_one("#stage-result", Static).content == "mapped"
