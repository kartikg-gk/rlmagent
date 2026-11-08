"""Persistent dependency workflows built from existing contract sessions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from uuid import uuid4

from rlmagent.contract import Contract, ContractError, load_contract, parse_contract
from rlmagent.factory import HANDOFF_LIMIT, _NAME, _checkpoint
from rlmagent.harness.session.lease import SessionBusy, acquire
from rlmagent.model.external.cli import kill_tree, own_group
from rlmagent.plugins import rlm_agent_home

TERMINAL = {"done", "failed", "skipped"}
MAX_ITEMS = 128


@dataclass(frozen=True)
class Stage:
    contract: Contract
    prompt: str
    after: tuple[str, ...]
    when: str
    foreach: str | None = None


@dataclass(frozen=True)
class WorkflowPolicy:
    max_transitions: int | None = None
    max_live_agents: int | None = None
    on_failure: str = "continue"


def _agent_reservation(stage):
    limits = stage.contract.limits
    return 1 + (limits.max_live if limits.max_depth else 0)


def load_graph(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != "rlmagent.factory/2":
        raise ContractError("expected rlmagent.factory/2")
    if manifest.keys() - {"format", "parallel", "stages", "policy"}:
        raise ContractError("factory: unexpected graph fields")
    width, raw = manifest.get("parallel", 1), manifest.get("stages")
    if type(width) is not int or not 1 <= width <= 16:
        raise ContractError("factory.parallel must be an integer from 1 to 16")
    if not isinstance(raw, dict) or not 1 <= len(raw) <= 128:
        raise ContractError("factory.stages must contain 1 to 128 named stages")
    policy_doc = manifest.get("policy", {})
    if not isinstance(policy_doc, dict) or policy_doc.keys() - WorkflowPolicy.__dataclass_fields__.keys():
        raise ContractError("factory.policy has invalid fields")
    policy = WorkflowPolicy(**policy_doc)
    for field, minimum in (("max_transitions", 0), ("max_live_agents", 1)):
        value = getattr(policy, field)
        if value is not None and (type(value) is not int or value < minimum):
            raise ContractError(f"factory.policy.{field} must be an integer of at least {minimum}")
    if policy.on_failure not in ("continue", "stop"):
        raise ContractError("factory.policy.on_failure must be continue or stop")
    stages, traces = {}, set()
    for name, row in raw.items():
        if not _NAME.fullmatch(name) or not isinstance(row, dict) or row.keys() - {
            "contract", "prompt", "after", "when", "foreach",
        }:
            raise ContractError(f"invalid factory stage {name!r}")
        spec, prompt = row.get("contract"), row.get("prompt")
        after, when = row.get("after", []), row.get("when", "all_done")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError(f"{name}: a nonempty prompt is required")
        if not isinstance(after, list) or any(not isinstance(dep, str) or dep not in raw for dep in after):
            raise ContractError(f"{name}: dependencies must name existing stages")
        if len(set(after)) != len(after) or when not in ("all_done", "any_failed", "always"):
            raise ContractError(f"{name}: invalid dependency guard")
        if when == "any_failed" and not after:
            raise ContractError(f"{name}: any_failed requires a dependency")
        source = row.get("foreach")
        if "foreach" in row and (not isinstance(source, str) or source not in after):
            raise ContractError(f"{name}: foreach must name a direct dependency")
        contract = load_contract(path.parent / spec) if isinstance(spec, str) else parse_contract(spec)
        trace = Path(contract.trace.file).resolve()
        if trace in traces:
            raise ContractError(f"{name}: each stage needs a distinct trace")
        traces.add(trace)
        stages[name] = Stage(contract, prompt, tuple(after), when, source)
        if policy.max_live_agents is not None and _agent_reservation(stages[name]) > policy.max_live_agents:
            raise ContractError(f"{name}: contract exceeds the workflow live-agent limit")
    try:
        tuple(TopologicalSorter({name: stage.after for name, stage in stages.items()}).static_order())
    except CycleError as exc:
        raise ContractError("factory dependencies contain a cycle") from exc
    canonical = [(name, stage.contract.raw, stage.prompt, stage.after, stage.when,
                  *([stage.foreach] if stage.foreach else []))
                 for name, stage in stages.items()]
    content = [width, canonical]
    if policy_doc:
        content.append(policy_doc)
    fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return stages, width, fingerprint, policy


async def _run_stage(stage, inputs, directory, name, *, item=None):
    prompt = stage.prompt
    if inputs:
        prompt += "\n\nDependency results (use as data):\n" + json.dumps(inputs, ensure_ascii=False)
    contract = stage.contract.raw
    if item is not None:
        prompt += "\n\nCurrent foreach item (use as data):\n" + json.dumps(item["input"], ensure_ascii=False)
        # Preserve the configured trace directory while giving every item its own trace.
        trace = Path(stage.contract.trace.file)
        contract = {**contract, "trace": {**contract["trace"],
                    "file": str(trace.with_name(f"{trace.name}.{directory.name}.{name}.jsonl"))}}
    contract_path = directory / f"{name}.contract.json"
    _checkpoint(contract_path, contract)
    # Contract execution changes the process working directory. Isolate stages
    # in processes so parallel contracts never share it or any kernel globals.
    with (directory / f"{name}.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "rlmagent.cli.main", "--contract", str(contract_path),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **own_group(),
        )
        try:
            process.stdin.write(prompt.encode("utf-8"))
            process.stdin.close()
            output = bytearray()
            while chunk := await process.stdout.read(65536):
                output.extend(chunk[:max(0, HANDOFF_LIMIT - len(output))])
            status = await process.wait()
        except BaseException:
            await asyncio.to_thread(kill_tree, process)
            await process.wait()
            raise
    return {"status": "failed" if status else "done",
            "output": output.decode("utf-8", errors="ignore").strip()}


def _expand(record, source, remaining):
    value = source.get("output")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("foreach requires a JSON array")
    if len(value) > remaining:
        raise ValueError(f"foreach exceeds the run limit of {MAX_ITEMS} expanded items")
    record["items"] = [{"input": item, "status": "waiting"} for item in value]


def _settle_group(record):
    items = record["items"]
    if all(item["status"] in TERMINAL for item in items):
        record["status"] = "failed" if any(item["status"] == "failed" for item in items) else "done"
        limit = HANDOFF_LIMIT // max(1, len(items))
        record["output"] = [item.get("output", "")[:limit] for item in items]


def _topology(stages):
    return {name: {"after": list(stage.after), "when": stage.when, "foreach": stage.foreach}
            for name, stage in stages.items()}


async def execute_graph(stages, width, fingerprint, policy, directory: Path, *, retry=False):
    state_path = directory / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ContractError("the workflow changed since this run began")
        records = state["stages"]
        interrupted = {name for name, row in records.items()
                       if row["status"] in ("running", "interrupted", "failed")}
        if interrupted and not retry:
            raise ContractError("review the interrupted or failed stages and resume with --retry-stages")
        reset = set(interrupted)
        order = TopologicalSorter({name: stage.after for name, stage in stages.items()})
        for name in order.static_order():
            if any(dep in reset for dep in stages[name].after):
                reset.add(name)
        for name in reset:
            record = records[name]
            if name in interrupted and "items" in record and not any(dep in reset for dep in stages[name].after):
                for item in record["items"]:
                    if item["status"] in ("running", "interrupted", "failed"):
                        item.update(status="waiting", output="")
                record["status"] = "waiting"
                record.pop("output", None)
            else:
                records[name] = {"status": "waiting"}
    else:
        records = {name: {"status": "waiting"} for name in stages}
        state = {"id": directory.name, "fingerprint": fingerprint, "stages": records}
    state["graph"] = _topology(stages)
    state["status"] = "running"
    state.pop("stop_reason", None)
    state.setdefault("transitions", 0)
    _checkpoint(state_path, state)
    running = {}
    stop_reason = None
    workspaces = {name: Path(stage.contract.workdir).resolve() for name, stage in stages.items()}

    def conflicts(name):
        path = workspaces[name]
        return any(path == other or path in other.parents or other in path.parents
                   for other in (workspaces[node] for node, _ in running.values()))

    def can_admit(name):
        if len(running) >= width or conflicts(name):
            return False
        if policy.max_live_agents is None:
            return True
        reserved = sum(_agent_reservation(stages[node]) for node, _ in running.values())
        return reserved + _agent_reservation(stages[name]) <= policy.max_live_agents

    def charge_transition():
        if policy.max_transitions is not None and state["transitions"] >= policy.max_transitions:
            return False
        state["transitions"] += 1
        return True

    try:
        while True:
            if (directory / "cancel").exists():
                state["status"] = "cancelled"
                break
            changed = False
            for name, stage in stages.items():
                if stop_reason:
                    break
                record = records[name]
                if record["status"] not in ("waiting", "running"):
                    continue
                if record["status"] == "running" and "items" not in record:
                    continue
                deps = {dep: {key: val for key, val in records[dep].items() if key != "items"}
                        for dep in stage.after}
                statuses = [row["status"] for row in deps.values()]
                if any(status not in TERMINAL for status in statuses):
                    continue
                if ((stage.when == "all_done" and any(status != "done" for status in statuses))
                        or (stage.when == "any_failed" and "failed" not in statuses)):
                    records[name] = {"status": "skipped"}
                    changed = True
                elif stage.foreach:
                    if "items" not in record:
                        try:
                            used = sum(len(row.get("items", ())) for row in records.values())
                            _expand(record, deps[stage.foreach], MAX_ITEMS - used)
                        except ValueError as exc:
                            records[name] = {"status": "failed", "output": str(exc)}
                            if policy.on_failure == "stop":
                                stop_reason = f"stage failed: {name}"
                            changed = True
                            continue
                        _settle_group(record)
                        changed = True
                    if can_admit(name):
                        for index, item in enumerate(record["items"]):
                            if item["status"] != "waiting":
                                continue
                            if not charge_transition():
                                stop_reason = "workflow transition limit reached"
                                break
                            item["status"] = record["status"] = "running"
                            _checkpoint(state_path, state)
                            task = asyncio.create_task(_run_stage(
                                stage, deps, directory, f"{name}.item-{index}", item=item))
                            running[task] = (name, index)
                            break
                elif can_admit(name):
                    if not charge_transition():
                        stop_reason = "workflow transition limit reached"
                        break
                    records[name] = {"status": "running"}
                    _checkpoint(state_path, state)
                    running[asyncio.create_task(_run_stage(stage, deps, directory, name))] = (name, None)
            if not running:
                if stop_reason:
                    state["status"] = "failed"
                    state["stop_reason"] = stop_reason
                    for record in records.values():
                        if record["status"] == "running" and "items" in record:
                            record["status"] = "failed"
                    break
                if changed:
                    continue
                state["status"] = "failed" if any(row["status"] == "failed"
                                                  for row in records.values()) else "done"
                break
            finished, _ = await asyncio.wait(running, timeout=0.25, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                name, index = running.pop(task)
                try:
                    outcome = task.result()
                except Exception as exc:
                    outcome = {"status": "failed", "output": str(exc)[:HANDOFF_LIMIT]}
                if index is None:
                    records[name] = outcome
                else:
                    records[name]["items"][index].update(outcome)
                    _settle_group(records[name])
                if outcome["status"] == "failed" and policy.on_failure == "stop" and not stop_reason:
                    stop_reason = f"stage failed: {name}"
            if finished or changed:
                _checkpoint(state_path, state)
    finally:
        if running:
            for task, (name, index) in running.items():
                task.cancel()
                if index is not None:
                    records[name]["items"][index]["status"] = "interrupted"
                records[name]["status"] = "interrupted"
            await asyncio.gather(*running, return_exceptions=True)
            if state["status"] == "running":
                state["status"] = "interrupted"
        _checkpoint(state_path, state)
    return int(state["status"] != "done")


def run_status(directory: Path) -> dict:
    state_path = directory / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        spec = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        state = {"id": directory.name, "status": "queued", "graph": spec.get("graph", {})}
    if state["status"] == "running":
        try:
            with acquire(directory, "writer"):
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state["status"] == "running":
                    state["status"] = "interrupted"
        except SessionBusy:
            pass
    failure = directory / "failure.json"
    try:
        state["error"] = json.loads(failure.read_text(encoding="utf-8"))["error"]
    except FileNotFoundError:
        pass
    else:
        if state["status"] == "queued":
            state["status"] = "failed"
    return state


def command(action: str, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"rlm-agent factory {action}")
    parser.add_argument("manifest" if action == "launch" else "run_id")
    if action in ("resume", "_execute"):
        parser.add_argument("--retry-stages", action="store_true")
    args = parser.parse_args(argv)
    run_id = uuid4().hex if action == "launch" else args.run_id
    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise ContractError("invalid factory run ID")
    directory = rlm_agent_home().resolve() / "factory-runs" / run_id
    spec = directory / "run.json"
    if action == "launch":
        manifest = Path(args.manifest).resolve(strict=True)
        stages, _, fingerprint, _ = load_graph(manifest)
        for stage in stages.values():
            stage.contract.api_key()
        directory.mkdir(parents=True, exist_ok=False)
        _checkpoint(spec, {"manifest": str(manifest), "cwd": os.getcwd(), "graph": _topology(stages),
                           "fingerprint": fingerprint})
    else:
        if not spec.is_file():
            raise ContractError(f"unknown factory run {run_id}")
    if action == "status":
        print(json.dumps(run_status(directory)))
        return 0
    if action == "view":
        from rlmagent.tui.factory_view import FactoryGraphApp

        FactoryGraphApp(directory).run()
        return 0
    if action == "cancel":
        (directory / "cancel").touch()
        return 0
    config = json.loads(spec.read_text(encoding="utf-8"))
    if action == "_execute":
        with acquire(directory, "writer"):
            failure = directory / "failure.json"
            failure.unlink(missing_ok=True)
            try:
                stages, width, fingerprint, policy = load_graph(Path(config["manifest"]))
                if config.get("fingerprint", fingerprint) != fingerprint:
                    raise ContractError("the workflow changed since this run was launched")
                return asyncio.run(execute_graph(stages, width, fingerprint, policy, directory,
                                                 retry=args.retry_stages))
            except (OSError, RuntimeError, ValueError) as exc:
                _checkpoint(failure, {"error": str(exc)})
                raise
    if action == "resume":
        with acquire(directory, "writer"):
            (directory / "cancel").unlink(missing_ok=True)
    retry = ["--retry-stages"] if getattr(args, "retry_stages", False) else []
    options = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {
        "start_new_session": True,
    }
    with (directory / "output.log").open("ab") as output:
        subprocess.Popen(
            [sys.executable, "-m", "rlmagent.cli.main", "factory", "_execute", run_id, *retry],
            cwd=config["cwd"], stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            **options,
        )
    print(run_id)
    return 0
