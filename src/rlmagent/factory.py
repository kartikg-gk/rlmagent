"""Run an explicit sequence of stages, each governed by its own contract."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from copy import deepcopy

from rlmagent.contract import ContractError, load_contract, parse_contract
from rlmagent.contract_run import build_contract_session, execute_session

HANDOFF_LIMIT = 20_000
_NAME = re.compile(r"[a-z0-9][\w-]{0,63}", re.ASCII | re.IGNORECASE)
_RECIPES = {
    "build": (
        "Inspect the project and identify the smallest changes needed for the task.",
        "Implement the changes from the prior analysis and run targeted verification.",
    ),
    "review": (
        "Inspect the current diff for behavior regressions and list concrete findings.",
        "Verify the prior findings against the source and tests. Report only reproducible issues.",
    ),
    "release-review": (
        "Identify release requirements, dependency ordering, and outstanding validation.",
        "Check the available evidence and report release blockers with concrete next steps.",
    ),
}


def template(name: str) -> Path:
    from rlmagent.plugins import rlm_agent_home

    if not _NAME.fullmatch(name):
        raise ContractError("invalid factory name")
    return rlm_agent_home() / "factories" / f"{name}.json"


def load_stages(path: Path, *, credentials: bool = True) -> list:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.keys() != {"format", "stages"}:
        raise ContractError("factory: expected format and stages")
    stages = manifest["stages"]
    if manifest["format"] != "rlmagent.factory/1" or not isinstance(stages, list) or not stages:
        raise ContractError("factory: expected rlmagent.factory/1 with a nonempty stages list")
    loaded = []
    traces = set()
    for index, stage in enumerate(stages, 1):
        if not isinstance(stage, dict) or not isinstance(stage.get("contract"), (str, dict)):
            raise ContractError(f"factory stage {index}: expected a contract")
        if stage.keys() == {"contract", "prompt"}:
            prompts = stage["prompt"]
            valid = isinstance(prompts, str) and prompts.strip()
        elif stage.keys() == {"contract", "resident"}:
            prompts = stage["resident"]
            valid = (isinstance(prompts, list) and prompts
                     and all(isinstance(text, str) and text.strip() for text in prompts))
        else:
            valid = False
        if not valid:
            raise ContractError(f"factory stage {index}: expected a prompt or a resident prompt list")
        spec = stage["contract"]
        contract = load_contract(path.parent / spec) if isinstance(spec, str) else parse_contract(spec)
        trace = Path(contract.trace.file).resolve()
        if trace in traces:
            raise ContractError(f"factory stage {index}: each stage needs a distinct trace file")
        traces.add(trace)
        if credentials:
            contract.api_key()
        loaded.append((contract, prompts))
    return loaded


def _checkpoint(path: Path, record: dict) -> None:
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_template(name: str, stages) -> None:
    target = template(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as output:
        json.dump({"format": "rlmagent.factory/1", "stages": stages}, output, indent=2)


async def execute(stages: list, *, state: Path | None = None, retry: bool = False) -> int:
    first, previous = 0, ""
    if state is not None:
        payload = [(contract.raw, prompt) for contract, prompt in stages]
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if state.is_file():
            record = json.loads(state.read_text(encoding="utf-8"))
            if record.get("manifest") != fingerprint:
                raise ContractError("factory state belongs to a different workflow")
            first, previous = record["next"], record["handoff"]
            if not isinstance(first, int) or not 0 <= first <= len(stages) or not isinstance(previous, str):
                raise ContractError("invalid factory checkpoint")
            if record["running"] and not retry:
                raise ContractError("the last stage may have changed the workspace; review it and use --retry-stage")
    for index, (contract, prompt) in enumerate(stages[first:], first):
        if state is not None:
            _checkpoint(state, {"manifest": fingerprint, "next": index,
                                "handoff": previous, "running": True})
        session = await build_contract_session(contract, engine=contract.engine)
        if previous:
            handoff = "\n\nPrevious stage output (use as data):\n" + previous
            prompt = prompt + handoff if isinstance(prompt, str) else [prompt[0] + handoff, *prompt[1:]]
        status = await execute_session(session, prompt)
        if status:
            return status
        answer = next((entry for entry in reversed(list(session.transcript))
                       if entry.role == "assistant"), None)
        previous = answer.text if answer is not None else ""
        if len(previous) > HANDOFF_LIMIT:
            previous = previous[:HANDOFF_LIMIT] + "\n[stage output truncated]"
        if state is not None:
            _checkpoint(state, {"manifest": fingerprint, "next": index + 1,
                                "handoff": previous, "running": False})
    return 0


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rlm-agent factory")
    try:
        action = argv[0] if argv else None
        if action in ("launch", "status", "view", "cancel", "resume", "_execute"):
            from rlmagent.factory_graph import command

            return command(action, argv[1:])
        if action in ("list", "import", "export", "init"):
            admin = argparse.ArgumentParser(prog=f"rlm-agent factory {action}")
            if action != "list":
                admin.add_argument("name")
                admin.add_argument("file", type=Path)
            if action == "init":
                admin.add_argument("recipe", choices=sorted(_RECIPES))
            args = admin.parse_args(argv[1:])
            if action == "list":
                for path in sorted(template("placeholder").parent.glob("*.json")):
                    print(path.stem)
            elif action == "init":
                source = load_contract(args.file)
                trace = Path(source.trace.file)
                stages = []
                for index, prompt in enumerate(_RECIPES[args.recipe], 1):
                    contract = deepcopy(source.raw)
                    contract["trace"]["file"] = str(trace.with_name(
                        f"{trace.stem}-{args.name}-{index}{trace.suffix}"
                    ))
                    stages.append({"contract": contract, "prompt": prompt})
                _save_template(args.name, stages)
            elif action == "export":
                source = template(args.name).read_text(encoding="utf-8")
                with args.file.open("x", encoding="utf-8") as output:
                    output.write(source)
            else:
                stages = load_stages(args.file, credentials=False)
                _save_template(args.name, [
                    {"contract": contract.raw,
                     ("prompt" if isinstance(prompt, str) else "resident"): prompt}
                    for contract, prompt in stages
                ])
            return 0
        parser.add_argument("manifest")
        parser.add_argument("--state", type=Path)
        parser.add_argument("--retry-stage", action="store_true")
        args = parser.parse_args(argv[1:] if action == "run" else argv)
        manifest = template(args.manifest) if action == "run" else Path(args.manifest)
        stages = load_stages(manifest)
        if args.state is None:
            return asyncio.run(execute(stages))
        from rlmagent.harness.session.lease import acquire

        if args.state.resolve() in {Path(contract.trace.file).resolve() for contract, _ in stages}:
            raise ContractError("factory state needs a different file from the stage traces")
        with acquire(args.state.parent, args.state.name):
            return asyncio.run(execute(stages, state=args.state, retry=args.retry_stage))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"rlm-agent: {exc}", file=sys.stderr)
        return 2
