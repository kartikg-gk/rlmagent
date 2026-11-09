"""Adapt the backend's recursive solver to the application's model providers."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import queue
import re
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import time

from rlmness import Config, solve
from rlmness.providers import Spend
from rlmness.session import Session
from rlmness.wasm_runtime import WasmRuntime

from rlmagent.agents.workspace import Workspace
from rlmagent.harness.contracts.transcript import (
    HumanEntry,
    ImageSegment,
    ModelEntry,
    TextSegment,
    UsageStats,
)
from rlmagent.harness.provider.wire import ContentChunkEvent, StreamCloseEvent, StreamFaultEvent
from rlmagent.kept_agents import KeptAgents


def wait_for_host(future, agent, background):
    while True:
        key = background.owners.get(agent) if background is not None else None
        if key is not None and background.jobs[key].token.is_set():
            future.cancel()
            from rlmness.limits import Abandoned

            raise Abandoned("background agent cancelled")
        try:
            return future.result(timeout=0.1)
        except FutureTimeout:
            if future.done():
                raise


_STATE_FIELDS = ("answered", "asking", "cells", "variables", "functions", "modules", "dropped")
MAX_CHECKPOINT_CHARS = 20_000_000


def limits_of(tree) -> dict:
    """The settings in force, read from where they are enforced."""
    allowance = tree.allowance
    return {
        "approval": tree.approver.policy,
        "network": tree.network,
        "kernel": getattr(tree, "kernel_kind", None),
        "max_depth": allowance.max_depth,
        "calls_used": allowance.calls,
        "max_calls": allowance.max_calls,
        "cost_used": allowance.cost,
        "max_cost": allowance.max_cost,
    }


SHOWN_OUTPUT_LINES = 8


def summed_usage(accounting):
    return UsageStats(**{name: sum(getattr(item, name) or 0 for item in accounting) for name in
        ("input", "output", "total_tokens", "cache_read", "cache_write", "cache_write_1h", "reasoning")},
        cost={name: sum(getattr(item.cost, name) for item in accounting) for name in
              ("total", "input", "output", "cache_read", "cache_write")})


def shown_output(output: str) -> str:
    """A cell's output for the screen: the start of it. The model still gets all of it."""
    lines = output.splitlines()
    if len(lines) <= SHOWN_OUTPUT_LINES:
        return output
    return "\n".join(lines[:SHOWN_OUTPUT_LINES]) + f"\n... ({len(lines) - SHOWN_OUTPUT_LINES} more lines)"


#: The engine's own first cells (its helpers, then the context): replayed after a restart.
SETUP_CELLS = 2
MAX_RESTARTS = 3

RESTARTED = ("The kernel stopped (it crashed or was killed) and has been started again. "
             "Variables, imports and functions from this run are gone; set up again what "
             "you need. Files on disk and background jobs are unaffected.")


def shape_cells(runtime, policy, rebuild=None) -> None:
    """Cap what a cell shows the model, add a hint when it failed in a familiar way, and
    start a fresh kernel when this one dies, rather than ending the run."""
    from rlmness.runtime import RuntimeGone

    from rlmagent.hints import hint_for
    from rlmagent.policy import DEFAULT_OUTPUT_BYTES, cap_output

    limit = getattr(policy, "max_tool_output_bytes", DEFAULT_OUTPUT_BYTES)
    setup: list[str] = []

    def run(code):
        return type(runtime).execute(runtime, code)

    def restart():
        fresh = rebuild()
        # The engine holds this object, so the new kernel moves into it.
        runtime.__dict__.clear()
        runtime.__dict__.update(fresh.__dict__)
        runtime.execute = execute
        for cell in setup:
            run(cell)

    def execute(code):
        if len(setup) < SETUP_CELLS:
            setup.append(code)
        try:
            outcome = run(code)
        except RuntimeGone:
            restarts = getattr(runtime, "restarts", 0)
            if rebuild is None or restarts >= MAX_RESTARTS:
                raise
            restart()
            runtime.restarts = restarts + 1
            from rlmness.runtime import CellOutcome

            return CellOutcome(error=RESTARTED)
        outcome.stdout = cap_output(outcome.stdout or "", limit)
        hint = hint_for(code, (outcome.stdout or "") + "\n" + (outcome.error or ""))
        if hint and outcome.error:
            outcome.error = f"{outcome.error}\nHint: {hint}"
        elif hint and code is not None and not code.strip():
            outcome.stdout = f"{outcome.stdout}\nHint: {hint}".lstrip()
        return outcome

    runtime.execute = execute


class Client:
    def __init__(
        self,
        provider,
        loop,
        signal,
        pending=None,
        tree=None,
        context=None,
        instructions="",
        background=None,
        accounting=None,
        record=None,
        images=(),
    ):
        self.provider, self.loop, self.signal = provider, loop, signal
        self.pending = pending if pending is not None else set()
        self.tree, self.context = tree, context
        self.instructions = instructions
        self.background = background
        self.accounting, self.record = accounting, record
        self.images = images
        self.host_context = contextvars.copy_context()

    def complete(self, messages, *, model):
        agent = getattr(self.context, "agent", "root")
        contract_model = getattr(self.tree, "contract_model", None)
        if contract_model is not None and model != contract_model:
            raise RuntimeError("model requests must use the model specified by the runtime contract")

        async def request():
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            system += "\n\n" + self.instructions
            refinement = getattr(self.tree, "refinement", None)
            if refinement is not None:
                system = refinement.filter_system(system)
            entries = [
                (
                    ModelEntry(content=[TextSegment(text=m["content"])])
                    if m["role"] == "assistant"
                    else HumanEntry(content=m["content"])
                )
                for m in messages
                if m["role"] != "system"
            ]
            if agent == "root" and self.images:
                for index, entry in enumerate(entries):
                    if isinstance(entry, HumanEntry):
                        entries[index] = HumanEntry(
                            content=[TextSegment(text=entry.text), *self.images]
                        )
                        break
            tracer = getattr(self.tree, "tracer", None)
            policy = getattr(self.tree, "policy", None)
            limit = getattr(policy, "agent_max_tokens", None)
            if limit is not None and tracer is not None:
                spent = tracer._totals.get(agent, {}).get("output", 0)
                if spent >= limit:
                    raise RuntimeError(f"agent {agent} reached its limit of {limit} "
                                       "completion tokens")
            child = self.background.owners.get(agent) if self.background is not None else None
            call = tracer.begin(agent, model, child=child) if tracer is not None else None
            if tracer is not None:
                def close_unfinished(task):
                    if call in tracer._open:
                        tracer.end(call, None, "aborted" if task.cancelled() else "error")
                asyncio.current_task().add_done_callback(close_unfinished)
            session = getattr(self.tree, "session", None)
            ledger = getattr(session, "steering", None)
            pending_steering = []
            if ledger is not None:
                metadata = getattr(self.tree, "backend_agents", {}).get(agent, {})
                text, pending_steering = ledger.context(
                    agent, root=child is None and metadata.get("depth", 0) == 0, child=child,
                )
                if text:
                    entries.append(HumanEntry(content="Steering instructions:\n" + text))
            from rlmagent.agents.trace import correlated
            from rlmagent.model.deadline import bounded_stream

            def start():
                return self.provider.stream_response(
                    model=model, system=system, messages=entries, tools=[], signal=self.signal,
                    cache_key=(str(getattr(session, "session_id", "run")) + ":" + (
                        "root" if getattr(self.tree, "backend_agents", {}).get(agent, {}).get("depth", 0) == 0 else agent)),
                )

            budget = getattr(getattr(self.tree, "allowance", None), "new_token_budget", None)
            stream = self.tree.provider.measured_stream(model, start) if budget is not None else start()
            async for event in correlated(bounded_stream(stream, model,
                    getattr(self.tree, "request_timeout", 300.0)), call, tracer):
                if isinstance(event, StreamFaultEvent):
                    if tracer is not None:
                        tracer.end(call, event.error.usage, event.reason)
                    raise RuntimeError(event.error.error_message or "model request failed")
                if isinstance(event, StreamCloseEvent):
                    if ledger is not None:
                        ledger.consumed(pending_steering, agent, call, child=child)
                        if pending_steering and tracer is not None:
                            tracer._write("steering", agent=agent, request=call,
                                          messages=pending_steering)
                    usage = event.message.usage
                    if self.accounting is not None:
                        self.accounting.append(usage)
                    if self.record is not None:
                        self.record(
                            {
                                "kind": "model_usage",
                                "request_id": call,
                                "run_id": agent,
                                "model": model,
                                "usage": usage.model_dump(by_alias=False),
                                "text": event.message.text,
                                "reason": event.reason,
                            }
                        )
                    if tracer is not None:
                        tracer.end(call, usage, event.reason)
                    return event.message.text, Spend(
                        prompt_tokens=usage.input + usage.cache_read + usage.cache_write,
                        completion_tokens=usage.output,
                        total_tokens=usage.total_tokens,
                        cost=usage.cost.total if usage.cost.total > 0 else None,
                        cached_tokens=usage.cache_read,
                        reasoning_tokens=usage.reasoning,
                    )
            if tracer is not None:
                tracer.end(call, None, "no_completion")
            raise RuntimeError("model returned no completion")

        future = self.host_context.copy().run(
            asyncio.run_coroutine_threadsafe, request(), self.loop
        )
        self.pending.add(future)
        try:
            return wait_for_host(future, agent, self.background)
        finally:
            self.pending.discard(future)


class BackendProvider:
    """A complete recursive backend run appears as one frontend completion."""

    is_backend = True

    def __init__(self, provider, fallback, tree):
        self.provider, self.fallback, self.tree = provider, fallback, tree
        self.follows_tree = True
        self.active = None
        self.background = KeptAgents(getattr(tree, "max_agents", 50))
        self.background.resolve_role = getattr(tree, "_role", None)
        self.pending = set()
        self.state = Session()
        self.instructions = ""
        self.prompt_context = {}
        self.events = []
        self._serial = asyncio.Lock()
        self.event_path = None
        self._event_lock = threading.Lock()
        # Called with each run event, from the engine's thread. A screen that
        # draws steps itself sets cards, and the reply then carries only text.
        self.listeners = []
        self.cards = False
        self.background.record_event = self.record_event

    def record_event(self, event):
        with self._event_lock:
            event = {"timestamp": time(), **event}
            self.events.append(event)
            for listener in list(self.listeners):
                listener(event)
            if self.event_path is not None:
                path = Path(self.event_path)
                path.parent.mkdir(parents=True, exist_ok=True)

                def encode(value):
                    return asdict(value) if is_dataclass(value) else str(value)

                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, default=encode, ensure_ascii=False) + "\n")
                from rlmagent.execution_log import records

                compatible = path.with_name(path.stem + ".run.jsonl")
                for row in records(event, self.events):
                    with compatible.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, default=encode, ensure_ascii=False) + "\n")

    def __getattr__(self, name):
        return getattr(self.fallback, name)

    def function_names(self) -> list[str]:
        """The names a cell can call, without starting a run."""
        fixed = {"FINAL", "capabilities", "rlm", "gather_rlm", "spawn", "children", "tell_parent",
                 "name_child", "child", "child_info", "steer_child"}
        optional = {"fetch_url", "guidance", "improve", "playbook"}
        workspace = Workspace(self.tree.cwd, self.tree.approver).functions()
        extended = getattr(self.tree, "extension_functions", {})
        builtin = getattr(self.tree, "host_functions", {})
        optional |= set(workspace) | set(extended) | set(builtin)
        policy = getattr(self.tree, "policy", None)
        if policy is not None:
            optional = {name for name in optional if policy.allows_tool(name)}
        if "playbook" in optional:
            optional.add("skill")
        return sorted(fixed | optional)

    def checkpoint_state(self):
        """A copy of the saved variables for /undo, or None when they are too large to keep."""
        fields = {name: getattr(self.state, name) for name in _STATE_FIELDS}
        if len(json.dumps(fields, default=str)) > MAX_CHECKPOINT_CHARS:
            return None
        return copy.deepcopy(fields)

    def restore_state(self, checkpoint) -> None:
        self.state.close_runtime()
        for name, value in copy.deepcopy(checkpoint).items():
            setattr(self.state, name, value)
        if self.state.path is not None:
            self.state.save()

    def stream_response(self, *, model, system, messages, tools, signal=None, cache_key=None):
        if not any(tool.name == "python" for tool in tools):
            return self.fallback.stream_response(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                signal=signal,
                cache_key=cache_key,
            )
        return self._run(model, messages, signal)

    async def _run(self, model, messages, signal):
        from rlmagent.model.external.textcalls import render

        latest = next((entry for entry in reversed(messages) if isinstance(entry, HumanEntry)), None)
        images = (
            [part for part in latest.content if isinstance(part, ImageSegment)]
            if latest is not None and isinstance(latest.content, list) else []
        )
        async with self._serial:
            self.last_stop_reason = None
            self.background.steering = getattr(getattr(self.tree, "session", None), "steering", None)
            self.background.open(self.state.path)
            loop = asyncio.get_running_loop()
            host_context = contextvars.copy_context()
            tree = self.tree
            from rlmagent.model.external import with_external

            if hasattr(tree, "provider"):
                self.provider = with_external(tree.provider.inner)
            workspace = Workspace(tree.cwd, tree.approver)
            workspace.policy = getattr(tree, "policy", None)
            workspace.on_output = lambda text: updates.put({"kind": "command_output", "text": text})
            updates = queue.Queue()
            context = threading.local()
            accounting = []
            runtimes = []
            self.active = (None, runtimes)

            owner = self

            class Sink:
                def __getattr__(self, name):
                    def record(**fields):
                        if name in ("step_started", "run_started"):
                            context.agent = fields["run_id"]
                        event = {"kind": name, **fields}
                        owner.record_event(event)
                        updates.put(event)
                        if name == "run_started":
                            if fields.get("depth", 0) > 0:
                                count = sum(
                                    1
                                    for item in owner.events
                                    if item["kind"] == "run_started" and item.get("depth", 0) > 0
                                )
                                if count > getattr(tree, "max_agents", 50):
                                    raise RuntimeError("recursive agent limit reached")
                            tree.__dict__.setdefault("backend_agents", {})[fields["run_id"]] = dict(
                                fields, status="running"
                            )
                            if fields.get("parent_run_id") and hasattr(tree, "tracer"):
                                loop.call_soon_threadsafe(
                                    tree.tracer.spawned, fields["parent_run_id"], fields["run_id"]
                                )
                        elif name in ("run_completed", "run_failed"):
                            agent = tree.__dict__.get("backend_agents", {}).get(fields["run_id"])
                            if agent is not None:
                                agent["status"] = (
                                    "completed" if name == "run_completed" else "failed"
                                )
                                if agent.get("parent_run_id") and hasattr(tree, "tracer"):
                                    loop.call_soon_threadsafe(
                                        tree.tracer.returned,
                                        agent["parent_run_id"],
                                        fields["run_id"],
                                    )

                    return record

            def factory(prompt, bridges, timeout, tools=(), **kwargs):
                agent = getattr(context, "agent", "root")
                reuse = kwargs.pop("reuse", None)
                owner_session = kwargs.pop("owner_session", None)
                node_prepare = kwargs.pop("_node_prepare", False)
                if child_id := self.background.owners.get(agent):
                    self.record_event({"kind": "child_bound", "run_id": agent,
                                       "child_id": child_id,
                                       "name": self.background.names.get(child_id)})

                def wrap(fn):
                    def call(*args, **kw):
                        async def invoke():
                            from rlmagent.agents.approval import Refused

                            try:
                                value = await fn(*args, **kw)
                            except (Refused, OSError, ValueError, UnicodeError) as error:
                                value = f"Error: {error}"
                            self.record_event(
                                {
                                    "kind": "host_result",
                                    "run_id": agent,
                                    "tool": fn.__name__,
                                    "result": value,
                                }
                            )
                            return value

                        future = host_context.copy().run(
                            asyncio.run_coroutine_threadsafe, invoke(), loop
                        )
                        self.pending.add(future)
                        try:
                            return wait_for_host(future, agent, self.background)
                        finally:
                            self.pending.discard(future)

                    return call

                from rlmagent.plugins import as_host_function

                joined = {
                    **{name: wrap(as_host_function(fn))
                       for name, fn in getattr(tree, "extension_functions", {}).items()},
                    **bridges,
                    **{name: wrap(fn) for name, fn in workspace.functions().items()},
                    **{name: wrap(fn)
                       for name, fn in getattr(tree, "host_functions", {}).items()},
                }
                remote = getattr(tree, "remote_tools", None)
                if remote is not None:
                    # Split execution: files, commands and jobs live in the tool host.
                    joined.update({name: wrap(fn) for name, fn in remote.functions(
                        tree.approver, getattr(tree, "policy", None)).items()})

                async def playbook(id=None):
                    improver = getattr(tree, "improver", None)
                    if improver is None:
                        raise RuntimeError("playbook unavailable in this run")
                    if id is not None:
                        return improver.entry(id)
                    return [
                        {"id": e.id, "kind": e.kind, "title": e.title} for e in improver.entries()
                    ]

                async def improve(instructions=None, shared=False):
                    if not getattr(tree, "playbook_writable", True):
                        raise RuntimeError("contract playbook is read-only")
                    improver = getattr(tree, "improver", None)
                    if improver is None:
                        raise RuntimeError("playbook unavailable in this run")
                    improver.request(instructions, shared)
                    return "Playbook update queued for the turn boundary."

                async def fetch_url(url, max_chars=50000, raw=False):
                    from rlmagent.agents.approval import Refused
                    from rlmagent.network import fetch_text

                    if not tree.network:
                        raise Refused("network access is disabled")
                    return await fetch_text(
                        url, max_chars=max_chars, raw=raw,
                        allow_private=getattr(tree, "allow_private_fetch", False))

                async def guidance(id=None):
                    refinement = getattr(tree, "refinement", None)
                    return refinement.read_guidance(id) if refinement is not None else []

                async def capabilities():
                    names = {name for name in joined if not name.startswith("_")}
                    if "_spawn" in joined:
                        names |= {"spawn", "children", "name_child", "child", "child_info", "steer_child"}
                    return {
                        "functions": sorted(names | {tool.name for tool in tools} | {"FINAL"}),
                        **limits_of(tree),
                    }

                joined["guidance"] = wrap(guidance)
                joined["capabilities"] = wrap(capabilities)
                joined.update(
                    playbook=wrap(playbook), improve=wrap(improve), fetch_url=wrap(fetch_url)
                )
                policy = getattr(tree, "policy", None)
                if policy is not None:
                    joined = {name: fn for name, fn in joined.items()
                              if name in bridges or name == "capabilities" or policy.allows_tool(name)}
                startup = (
                    "async def skill(id):\n"
                    "    entry = await playbook(id)\n"
                    "    namespace = {}\n"
                    "    exec(entry['content'], namespace)\n"
                    "    return namespace[entry['callable']]"
                ) if "playbook" in joined else ""
                if "_delivery" in joined:
                    startup += ("\nasync def child_delivery(handle, receipt):\n"
                                "    return await _delivery(handle.id, receipt)\n")
                if "_spawn" in bridges:
                    from rlmagent.agents.child_handles import HELPERS

                    async def child_registry(action, key=None, name=None, text=None, message_id=None):
                        background = self.background
                        parent = background.owners.get(agent, agent)
                        if not node_prepare and parent not in background.saved:
                            parent = agent
                        # Node launches the job before returning the ID, but its first
                        # metadata callback can arrive just after the spawn response.
                        for attempt in range(40):
                            try:
                                with background.lock:
                                    operation = "info" if action == "steer" else action
                                    info = background.child_handle(parent, operation, key, name)
                                    ledger = background.steering
                                    if action == "steer":
                                        if ledger is None:
                                            raise ValueError("child steering requires an attached session")
                                        active = info["status"] in ("starting", "running")
                                        if not active:
                                            ledger.expire_child(key)
                                        return ledger.accept(text, message_id, key, once=True,
                                                             allow_new=active)
                                    if action == "info" and ledger is not None:
                                        if info["status"] not in ("starting", "running"):
                                            ledger.expire_child(key)
                                        info["steering"] = ledger.receipts(target=key)
                                break
                            except KeyError:
                                if not node_prepare or attempt == 39:
                                    raise ValueError("unknown child handle") from None
                                await asyncio.sleep(0.025)
                        if action == "info":
                            tracer = getattr(tree, "tracer", None)
                            info["usage"] = tracer.child_usage(key) if tracer is not None else None
                            info.setdefault("steering", [])
                        return info

                    joined["_child_registry"] = wrap(child_registry)
                    startup += (
                        "\nasync def _child_operation(action, key=None, name=None, **arguments):\n"
                        "    result = await _child_registry(action, key, name, **arguments)\n"
                        "    if isinstance(result, str):\n"
                        "        raise ValueError(result)\n"
                        "    return result\n"
                        "def _child_handle(key):\n"
                        "    return _AgentHandle(key)\n"
                    ) + HELPERS
                if node_prepare:
                    from rlmagent.hints import EMPTY_CELL, _RULES
                    from rlmagent.policy import DEFAULT_OUTPUT_BYTES

                    if tree.network:
                        from rlmagent.agents.wasm_api import NETWORK
                        startup += "\n" + NETWORK
                    return {
                        "host_functions": {key: value for key, value in joined.items() if value is not None},
                        "startup": startup,
                        "cell_policy": {
                            "maxOutputBytes": getattr(workspace.policy, "max_tool_output_bytes",
                                                      DEFAULT_OUTPUT_BYTES),
                            "emptyHint": EMPTY_CELL,
                            "hints": [[pattern.pattern, bool(pattern.flags & re.IGNORECASE), hint]
                                      for pattern, hint in _RULES],
                        },
                    }
                from rlmness.runtime import SubprocessRuntime

                runtime_type = (
                    WasmRuntime
                    if getattr(tree, "kernel_kind", "wasm") == "wasm"
                    else SubprocessRuntime
                )
                # Host startup and helper installation have their own bounded window.
                # The requested cell deadline still applies to every model-written cell.
                def build():
                    if owner_session is not None:
                        kwargs["session"] = owner_session.state_for_guest()
                    built = runtime_type(prompt, joined, max(30.0, timeout), tools, **kwargs)
                    built.execute(startup)
                    if tree.network and isinstance(built, WasmRuntime):
                        from rlmagent.agents.wasm_api import NETWORK

                        outcome = built.execute(NETWORK)
                        if outcome.error:
                            built.close()
                            raise RuntimeError(outcome.error)
                    built.timeout = timeout
                    return built

                if reuse is not None and isinstance(reuse, runtime_type) and not reuse._closed:
                    reuse.rebind(prompt, joined, max(30.0, timeout), tools)
                    reuse.timeout = timeout
                    runtime = reuse
                else:
                    if reuse is not None:
                        reuse.close()
                    runtime = build()
                runtimes.append(runtime)
                shape_cells(runtime, getattr(tree, "policy", None), rebuild=build)
                return runtime

            factory.SEALED = True
            factory.PERSISTENT = True
            factory.NEEDS_SOURCE = True
            factory.MAX_LIVE = WasmRuntime.MAX_LIVE
            cancel = threading.Event()
            self.active = (cancel, runtimes)
            from rlmness.background import PRELUDE

            self.state.functions["_AgentHandle"] = PRELUDE.split("async def spawn(")[0]
            def prepare_node(prompt, bridge_names, timeout, *, session, agent):
                context.agent = agent
                return factory(prompt, dict.fromkeys(bridge_names), timeout,
                               owner_session=session, _node_prepare=True)

            factory.NODE_PREPARE = prepare_node
            config = Config(
                engine=getattr(tree, "recursive_engine",
                               "node" if getattr(tree, "kernel_kind", "wasm") == "wasm" else "python"),
                primary_agent=model,
                max_steps=tree.allowance.max_calls,
                max_calls=tree.allowance.max_calls,
                max_cost=tree.allowance.max_cost,
                max_depth=tree.allowance.max_depth,
                max_live=tree.allowance.max_live,
                timeout=tree.cell_timeout,
                inherit_tools=True,
            )
            instruction = (
                self.instructions
                + "\n\n"
                + (
                    "Complete the user's coding task. Workspace access uses async read_file(path, start=None, end=None) "
                    "(line numbers from 1, both ends included), list_files(pattern), "
                    "search_files(regex, path='.', glob=None, ignore_case=False) for matching lines, write_file(path, content), edit_file(path, old, new), "
                    "apply_patch(unified_diff) for several edits or files, all or none, "
                    "and run(cmd, timeout=120). Changes and commands use host approval. "
                    "await ask_user(question, choices=None, multiple=False) asks the user; use it only "
                    "when you truly cannot decide. "
                    "Failed or refused tool calls return text starting with Error:; inspect results "
                    "before continuing. "
                    "await fetch_url(url, max_chars=50000) fetches bounded text "
                    "when network is enabled. "
                    "await instruction_docs() discovers project instruction and skill files; "
                    "await instruction_docs(path) reads a selected file. "
                    "Read relevant instructions before editing. "
                    "await capabilities() reports the functions and limits you actually have; "
                    "check it rather than answering from memory about what is available. "
                    "Use the backend's rlm(piece, instruction=question) and gather_rlm APIs. "
                    "Background jobs use await spawn(task, data, keep=False, "
                    "role=None, model=None); "
                    "the returned handle provides result(timeout=None), status(), send(text), "
                    "messages(), cancel(), and id. Persistent jobs use keep=True. "
                    "Children can await tell_parent(text). await children() lists handles. "
                    "await name_child(handle, name) names a child, await child(name) finds it, "
                    "and await child_info(handle) shows status and recorded usage. "
                    "Use await steer_child(handle, text) to guide an active child's next model request. "
                    "Playbooks use await playbook(id=None), await skill(id), and "
                    "await improve(instructions, shared=False). "
                    "For work with several parts keep a task list: await todo('add', text), "
                    "await todo('done', n), await todo() to see it; the user sees it too. "
                    "await mcp_tools() lists tools on configured MCP servers and "
                    "await mcp(server, tool, arguments) calls one. "
                    "await factory_run(name) runs a named workflow from the user's library "
                    "when workspace approval allows it. "
                    "await web_search(query) searches the web when a search key is set. "
                    "await context_usage() reports how full the context is; await compact() "
                    "summarizes older conversation when this turn ends. "
                    "Other agents: await list_agents(), await send_message(to, text), "
                    "await read_messages(); await schedule(task, in_seconds=..., "
                    "every=None, worker=None) queues work on a background worker; "
                    "await heartbeat(text) reports progress on long work. "
                    "Long commands: await job_start(cmd) then job_read(id, cursor), "
                    "job_wait(id), job_stop(id); await service_start(name, cmd, ready=regex) "
                    "for servers; await watch(job=id | path=p | messages=True) waits for a "
                    "change. await history(start, end, agent=None) reads the full record "
                    "after compaction. "
                    "When the user set a goal, await goal() shows it and "
                    "await goal_done(summary) marks it achieved. "
                    "Return the answer with FINAL. The conversation is in PROMPT."
                )
            )
            policy = getattr(tree, "policy", None)
            if policy is not None and policy.allowed_tools is not None:
                instruction = (
                    self.instructions + "\n\nComplete the task in Python and return the answer with FINAL. "
                    "The conversation is in PROMPT. Use await capabilities() to inspect the permitted "
                    "host functions. Delegation follows the configured depth and inherits the same policy."
                )
            task = asyncio.create_task(
                asyncio.to_thread(
                    solve,
                    {**self.prompt_context, "conversation": render(messages)},
                    Client(
                        self.provider,
                        loop,
                        signal,
                        self.pending,
                        tree,
                        context,
                        instruction,
                        self.background,
                        accounting,
                        self.record_event,
                        images=images,
                    ),
                    config=config,
                    instruction=instruction,
                    runtime_factory=factory,
                    trace=Sink(),
                    allowance=tree.allowance,
                    cancel=cancel,
                    session=self.state,
                    background=self.background,
                    run_id="root",
                )
            )
            try:
                progress = ""
                while not task.done() or not updates.empty():
                    while not updates.empty():
                        event = updates.get_nowait()
                        name = event["kind"]
                        text = ""
                        if self.cards:
                            pass
                        elif (
                            name == "code_generated"
                            and event.get("step", 0) > 0
                            and event.get("code")
                        ):
                            text = f"\n```python\n{event['code']}\n```\n"
                        elif (
                            name == "output_received"
                            and event.get("step", 0) > 0
                            and event.get("output")
                        ):
                            text = f"\n```text\n{shown_output(str(event['output']))}\n```\n"
                        elif name == "command_output":
                            text = f"\n```text\n{event['text'].rstrip()}\n```\n"
                        elif name in ("run_started", "run_completed", "run_failed"):
                            text = f"[{name}: {event.get('run_id', '')}]\n"
                        if text:
                            progress += text
                            yield ContentChunkEvent(
                                content_index=0,
                                delta=text,
                                partial=ModelEntry(
                                    model=model, content=[TextSegment(text=progress)]
                                ),
                            )
                    if signal is not None and signal.is_cancelled():
                        self.cancel()
                    await asyncio.wait({task}, timeout=0.1)
                answer = task.result()
            except asyncio.CancelledError:
                self.cancel()
                self.active = None
                if asyncio.current_task().cancelling():
                    raise
                yield StreamFaultEvent(
                    reason="aborted",
                    error=ModelEntry(model=model, stop_reason="aborted", error_message="cancelled",
                                     usage=summed_usage(accounting)),
                )
                return
            except Exception as error:
                from rlmness import AllowanceSpent
                from rlmness.engine import StepsUsedUp
                from rlmness.runtime import CellTimeout, RuntimeGone

                if isinstance(error, AllowanceSpent):
                    tree.provider.refusal = str(error)
                if isinstance(error, StepsUsedUp):
                    self.last_stop_reason = "max_calls"
                if isinstance(error, RuntimeGone):
                    for future in list(self.pending):
                        future.cancel()
                if isinstance(error, CellTimeout):
                    self.last_stop_reason = "cell_timeout"
                    from rlmness.wasm_runtime import reset_host

                    if not getattr(WasmRuntime, "ISOLATED", False):
                        await asyncio.to_thread(reset_host)
                self.active = None
                aborted = cancel.is_set()
                yield StreamFaultEvent(
                    reason="aborted" if aborted else "error",
                    error=ModelEntry(
                        model=model,
                        stop_reason="aborted" if aborted else "error",
                        error_message=str(error),
                        usage=summed_usage(accounting),
                    ),
                )
                return
            self.active = None
            usage = summed_usage(accounting)
            message = ModelEntry(
                model=model, content=[TextSegment(text=str(answer.output))], usage=usage
            )
            yield StreamCloseEvent(reason="stop", message=message)

    def cancel(self):
        if self.active is not None:
            cancel, runtimes = self.active
            if cancel is not None:
                cancel.set()
            for future in list(self.pending):
                future.cancel()
            for runtime in list(runtimes):
                runtime.channel.inbox.put(None)
                runtime.close()
            if (runtimes and not getattr(WasmRuntime, "ISOLATED", False)
                    and any(isinstance(runtime, WasmRuntime) for runtime in runtimes)):
                import threading

                from rlmness.wasm_runtime import reset_host

                threading.Thread(target=reset_host, daemon=True).start()

    async def close(self):
        self.cancel()
        await asyncio.to_thread(self.state.close_runtime)
        self.background.close()
        await asyncio.to_thread(self.background.pool.shutdown, wait=True, cancel_futures=True)
