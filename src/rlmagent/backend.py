"""Adapt the backend's recursive solver to the application's model providers."""

from __future__ import annotations

import asyncio
import contextvars
import json
import queue
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import time

from rlmness import Config, solve
from rlmness.background import Background
from rlmness.providers import Spend
from rlmness.session import Session
from rlmness.wasm_runtime import WasmRuntime

from rlmagent.agents.workspace import Workspace
from rlmagent.harness.contracts.transcript import HumanEntry, ModelEntry, TextSegment, UsageStats
from rlmagent.harness.provider.wire import ContentChunkEvent, StreamCloseEvent, StreamFaultEvent


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
    ):
        self.provider, self.loop, self.signal = provider, loop, signal
        self.pending = pending if pending is not None else set()
        self.tree, self.context = tree, context
        self.instructions = instructions
        self.background = background
        self.accounting, self.record = accounting, record
        self.host_context = contextvars.copy_context()

    def complete(self, messages, *, model):
        agent = getattr(self.context, "agent", "root")

        async def request():
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            system += "\n\n" + self.instructions
            entries = [
                (
                    ModelEntry(content=[TextSegment(text=m["content"])])
                    if m["role"] == "assistant"
                    else HumanEntry(content=m["content"])
                )
                for m in messages
                if m["role"] != "system"
            ]
            tracer = getattr(self.tree, "tracer", None)
            call = tracer.begin(agent, model) if tracer is not None else None
            async for event in self.provider.stream_response(
                model=model, system=system, messages=entries, tools=[], signal=self.signal
            ):
                if isinstance(event, StreamFaultEvent):
                    raise RuntimeError(event.error.error_message or "model request failed")
                if isinstance(event, StreamCloseEvent):
                    usage = event.message.usage
                    if self.accounting is not None:
                        self.accounting.append(usage)
                    if self.record is not None:
                        self.record(
                            {
                                "kind": "model_usage",
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
                        cost=usage.cost.total,
                        cached_tokens=usage.cache_read,
                        reasoning_tokens=usage.reasoning,
                    )
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
        self.background = Background(getattr(tree, "max_agents", 50))
        self.background.resolve_role = getattr(tree, "_role", None)
        self.pending = set()
        self.state = Session()
        self.instructions = ""
        self.prompt_context = {}
        self.events = []
        self._serial = asyncio.Lock()
        self.event_path = None
        self._event_lock = threading.Lock()

    def record_event(self, event):
        with self._event_lock:
            event = {"timestamp": time(), **event}
            self.events.append(event)
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

        async with self._serial:
            loop = asyncio.get_running_loop()
            host_context = contextvars.copy_context()
            tree = self.tree
            from rlmagent.model.external import with_external

            if hasattr(tree, "provider"):
                self.provider = with_external(tree.provider.inner)
            workspace = Workspace(tree.cwd, tree.approver)
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

                def wrap(fn):
                    def call(*args, **kw):
                        async def invoke():
                            from rlmagent.agents.approval import Refused

                            try:
                                value = await fn(*args, **kw)
                            except Refused:
                                raise
                            except (OSError, ValueError, UnicodeError) as error:
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

                joined = {
                    **bridges,
                    **{name: wrap(fn) for name, fn in workspace.functions().items()},
                }

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

                async def fetch_url(url, max_chars=50000):
                    from rlmagent.agents.approval import Refused
                    from rlmagent.network import fetch_text

                    if not tree.network:
                        raise Refused("network access is disabled")
                    return await fetch_text(url, max_chars=max_chars)

                async def guidance(id=None):
                    refinement = getattr(tree, "refinement", None)
                    return refinement.read_guidance(id) if refinement is not None else []

                joined["guidance"] = wrap(guidance)
                joined.update(
                    playbook=wrap(playbook), improve=wrap(improve), fetch_url=wrap(fetch_url)
                )
                from rlmness.runtime import SubprocessRuntime

                runtime_type = (
                    WasmRuntime
                    if getattr(tree, "kernel_kind", "wasm") == "wasm"
                    else SubprocessRuntime
                )
                # Host startup and helper installation have their own bounded window.
                # The requested cell deadline still applies to every model-written cell.
                runtime = runtime_type(prompt, joined, max(30.0, timeout), tools, **kwargs)
                runtimes.append(runtime)
                runtime.execute(
                    "async def skill(id):\n"
                    "    entry = await playbook(id)\n"
                    "    namespace = {}\n"
                    "    exec(entry['content'], namespace)\n"
                    "    return namespace[entry['callable']]"
                )
                if tree.network and isinstance(runtime, WasmRuntime):
                    from rlmagent.agents.wasm_api import NETWORK

                    outcome = runtime.execute(NETWORK)
                    if outcome.error:
                        runtime.close()
                        raise RuntimeError(outcome.error)
                runtime.timeout = timeout
                return runtime

            factory.SEALED = True
            factory.NEEDS_SOURCE = True
            factory.MAX_LIVE = WasmRuntime.MAX_LIVE
            cancel = threading.Event()
            self.active = (cancel, runtimes)
            from rlmness.background import PRELUDE

            self.state.functions["_AgentHandle"] = PRELUDE.split("async def spawn(")[0]
            config = Config(
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
                    "Complete the user's coding task. Workspace access uses async read_file(path), "
                    "list_files(pattern), write_file(path, content), edit_file(path, old, new), "
                    "and run(cmd, timeout=120). Changes and commands use host approval. "
                    "Ordinary tool failures return text starting with Error:; inspect results "
                    "before continuing. "
                    "await fetch_url(url, max_chars=50000) fetches bounded text "
                    "when network is enabled. "
                    "await instruction_docs() discovers project instruction and skill files; "
                    "await instruction_docs(path) reads a selected file. "
                    "Read relevant instructions before editing. "
                    "Use the backend's rlm(piece, instruction=question) and gather_rlm APIs. "
                    "Background jobs use await spawn(task, data, keep=False, "
                    "role=None, model=None); "
                    "the returned handle provides result(timeout=None), status(), send(text), "
                    "messages(), cancel(), and id. Persistent jobs use keep=True. "
                    "Children can await tell_parent(text). await children() lists handles. "
                    "Playbooks use await playbook(id=None), await skill(id), and "
                    "await improve(instructions, shared=False). "
                    "Return the answer with FINAL. The conversation is in PROMPT."
                )
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
                        if (
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
                            text = str(event["output"]) + "\n"
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
                    error=ModelEntry(model=model, stop_reason="aborted", error_message="cancelled"),
                )
                return
            except Exception as error:
                from rlmness.runtime import CellTimeout

                if isinstance(error, CellTimeout):
                    from rlmness.wasm_runtime import reset_host

                    await asyncio.to_thread(reset_host)
                self.active = None
                aborted = cancel.is_set()
                yield StreamFaultEvent(
                    reason="aborted" if aborted else "error",
                    error=ModelEntry(
                        model=model,
                        stop_reason="aborted" if aborted else "error",
                        error_message=str(error),
                    ),
                )
                return
            self.active = None
            usage = UsageStats(
                **{
                    name: sum(getattr(item, name) or 0 for item in accounting)
                    for name in (
                        "input",
                        "output",
                        "total_tokens",
                        "cache_read",
                        "cache_write",
                        "cache_write_1h",
                        "reasoning",
                    )
                },
                cost={
                    name: sum(getattr(item.cost, name) for item in accounting)
                    for name in ("total", "input", "output", "cache_read", "cache_write")
                },
            )
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
            if runtimes and any(isinstance(runtime, WasmRuntime) for runtime in runtimes):
                import threading

                from rlmness.wasm_runtime import reset_host

                threading.Thread(target=reset_host, daemon=True).start()

    async def close(self):
        self.cancel()
        self.background.close()
