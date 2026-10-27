"""Every agent in one run: the root, its sub-agents, and theirs."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rlmness import Allowance, AllowanceSpent

from rlmagent.agents.budget import BudgetedProvider
from rlmagent.agents.trace import TracedProvider, Tracer
from rlmagent.agents.bridge import BridgeServer, kernel_env
from rlmagent.agents.kernel_api import KERNEL_API
from rlmagent.agents.approval import Approver
from rlmagent.agents.wasm_api import wasm_api
from rlmagent.agents.workspace import Workspace
from rlmagent.kernel import KernelSession
from rlmagent.kernel.wasm import WasmKernel
from rlmagent.tools import build_tool_registry
from rlmagent.model.external import is_external, parse_model, with_external
from rlmagent.harness.contracts.tooling import ToolSpec

NUDGE = (
    "You have not called FINAL. When your work is done, call FINAL(value) in the "
    "python tool with the result the agent that started you asked for."
)
NEVER_FINAL = "[this sub-agent never called FINAL; this is its last reply]"
_ENDED = ("done", "failed", "cancelled")
_SETTLED = (*_ENDED, "idle")


@dataclass
class AgentNode:
    id: str
    depth: int
    parent_id: str | None
    task: str = ""
    context: object = None
    final_value: object = None
    final_given: bool = False
    final_note: str = ""
    role_text: str = ""
    model: str | None = None


@dataclass
class ChildRecord:
    node: AgentNode
    keep: bool = False
    status: str = "starting"
    task: asyncio.Task | None = None
    session: object = None
    value: object = None
    error: str | None = None
    holds_slot: bool = False
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    outbox: list = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    confirm: tuple[int, int] | None = None

    def set(self, status: str) -> None:
        self.status = status
        self.changed.set()
        self.changed = asyncio.Event()


GUARD_MIN_CHARS = 5000
GUARD_RATIO = 0.6

CONFIRM = (
    "STOP: confirm before doing this task. You were handed a context of {child:,} characters, "
    "{pct}% of the delegating agent's own context ({parent:,} characters), so it was passed on "
    "with little or no cutting down. Sub-agents work best when the caller slices, filters or "
    "summarises in its own kernel first and delegates only the reduced part.\n"
    "Reply on the FIRST line with exactly YES (this delegation is right, e.g. one map step over "
    "one chunk) or NO (the caller should cut it down first). If NO, give a one-line reason, then "
    "a ```python block the caller can run in its own kernel to cut the context down (slicing, "
    "searching, chunking) before delegating. Do not suggest re-sending the full context.")
CONFIRM_BATCH = (
    "STOP: confirm before this parallel batch of {n} sub-agents runs. Your context is "
    "{parent:,} characters. The sub-agents would receive:\n{lines}\n{flagged} of them get a "
    "large, barely reduced slice of it. Sub-agents work best when you slice, filter or summarise "
    "in your own kernel first and delegate only the reduced part.\n"
    "Approve the WHOLE batch? Reply YES or NO on the first line, then a one-line reason.")
REJECTED = (
    "DELEGATION_REJECTED: declined as barely reduced ({pct}% of your context). Cut the context "
    "down in your own kernel (slice, filter, summarise into a smaller variable), then delegate "
    "only that. Do not re-send the full context. The reviewer suggests:\n{reason}")
BATCH_REJECTED = (
    "BATCH_DELEGATION_REJECTED: this gather_rlm batch was declined as barely reduced. Cut each "
    "context down in your own kernel first, then delegate only the reduced parts.\n{reason}")


def _size(value: object) -> int:
    return len(value) if isinstance(value, str) else len(json.dumps(value, default=repr))


class AgentTree:
    def __init__(
        self,
        *,
        provider,
        provider_name: str,
        model: str,
        cwd: str,
        allowance: Allowance,
        sessions_dir: Path | None,
        system_for: Callable[[AgentNode, list[ToolSpec]], str],
        first_message_for: Callable[[AgentNode], str],
        cell_timeout: float = 300.0,
        max_agents: int = 50,
        tracer: Tracer | None = None,
        kernel: str = "local",
        approver: Approver | None = None,
        network: bool = False,
        guard: bool = True,
    ) -> None:
        self.guard = guard
        self.kernel_kind = kernel
        self.network = network
        self.approver = approver or Approver("deny")
        self.provider = BudgetedProvider(with_external(provider), allowance)
        self.provider_name = provider_name
        self.model = model
        self.cwd = cwd
        self.allowance = allowance
        self.sessions_dir = sessions_dir
        self.system_for = system_for
        self.first_message_for = first_message_for
        self.cell_timeout = cell_timeout
        self.max_agents = max_agents
        self.tracer = tracer or Tracer(None)
        self.tool_filter = None
        self.improver = None
        self._finished = False
        self.records: dict[str, ChildRecord] = {}
        self._kids: dict[str, list[str]] = {}
        self._agents_started = 0
        self.nodes: dict[str, AgentNode] = {}
        self.kernels: dict[str, KernelSession] = {}
        self.sessions: list = []
        self._children: dict[str, list[str]] = {}
        self._session_of: dict[str, object] = {}
        self._bridge = BridgeServer(self._handle)
        self._live = asyncio.Semaphore(max(1, allowance.max_live))
        self._ids = itertools.count(1)
        self._root_id: str | None = None

    async def start(self) -> None:
        await self._bridge.start()

    def provider_for(self, agent_id: str) -> TracedProvider:
        return TracedProvider(self, agent_id)

    def use_provider(self, provider, provider_name: str) -> BudgetedProvider:
        """Switch every agent, from the next call on, to `provider`, under the same budget."""
        self.provider = BudgetedProvider(with_external(provider), self.allowance)
        self.provider_name = provider_name
        return self.provider

    def attach(self, agent_id: str, session) -> None:
        """Record sub-agents this agent starts in `session`'s saved file."""
        self._session_of[agent_id] = session

    def root_id(self) -> str:
        if self._root_id is None:
            self.root_kernel()
        return self._root_id

    def _can_delegate(self, node: AgentNode) -> bool:
        return node.depth < self.allowance.max_depth

    def _kernel_for(self, node: AgentNode) -> KernelSession:
        if self.kernel_kind == "wasm":
            return self._sandbox_for(node)
        if not self._bridge.port:
            raise RuntimeError("start() the agent tree before creating kernels")
        env = kernel_env(
            self._bridge.port,
            self._bridge.issue(node.id),
            node.id,
            can_delegate=self._can_delegate(node),
            is_child=node.parent_id is not None,
        )
        kernel = KernelSession(
            cwd=self.cwd, env=env, startup_code=KERNEL_API, timeout=self.cell_timeout
        )
        self.kernels[node.id] = kernel
        return kernel

    def _sandbox_for(self, node: AgentNode) -> WasmKernel:
        workspace = Workspace(self.cwd, self.approver)

        async def call(op, args):
            return await self._handle(node.id, op, args)

        async def final(value):
            await self._handle(node.id, "final", {"value": value, "note": ""})

        kernel = WasmKernel(
            cwd=self.cwd, bridges={"_rlmagent": call, **workspace.functions()},
            startup_code=wasm_api(self._can_delegate(node), node.parent_id is not None,
                                  network=self.network),
            timeout=self.cell_timeout, on_final=final,
        )
        self.kernels[node.id] = kernel
        return kernel

    def root_kernel(self) -> KernelSession:
        if self._root_id is None:
            node = AgentNode(id="root", depth=0, parent_id=None)
            self.nodes[node.id] = node
            self._root_id = node.id
            self._kernel_for(node)
        return self.kernels[self._root_id]

    async def _handle(self, agent_id: str, op: str, args: dict) -> object:
        node = self.nodes.get(agent_id)
        if node is None:
            raise RuntimeError(f"unknown agent {agent_id!r}")
        if op == "context":
            return node.context
        if op == "final":
            node.final_value = args.get("value")
            node.final_note = str(args.get("note") or "")
            node.final_given = True
            return None
        if op in ("improve", "playbook"):
            improver = self._improver_of(node)
            if improver is None:
                raise RuntimeError("the playbook is not available in this run")
            if op == "playbook":
                if args.get("id"):
                    return improver.entry(str(args["id"]))
                return [{"id": e.id, "kind": e.kind, "title": e.title} for e in improver.entries()]
            improver.request(args.get("instructions"), bool(args.get("shared")))
            return "Noted: the playbook will be updated when this turn ends."
        if op == "tell_parent":
            rec = self.records.get(agent_id)
            if rec is None:
                raise RuntimeError("only sub-agents have a parent")
            rec.outbox.append(str(args.get("text", "")))
            return None
        if op == "spawn":
            return self.spawn(
                agent_id, str(args.get("task", "")), args.get("context"), bool(args.get("keep")),
                role=args.get("role"), model=args.get("model"),
            )
        if op == "children":
            return list(self._kids.get(agent_id, []))
        if op == "rlm":
            child = self.records[self.spawn(agent_id, str(args.get("task", "")), args.get("context"),
                                            role=args.get("role"), model=args.get("model"))]
            async with self._waiting_on_children(node):
                try:
                    return await self._result(child, None)
                except asyncio.CancelledError:
                    await self._cancel(child)
                    raise
        if op == "gather":
            jobs = [(str(task), context) for task, context in args.get("jobs", [])]
            children = []
            try:
                await self._confirm_batch(node, jobs)
                for task, context in jobs:
                    children.append(self.records[self.spawn(agent_id, task, context, confirm=False)])
                async with self._waiting_on_children(node):
                    return await self._gather(children)
            except BaseException:
                for child in children:
                    await self._cancel(child)
                raise
        rec = self._own(agent_id, str(args.get("id")))
        if op == "status":
            return rec.status
        if op == "messages":
            out, rec.outbox = rec.outbox, []
            return out
        if op == "cancel":
            await self._cancel(rec)
            return None
        if op == "send":
            await self._send(rec, str(args.get("text", "")))
            return None
        if op == "result":
            async with self._waiting_on_children(node):
                return await self._result(rec, args.get("timeout"))
        raise RuntimeError(f"unknown request {op!r}")

    def _improver_of(self, node: AgentNode):
        if node.parent_id is None:
            return self.improver
        rec = self.records.get(node.id)
        return getattr(rec.session, "improver", None) if rec else None

    def _role(self, role: str) -> str:
        try:
            entry = self.improver.entry(role) if self.improver else None
        except KeyError:
            entry = None
        if entry is None or entry["kind"] != "role":
            raise RuntimeError(f"no playbook role named {role!r}")
        return f"## Your role: {entry['title']}\n{entry['content']}"

    def _own(self, agent_id: str, child_id: str) -> ChildRecord:
        rec = self.records.get(child_id)
        if rec is None or rec.node.parent_id != agent_id:
            raise RuntimeError(f"{child_id} is not your sub-agent")
        return rec

    async def _gather(self, children: list[ChildRecord]) -> list:
        try:
            async with asyncio.TaskGroup() as group:
                waits = [group.create_task(self._result(c, None)) for c in children]
        except* Exception as failed:
            raise failed.exceptions[0] from None
        return [w.result() for w in waits]

    @contextlib.asynccontextmanager
    async def _waiting_on_children(self, node: AgentNode):
        """Pause the agent's cell timeout and free its live slot while it awaits sub-agents."""
        with self.kernels[node.id].waiting_outside():
            rec = self.records.get(node.id)
            if rec is None or not rec.holds_slot:
                yield
                return
            self._release(rec)
            try:
                yield
            finally:
                await self._acquire(rec)

    async def _acquire(self, rec: ChildRecord) -> None:
        await self._live.acquire()
        rec.holds_slot = True

    def _release(self, rec: ChildRecord) -> None:
        if rec.holds_slot:
            rec.holds_slot = False
            self._live.release()

    def spawn(self, parent_id: str, task: str, context: object, keep: bool = False,
              role: str | None = None, model: str | None = None, confirm: bool = True) -> str:
        parent = self.nodes[parent_id]
        role_text = self._role(role) if role else ""
        if is_external(model):
            parse_model(model)
        if not self._can_delegate(parent):
            raise RuntimeError("sub-agents are not available at this depth")
        if self._agents_started >= self.max_agents:
            raise RuntimeError(f"the limit of {self.max_agents} sub-agents started is reached")
        self._agents_started += 1
        node = AgentNode(
            id=f"{parent_id}.{next(self._ids)}",
            depth=parent.depth + 1,
            parent_id=parent_id,
            task=task,
            context=context,
            role_text=role_text,
            model=model or parent.model,
        )
        self.nodes[node.id] = node
        rec = ChildRecord(node=node, keep=keep,
                          confirm=self._barely_compressed(parent, context) if confirm else None)
        self.records[node.id] = rec
        self._kids.setdefault(parent_id, []).append(node.id)
        self.tracer.spawned(parent_id, node.id)
        rec.task = asyncio.get_running_loop().create_task(self._run(rec))
        return node.id

    async def run_child(self, parent_id: str, task: str, context: object) -> object:
        return await self._result(self.records[self.spawn(parent_id, task, context)], None)

    async def _result(self, rec: ChildRecord, timeout: float | None) -> object:
        async def settled() -> object:
            while rec.status not in _SETTLED:
                await rec.changed.wait()
            if rec.status == "failed":
                raise RuntimeError(rec.error)
            if rec.status == "cancelled":
                raise RuntimeError(f"{rec.node.id} was cancelled")
            self.tracer.returned(rec.node.parent_id, rec.node.id)
            return rec.value

        if timeout is None:
            return await settled()
        try:
            return await asyncio.wait_for(settled(), float(timeout))
        except (asyncio.TimeoutError, TimeoutError):
            raise TimeoutError(f"{rec.node.id} is still {rec.status}") from None

    async def _cancel(self, rec: ChildRecord) -> None:
        if rec.task is None:
            return
        if rec.status not in _ENDED:
            rec.task.cancel()
        # A finished agent may still be shutting its kernel down; let that end.
        await asyncio.gather(rec.task, return_exceptions=True)

    async def _send(self, rec: ChildRecord, text: str) -> None:
        if rec.status in _ENDED:
            raise RuntimeError(f"{rec.node.id} has finished; start a new sub-agent")
        if rec.status == "idle":
            rec.set("starting")
            rec.inbox.put_nowait(text)
        elif rec.session is not None:
            rec.session.inject(text)
        else:
            rec.inbox.put_nowait(text)

    async def _run(self, rec: ChildRecord) -> None:
        from rlmagent.conversation import CodingSession

        node = rec.node
        kernel = None
        try:
            await self._acquire(rec)
            rec.set("running")
            if rec.confirm is not None:
                await self._confirm(node, *rec.confirm)
            kernel = self._kernel_for(node)
            tools = (self.tool_filter or build_tool_registry)(kernel)
            session = await CodingSession.create(
                provider=self.provider_for(node.id),
                provider_name=self.provider_name,
                model=node.model or self.model,
                system=self.system_for(node, tools) + (f"\n\n{node.role_text}" if node.role_text else ""),
                tools=tools,
                sessions_dir=self.sessions_dir,
                cwd=self.cwd,
            )
            rec.session = session
            if self.improver is not None:
                from rlmagent.playbook.improver import Improver

                own = self.sessions_dir / f"{session.session_id}.playbook.json" if self.sessions_dir else None
                session.improver = Improver(session, local_path=own,
                                            shared_path=self.improver.paths["global"])
                session.improver.auto = False
            session.on_compacted = lambda: self.tracer.compacted(node.id)
            await session.set_name(f"sub-agent {node.id}: {node.task[:60]}")
            self.sessions.append(session)
            self._children.setdefault(node.parent_id, []).append(session.session_id)
            self.attach(node.id, session)
            parent_session = self._session_of.get(node.parent_id)
            if parent_session is not None:
                await parent_session.note_sub_agent(session.session_id)
            message = self.first_message_for(node)
            while True:
                rec.value = await self._round(session, node, message)
                if not rec.keep:
                    rec.set("done")
                    return
                self._release(rec)
                rec.set("idle")
                message = await rec.inbox.get()
                await self._acquire(rec)
                node.final_given = False
                rec.set("running")
        except asyncio.CancelledError:
            rec.set("cancelled")
        except Exception as exc:
            rec.error = str(exc) or type(exc).__name__
            rec.set("failed")
        finally:
            self._release(rec)
            for child_id in self._kids.get(node.id, []):
                await self._cancel(self.records[child_id])
            if kernel is not None:
                if "RLM_AGENT_BRIDGE_TOKEN" in kernel.env:
                    self._bridge.revoke(kernel.env["RLM_AGENT_BRIDGE_TOKEN"])
                await kernel.shutdown()

    # -- delegation guard: a sub-agent handed most of its parent's context confirms first --

    def _barely_compressed(self, parent: AgentNode, context: object) -> tuple[int, int] | None:
        if not self.guard or parent.context is None:
            return None
        parent_chars, child_chars = _size(parent.context), _size(context)
        if parent_chars >= GUARD_MIN_CHARS and child_chars >= GUARD_RATIO * parent_chars:
            return child_chars, parent_chars
        return None

    async def _judge(self, node: AgentNode, question: str) -> tuple[bool, str]:
        """One tool-less call with the sub-agent's own instructions; only an explicit NO refuses."""
        from rlmagent.conversation import CodingSession

        session = await CodingSession.create(
            provider=self.provider_for(node.id), provider_name=self.provider_name,
            model=node.model or self.model, system=self.system_for(node, []), tools=[],
            sessions_dir=None, cwd=self.cwd)
        try:
            async for _event in session.submit(question):
                pass
            reply = _last_text(session).strip()
        finally:
            await session.shutdown()
        if self.provider.refusal is not None:
            raise AllowanceSpent(self.provider.refusal)
        word = re.search(r"[A-Za-z]+", reply)
        return (word is None or word.group().upper() != "NO"), reply or "(no reason given)"

    async def _confirm(self, node: AgentNode, child_chars: int, parent_chars: int) -> None:
        pct = round(100 * child_chars / max(1, parent_chars))
        ok, reason = await self._judge(node, self.first_message_for(node) + "\n\n" + CONFIRM.format(
            child=child_chars, pct=pct, parent=parent_chars))
        if not ok:
            raise RuntimeError(REJECTED.format(pct=pct, reason=reason))

    async def _confirm_batch(self, parent: AgentNode, jobs: list[tuple[str, object]]) -> None:
        """One check for a whole gather_rlm, and only if some job gets most of the parent's context."""
        flagged = [self._barely_compressed(parent, context) for _, context in jobs]
        if not any(flagged):
            return
        parent_chars = _size(parent.context)
        lines = "\n".join(
            f"  [{i}] {_size(c):,} chars ({round(100 * _size(c) / max(1, parent_chars))}% of the "
            f"parent's context): {task[:140]}" for i, (task, c) in enumerate(jobs, 1))
        ok, reason = await self._judge(parent, CONFIRM_BATCH.format(
            n=len(jobs), parent=parent_chars, lines=lines, flagged=sum(1 for f in flagged if f)))
        if not ok:
            raise RuntimeError(BATCH_REJECTED.format(reason=reason))

    async def _round(self, session, node: AgentNode, message: str) -> object:
        for _ in range(2):
            async for _event in session.submit(message):
                pass
            if self.provider.refusal is not None:
                raise AllowanceSpent(self.provider.refusal)
            if node.final_given:
                if node.final_note:
                    return f"{node.final_value}\n[{node.final_note}]"
                return node.final_value
            message = NUDGE
        return f"{_last_text(session)}\n{NEVER_FINAL}"

    def child_session_ids(self, parent_id: str) -> list[str]:
        """Saved-session ids of the sub-agents an agent started, in order."""
        return list(self._children.get(parent_id, []))

    async def close(self) -> None:
        for rec in list(self.records.values()):
            await self._cancel(rec)
        for kernel in list(self.kernels.values()):
            await kernel.shutdown()
        await self._bridge.close()
        self.finish_trace()

    def finish_trace(self, complete: bool = True) -> dict | None:
        if self._finished:
            return None
        self._finished = True
        root = self._session_of.get(self._root_id or "root")
        agents = [{"agent": "root", "parent": None, "task": None, "status": "done",
                   "session": getattr(root, "session_id", None)}]
        agents += [
            {"agent": r.node.id, "parent": r.node.parent_id, "task": r.node.task,
             "status": r.status, "session": getattr(r.session, "session_id", None)}
            for r in self.records.values()
        ]
        return self.tracer.finish(agents, self.provider.refusal, complete)


def _last_text(session) -> str:
    for entry in reversed(session.transcript):
        if getattr(entry, "role", None) == "assistant":
            return entry.text
    return ""
