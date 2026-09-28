"""Every agent in one run: the root, its sub-agents, and theirs."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rlmness import Allowance, AllowanceSpent

from rlmagent_app.agents.budget import BudgetedProvider
from rlmagent_app.agents.bridge import BridgeServer, kernel_env
from rlmagent_app.agents.kernel_api import KERNEL_API
from rlmagent_app.kernel import KernelSession
from rlmagent_app.tools import build_tool_registry
from rlmagent_harness.contracts.tooling import ToolSpec

NUDGE = (
    "You have not called FINAL. When your work is done, call FINAL(value) in the "
    "python tool with the result the agent that started you asked for."
)
NEVER_FINAL = "[this sub-agent never called FINAL; this is its last reply]"


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
    ) -> None:
        self.provider = BudgetedProvider(provider, allowance)
        self.provider_name = provider_name
        self.model = model
        self.cwd = cwd
        self.allowance = allowance
        self.sessions_dir = sessions_dir
        self.system_for = system_for
        self.first_message_for = first_message_for
        self.cell_timeout = cell_timeout
        self.nodes: dict[str, AgentNode] = {}
        self.kernels: dict[str, KernelSession] = {}
        self.sessions: list = []
        self._children: dict[str, list[str]] = {}
        self._bridge = BridgeServer(self._handle)
        self._live = asyncio.Semaphore(max(1, allowance.max_live))
        self._ids = itertools.count(1)
        self._root_id: str | None = None

    async def start(self) -> None:
        await self._bridge.start()

    def use_provider(self, provider, provider_name: str) -> BudgetedProvider:
        """Switch every agent, from the next call on, to `provider`, under the same budget."""
        self.provider = BudgetedProvider(provider, self.allowance)
        self.provider_name = provider_name
        return self.provider

    def root_id(self) -> str:
        if self._root_id is None:
            self.root_kernel()
        return self._root_id

    def _can_delegate(self, node: AgentNode) -> bool:
        return node.depth < self.allowance.max_depth

    def _kernel_for(self, node: AgentNode) -> KernelSession:
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
        if op == "rlm":
            async with self._waiting_on_children(node):
                return await self.run_child(
                    agent_id, str(args.get("task", "")), args.get("context")
                )
        if op == "gather":
            jobs = [(str(task), context) for task, context in args.get("jobs", [])]
            async with self._waiting_on_children(node):
                try:
                    async with asyncio.TaskGroup() as group:
                        tasks = [
                            group.create_task(self.run_child(agent_id, t, c)) for t, c in jobs
                        ]
                except* Exception as failed:
                    raise failed.exceptions[0] from None
            return [task.result() for task in tasks]
        raise RuntimeError(f"unknown request {op!r}")

    @contextlib.asynccontextmanager
    async def _waiting_on_children(self, node: AgentNode):
        """Pause the agent's cell timeout and free its live slot while it awaits sub-agents."""
        with self.kernels[node.id].waiting_outside():
            if node.parent_id is None:
                yield
                return
            self._live.release()
            try:
                yield
            finally:
                await self._live.acquire()

    async def run_child(self, parent_id: str, task: str, context: object) -> object:
        from rlmagent_app.conversation import CodingSession

        parent = self.nodes[parent_id]
        if not self._can_delegate(parent):
            raise RuntimeError("sub-agents are not available at this depth")
        node = AgentNode(
            id=f"{parent_id}.{next(self._ids)}",
            depth=parent.depth + 1,
            parent_id=parent_id,
            task=task,
            context=context,
        )
        self.nodes[node.id] = node
        async with self._live:
            kernel = self._kernel_for(node)
            tools = build_tool_registry(kernel)
            session = await CodingSession.create(
                provider=self.provider,
                provider_name=self.provider_name,
                model=self.model,
                system=self.system_for(node, tools),
                tools=tools,
                sessions_dir=self.sessions_dir,
                cwd=self.cwd,
            )
            await session.set_name(f"sub-agent {node.id}: {task[:60]}")
            self.sessions.append(session)
            self._children.setdefault(parent_id, []).append(session.session_id)
            try:
                message = self.first_message_for(node)
                for attempt in range(2):
                    async for _ in session.submit(message):
                        pass
                    if self.provider.refusal is not None:
                        raise AllowanceSpent(self.provider.refusal)
                    if node.final_given:
                        if node.final_note:
                            return f"{node.final_value}\n[{node.final_note}]"
                        return node.final_value
                    message = NUDGE
                return f"{_last_text(session)}\n{NEVER_FINAL}"
            finally:
                self._bridge.revoke(kernel.env["RLM_AGENT_BRIDGE_TOKEN"])
                await kernel.shutdown()

    def child_session_ids(self, parent_id: str) -> list[str]:
        """Saved-session ids of the sub-agents an agent started, in order."""
        return list(self._children.get(parent_id, []))

    async def close(self) -> None:
        for kernel in list(self.kernels.values()):
            await kernel.shutdown()
        await self._bridge.close()


def _last_text(session) -> str:
    for entry in reversed(session.transcript):
        if getattr(entry, "role", None) == "assistant":
            return entry.text
    return ""
