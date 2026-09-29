"""Run from a contract: settings come from the contract file and nowhere else."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rlmness import Allowance

from rlmagent_app.agents.prompts import child_first_message, child_system, root_system
from rlmagent_app.agents.trace import Tracer
from rlmagent_app.agents.tree import AgentTree
from rlmagent_app.contract import Contract, ContractError, load_contract
from rlmagent_app.instructions import system_prompt
from rlmagent_app.tools import build_tool_registry


def _provider(c: Contract):
    from rlmagent_model.settings import AnthropicProfile, Credential, OpenAIProfile

    credential = Credential(api_key=c.api_key(), base_url=c.model.base_url)
    if c.model.provider == "anthropic":
        from rlmagent_model.claude import AnthropicProvider

        return AnthropicProvider(AnthropicProfile(name="anthropic", credential=credential))
    from rlmagent_model.oai_compatible import OpenAIProvider

    return OpenAIProvider(OpenAIProfile(name=c.model.provider, credential=credential))


def _tools(c: Contract, kernel) -> list:
    return [t for t in build_tool_registry(kernel) if t.name in c.tools]


async def build_contract_session(c: Contract, *, provider=None):
    from rlmagent_app.conversation import CodingSession

    provider = provider or _provider(c)
    workdir = str(Path(c.workdir).resolve())
    identity = Path(c.prompt.system_file).read_text(encoding="utf-8") if c.prompt.system_file else None
    tracer = Tracer(c.trace.file, contract=c.raw)
    lim = c.limits
    allowance = Allowance(
        max_depth=lim.max_depth, max_calls=lim.max_calls, max_cost=lim.max_cost,
        max_live=lim.max_live, max_seconds=lim.max_seconds,
    )

    def base(tools) -> str:
        return system_prompt(tools=tools, cwd=workdir, include_date=False,
                             identity=identity, project=False)

    def extra(text: str | None) -> str:
        text = text if text is not None else c.prompt.append
        return f"\n\n{text}" if text else ""

    def system_for(node, tools) -> str:
        delegates = node.depth < lim.max_depth
        added = c.prompt.append_sub_agent if delegates else c.prompt.append_leaf
        return child_system(base(tools), node, delegates) + extra(added)

    sessions_dir = None
    if c.sessions.save:
        sessions_dir = Path(c.sessions.dir or Path(workdir) / ".rlm-agent" / "sessions")
    tree = AgentTree(
        provider=provider, provider_name=c.model.provider, model=c.model.name, cwd=workdir,
        allowance=allowance, sessions_dir=sessions_dir, system_for=system_for,
        first_message_for=child_first_message, cell_timeout=lim.cell_timeout,
        max_agents=lim.max_agents, tracer=tracer,
    )
    tree.tool_filter = lambda kernel: _tools(c, kernel)
    await tree.start()
    kernel = tree.root_kernel()
    tools = _tools(c, kernel)
    system = root_system(base(tools), can_delegate=lim.max_depth > 0) + extra(c.prompt.append)
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name=c.model.provider, model=c.model.name,
        system=system, tools=tools, sessions_dir=sessions_dir, cwd=workdir,
    )
    await session.set_name("contract run")
    session.agent_tree = tree
    tree.attach(tree.root_id(), session)
    session.on_compacted = lambda: tracer.compacted("root")
    return session


async def run_contract(ns) -> int:
    from rlmagent_app.cli.main import _run_one_shot

    if ns.prompt is None and not sys.stdin.isatty():
        ns.prompt = sys.stdin.read().strip()
    if not ns.prompt:
        print("rlm-agent: --contract needs a prompt (-p \"...\")", file=sys.stderr)
        return 2
    try:
        contract = load_contract(ns.contract)
        contract.api_key()
        session = await build_contract_session(contract)
    except (ContractError, OSError) as exc:
        print(f"rlm-agent: {exc}", file=sys.stderr)
        return 2
    os.chdir(session.agent_tree.cwd)
    tree = session.agent_tree
    try:
        return await _run_one_shot(session, ns.prompt, print_mode=True)
    finally:
        await session.shutdown()
        summary = tree.finish_trace() or tree.tracer.records[-1]
        print(json.dumps({k: v for k, v in summary.items() if k != "kind"}), file=sys.stderr)
