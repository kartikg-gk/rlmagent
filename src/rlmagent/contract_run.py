"""Run from a contract: settings come from the contract file and nowhere else."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rlmness import Allowance

from rlmagent.agents.approval import Approver
from rlmagent.agents.prompts import child_first_message, child_system, root_system
from rlmagent.agents.trace import Tracer
from rlmagent.agents.tree import AgentTree
from rlmagent.contract import Contract, ContractError, load_contract
from rlmagent.instructions import system_prompt
from rlmagent.tools import build_tool_registry


def _provider(c: Contract):
    from rlmagent.model.settings import AnthropicProfile, Credential, OpenAIProfile

    credential = Credential(api_key=c.api_key(), base_url=c.model.base_url)
    if c.model.provider == "anthropic":
        from rlmagent.model.claude import AnthropicProvider

        return AnthropicProvider(AnthropicProfile(name="anthropic", credential=credential))
    from rlmagent.model.oai_compatible import OpenAIProvider

    return OpenAIProvider(OpenAIProfile(name=c.model.provider, credential=credential))


def _tools(c: Contract, kernel) -> list:
    return [t for t in build_tool_registry(kernel) if t.name in c.tools]


async def build_contract_session(c: Contract, *, provider=None, engine="session"):
    from rlmagent.conversation import CodingSession

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
        return child_system(base(tools), node, delegates, sandboxed=sandboxed,
                            network=c.kernel.network) + extra(added)

    sandboxed = c.kernel.kind == "wasm"
    if sandboxed:
        from rlmagent.kernel.wasm import available

        if not available():
            # No silent fallback: the contract names the kernel the run must use.
            raise ContractError("kernel.kind: 'wasm' but the sandbox is not set up (rlmness --setup)")
    sessions_dir = None
    if c.sessions.save:
        sessions_dir = Path(c.sessions.dir or Path(workdir) / ".rlm-agent" / "sessions")
    from rlmagent.tools._shared import set_workspace

    set_workspace(workdir)
    if engine == "rlmness":
        from rlmagent.backend_context import BackendContext
        tree_type = BackendContext
    else:
        tree_type = AgentTree
    tree = tree_type(
        provider=provider, provider_name=c.model.provider, model=c.model.name, cwd=workdir,
        allowance=allowance, sessions_dir=sessions_dir, system_for=system_for,
        first_message_for=child_first_message, cell_timeout=lim.cell_timeout,
        max_agents=lim.max_agents, tracer=tracer,
        kernel=c.kernel.kind, restore_kernel=c.kernel.restore, approver=Approver(c.kernel.approve), network=c.kernel.network,
    )
    tree.tool_filter = lambda kernel: _tools(c, kernel)
    from rlmagent.policy import DEFAULT_OUTPUT_BYTES, Policy

    p = c.policy or {}
    tree.policy = Policy(
        allow_git=p.get("allow_git", True),
        max_tool_output_bytes=p.get("max_tool_output_bytes") or DEFAULT_OUTPUT_BYTES,
        max_compactions=p.get("max_compactions"),
        agent_max_tokens=p.get("agent_max_tokens"),
        search_block=list(p.get("search_block") or []),
    )
    await tree.start()
    kernel = tree.root_kernel()
    tools = _tools(c, kernel)
    system = root_system(base(tools), can_delegate=lim.max_depth > 0,
                         sandboxed=sandboxed, network=c.kernel.network) + extra(c.prompt.append)
    session_provider = tree.provider_for("root")
    if engine == "rlmness":
        from rlmagent.backend import BackendProvider
        from rlmagent.model.external import with_external
        session_provider = BackendProvider(with_external(provider), session_provider, tree)
        session_provider.instructions = base([]) + extra(c.prompt.append)
        session_provider.event_path = Path(str(c.trace.file) + ".backend.events.jsonl")
    session = await CodingSession.create(
        provider=session_provider, provider_name=c.model.provider, model=c.model.name,
        system=system, tools=tools, sessions_dir=sessions_dir, cwd=workdir,
    )
    await session.set_name("contract run")
    if engine == "rlmness":
        from rlmness.session import Session as BackendState
        session.backend = session_provider
        if sessions_dir is not None:
            session_provider.state = BackendState.load(sessions_dir / f"{session.session_id}.backend.json")
    session.agent_tree = tree
    session.policy = tree.policy
    if engine == "rlmness":
        # Background jobs, watches and the full record, under the contract's policy.
        from rlmagent.agents.approval import Request, check
        from rlmagent.history import make_history
        from rlmagent.jobs import Jobs
        from rlmagent.policy import git_refusal

        async def approve(command: str) -> None:
            request = Request("run", command)
            check(await tree.approver.approve(request), request)

        session.jobs = Jobs(workdir, Path(str(c.trace.file) + ".jobs"), approve,
                            lambda command: git_refusal(command, tree.policy))
        tree.host_functions = {
            **session.jobs.functions(),
            "history": make_history(lambda: getattr(getattr(session, "_vault", None), "path", None),
                                    lambda: session_provider.event_path),
        }
    tree.attach(tree.root_id(), session)
    if c.playbook is not None:
        from rlmagent.playbook.improver import Improver
        path = Path(c.playbook["file"])
        if not path.is_absolute():
            path = Path(workdir) / path
        session.improver = tree.improver = Improver(session, path, None)
        tree.playbook_writable = c.playbook["update"]
        session.improver.auto = c.playbook["update"]
        session.improver.writable = c.playbook["update"]
        session.improver.show_digest()
    session.on_compacted = lambda: (tracer.compacted("root"),
        session.improver.compacted() if getattr(session, "improver", None) else None)
    return session


async def run_contract(ns) -> int:
    from rlmagent.cli.main import _run_one_shot

    if ns.prompt is None and not sys.stdin.isatty():
        ns.prompt = sys.stdin.read().strip()
    if not ns.prompt:
        print("rlm-agent: --contract needs a prompt (-p \"...\")", file=sys.stderr)
        return 2
    try:
        contract = load_contract(ns.contract)
        contract.api_key()
        session = await build_contract_session(contract, engine=contract.engine)
    except (ContractError, OSError) as exc:
        print(f"rlm-agent: {exc}", file=sys.stderr)
        return 2
    os.chdir(session.agent_tree.cwd)
    tree = session.agent_tree
    try:
        return await _run_one_shot(session, ns.prompt, print_mode=True)
    finally:
        from rlmagent.stops import of_session

        stop = of_session(session, tree.provider.refusal)
        await session.shutdown()
        summary = tree.finish_trace() or tree.tracer.records[-1]
        print(json.dumps({**{k: v for k, v in summary.items() if k != "kind"}, **stop}),
              file=sys.stderr)
