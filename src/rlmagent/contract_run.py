"""Run from a contract: settings come from the contract file and nowhere else."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rlmagent.agents.approval import Approver
from rlmagent.agents.budget import ContractAllowance, NewTokenBudget
from rlmagent.agents.prompts import child_first_message, child_system, root_system
from rlmagent.agents.trace import Tracer
from rlmagent.agents.tree import AgentTree
from rlmagent.contract import ACCOUNT_PROVIDERS, Contract, ContractError, load_contract
from rlmagent.instructions import system_prompt
from rlmagent.tools import build_tool_registry


def _provider(c: Contract, credential=None):
    from rlmagent.model.settings import AnthropicProfile, Credential, OpenAIProfile

    if c.model.provider in ACCOUNT_PROVIDERS:
        if credential is not None:
            raise ContractError("Account routes do not accept injected credentials")
        if c.model.provider == "claude-subscription":
            from rlmagent.model.external import ExternalAgentProvider

            return ExternalAgentProvider()
        from rlmagent.model.oai_compatible import OpenAIProvider
        from rlmagent.model.subscription import RESOURCE, resolve

        return OpenAIProvider(OpenAIProfile(name="codex", credential=Credential(None, RESOURCE),
            resolver=resolve, endpoint="responses", transport="http", idempotency=True))
    credential = credential or Credential(api_key=c.api_key(), base_url=c.model.base_url)
    if c.model.provider == "google":
        from rlmagent.model.google import GoogleProvider

        return GoogleProvider(credential)
    if c.model.provider == "anthropic":
        from rlmagent.model.claude import AnthropicProvider

        return AnthropicProvider(AnthropicProfile(name="anthropic", credential=credential,
                                                  idempotency=True))
    from rlmagent.model.oai_compatible import OpenAIProvider

    if c.model.provider == "azure":
        credential = Credential(None, credential.base_url,
                                (*credential.extra_headers, ("api-key", credential.api_key)))
    return OpenAIProvider(OpenAIProfile(
        name=c.model.provider, credential=credential, idempotency=True,
        endpoint="responses" if c.model.provider == "azure" else "chat" if c.model.provider == "mistral" else None,
    ))


def _tools(c: Contract, kernel, policy) -> list:
    return [t for t in build_tool_registry(kernel) if t.name in c.tools and policy.allows_tool(t.name)]


async def build_contract_session(c: Contract, *, provider=None, engine="session"):
    from rlmagent.conversation import CodingSession

    provider = provider or _provider(c)
    workdir = str(Path(c.workdir).resolve())
    identity = Path(c.prompt.system_file).read_text(encoding="utf-8") if c.prompt.system_file else None
    trace_file = Path(c.trace.file).resolve()
    tracer = Tracer(trace_file, contract=c.raw)
    lim = c.limits
    allowance = ContractAllowance(
        max_depth=lim.max_depth, max_calls=lim.max_calls, max_cost=lim.max_cost,
        max_live=lim.max_live, max_seconds=lim.max_seconds,
        new_token_budget=(NewTokenBudget(lim.max_new_tokens)
                          if lim.max_new_tokens is not None else None),
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
    tree.request_timeout = lim.request_timeout
    from rlmagent.policy import DEFAULT_OUTPUT_BYTES, Policy

    p = c.policy or {}
    tree.policy = Policy(
        allow_git=p.get("allow_git", True),
        max_tool_output_bytes=(p["max_tool_output_bytes"] if p.get("max_tool_output_bytes") is not None
                               else DEFAULT_OUTPUT_BYTES),
        max_compactions=p.get("max_compactions"),
        agent_max_tokens=p.get("agent_max_tokens"),
        search_block=list(p.get("search_block") or []),
        shell_access=p.get("shell_access", "workspace"),
        shell_network=p.get("shell_network", False),
        allowed_tools=frozenset(p["allowed_tools"]) if "allowed_tools" in p else None,
    )
    tree.tool_filter = lambda kernel: _tools(c, kernel, tree.policy)
    await tree.start()
    kernel = tree.root_kernel()
    tools = tree.tool_filter(kernel)
    system = root_system(base(tools), can_delegate=lim.max_depth > 0,
                         sandboxed=sandboxed, network=c.kernel.network) + extra(c.prompt.append)
    session_provider = tree.provider_for("root")
    if engine == "rlmness":
        from rlmagent.backend import BackendProvider
        from rlmagent.model.external import with_external
        session_provider = BackendProvider(with_external(provider), session_provider, tree)
        session_provider.instructions = base([]) + extra(c.prompt.append)
        session_provider.event_path = Path(str(trace_file) + ".backend.events.jsonl")
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
    session.runtime_contract = c
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

        session.jobs = Jobs(workdir, Path(str(trace_file) + ".jobs"), approve,
                            lambda command: git_refusal(command, tree.policy),
                            shell_access=tree.policy.shell_access,
                            shell_network=tree.policy.shell_network)
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
    return await execute_session(session, ns.prompt)


async def execute_session(session, prompt: str | list[str]) -> int:
    from rlmagent.cli.main import _run_one_shot

    previous_dir = Path.cwd()
    tree = session.agent_tree
    try:
        os.chdir(tree.cwd)
        try:
            for text in [prompt] if isinstance(prompt, str) else prompt:
                status = await _run_one_shot(session, text, print_mode=True)
                if status:
                    return status
            return 0
        finally:
            from rlmagent.stops import of_session

            stop = of_session(session, tree.provider.refusal)
            await session.shutdown()
            summary = tree.finish_trace() or tree.tracer.records[-1]
            print(json.dumps({**{k: v for k, v in summary.items() if k != "kind"}, **stop}),
                  file=sys.stderr)
    finally:
        os.chdir(previous_dir)
