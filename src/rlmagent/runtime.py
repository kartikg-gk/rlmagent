"""Shared session construction for every rlm-agent frontend."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlmagent.conversation import CodingSession
    from rlmagent.trust import Asker, Choice, TrustOutcome, TrustRequest


async def build_session(ns: argparse.Namespace) -> CodingSession:
    """Resolve configuration and return a ready ``CodingSession``."""
    from uuid import uuid4

    from rlmagent.cli.main import _die, _sessions_dir
    from rlmagent.harness.session.lease import SessionBusy, acquire

    if getattr(ns, "no_session", False):
        return await _build_session(ns)
    session_id = getattr(ns, "resume", None) or uuid4().hex[:12]
    try:
        lease = acquire(_sessions_dir(getattr(ns, "session_dir", None)), session_id)
    except (SessionBusy, ValueError) as exc:
        _die(str(exc))
    try:
        session = await _build_session(ns, session_id=session_id)
    except BaseException:
        lease.close()
        raise
    session._writer_lease = lease
    return session


async def _build_session(ns: argparse.Namespace, *, session_id: str | None = None) -> CodingSession:
    from rlmagent.cli.main import (
        _die,
        _info,
        _install_hooks,
        _install_safety,
        _load_extensions,
        _load_prompt_templates,
        _load_skills,
        _load_tools,
        _resolve_model,
        _resolve_provider,
        _resolve_provider_name,
        _resolve_system,
        _sessions_dir,
    )
    from rlmagent.config.loader import apply_config, load_model
    from rlmagent.conversation import CodingSession
    from rlmagent.agents.prompts import root_system
    from rlmagent.safety import ApprovalPolicy

    provider_name = _resolve_provider_name(getattr(ns, "provider", None))
    # Export saved credentials for this process so the provider loaders
    # (which read the environment) see them. Real env vars keep precedence.
    apply_config(provider_name)
    model = (
        getattr(ns, "model", None)
        or load_model(provider_name)
        or _resolve_model(None, provider_name)
    )
    provider = _resolve_provider(getattr(ns, "provider", None))

    verbose = getattr(ns, "verbose", False)
    await settle_project_trust(ns, ask=getattr(ns, "trust_ask", None))

    no_session = getattr(ns, "no_session", False)
    sessions_dir = None if no_session else _sessions_dir(getattr(ns, "session_dir", None))

    # The root agent and every sub-agent share one tree: one budget, one
    # channel back from the kernels, one place that shuts them all down.
    from rlmagent.tools._shared import set_workspace

    set_workspace(os.getcwd())
    from rlmagent.plugins import rlm_agent_home
    from rlmagent.undo import Shadow

    tree = make_agent_tree(ns, provider, provider_name, model, sessions_dir, cwd=os.getcwd(),
                           backend=getattr(ns, "engine", "session") == "rlmness")
    await tree.start()
    kernel = tree.root_kernel()
    raw_provider = provider
    provider = tree.provider_for("root")
    if getattr(ns, "engine", "session") == "rlmness":
        from rlmagent.backend import BackendProvider
        from rlmagent.model.external import with_external
        provider = BackendProvider(with_external(raw_provider), provider, tree)

    from rlmagent.plugins import (
        as_tool, contributions, load_extensions, reload_extensions, rlm_agent_home,
    )
    from rlmagent.tasks import TaskList

    # Harness functions every agent gets: cells call them, the session engine sees tools.
    tasks = TaskList()
    from pathlib import Path

    from rlmagent.mcp import McpHub, load_configs
    from rlmagent.trust import project_inputs_allowed

    here = os.getcwd()
    configs, problems = load_configs(rlm_agent_home(),
                                     Path(here) if project_inputs_allowed(here) else None)
    supplied = getattr(ns, "acp_mcp", {})
    if conflicts := sorted(configs.keys() & supplied.keys()):
        _die("Editor MCP names already configured: " + ", ".join(conflicts))
    configs.update(supplied)
    hub = McpHub(configs, here, tree.approver, problems)
    from rlmagent.agentnet import AgentNet

    net = AgentNet("", here)
    from rlmagent.autopilot import Autopilot

    pilot = Autopilot(cwd=here)
    from rlmagent.websearch import make_web_search

    built: dict = {}  # the session, once it exists; these functions run only after that

    async def context_usage() -> dict:
        """How full the conversation context is: used and limit in tokens, and percent."""
        usage = built["session"].usage
        limit = max(usage.context_limit, 1)
        return {"used": usage.estimated_context_tokens, "limit": usage.context_limit,
                "percent": round(100 * usage.estimated_context_tokens / limit)}

    async def compact() -> str:
        """Summarize older conversation when this turn ends, to free context for more work."""
        built["session"].compact_requested = True
        return "older conversation will be summarized when this turn ends"

    from rlmagent.agents.approval import Request, check
    from rlmagent.history import make_history
    from rlmagent.jobs import Jobs
    from rlmagent.policy import DEFAULT_OUTPUT_BYTES, Policy, git_refusal

    policy = tree.policy = Policy(
        allow_git=not getattr(ns, "deny_git", False),
        max_tool_output_bytes=getattr(ns, "max_tool_output", None) or DEFAULT_OUTPUT_BYTES,
        max_compactions=getattr(ns, "max_compactions", None),
        agent_max_tokens=getattr(ns, "agent_max_tokens", None),
        search_block=list(getattr(ns, "search_block", None) or []),
    )

    async def _approve_job(command: str) -> None:
        request = Request("run", command)
        check(await tree.approver.approve(request), request)

    async def factory_run(name: str) -> dict:
        """Run a named factory from the user's library after workspace approval."""
        import shlex

        from rlmagent.agents.workspace import Workspace
        from rlmagent.factory import template

        if not template(name).is_file():
            raise ValueError(f"no factory named {name!r}")
        command = shlex.join([Path(sys.executable).as_posix(), "-m", "rlmagent.cli.main",
                              "factory", "run", name])
        workspace = Workspace(here, tree.approver)
        workspace.policy = policy
        return await workspace.run(command, timeout=1800)

    jobs = Jobs(here, (sessions_dir or rlm_agent_home()) / "jobs", _approve_job,
                lambda command: git_refusal(command, policy))
    tools_at = getattr(ns, "tools_at", None)
    if tools_at:
        if getattr(ns, "engine", "session") != "rlmness":
            _die("--tools-at needs the default engine (--engine rlmness).")
        from rlmagent.toolhost import connect

        try:
            tree.remote_tools = await connect(tools_at)
        except (OSError, ValueError, ConnectionError) as exc:
            _die(f"Cannot reach the tool host at {tools_at}: {exc}")
    tree.host_functions = {"todo": tasks.todo, "mcp": hub.mcp, "mcp_tools": hub.mcp_tools,
                           **net.functions(), **pilot.functions(),
                           **jobs.functions(inbox=lambda: net.take()),
                           "context_usage": context_usage, "compact": compact,
                           "factory_run": factory_run,
                           "web_search": make_web_search(lambda: tree.network, policy),
                           "history": make_history(
                               lambda: getattr(getattr(built.get("session"), "_vault", None),
                                               "path", None),
                               lambda: getattr(getattr(built.get("session"), "backend", None),
                                               "event_path", None))}
    builtin_tools = [as_tool(name, fn) for name, fn in tree.host_functions.items()]
    extended: dict = {"tools": [], "functions": {}}

    def _take_extensions(result) -> None:
        for failure in result.failures:
            print(f"rlm-agent: extension {failure.name!r} failed: {failure.error}", file=sys.stderr)
        extra, functions = contributions(result)
        extended["tools"] = [*extra, *(as_tool(name, fn) for name, fn in functions.items())]
        extended["functions"] = functions
        tree.extension_functions = functions

    _load_extensions(verbose=verbose)
    _take_extensions(load_extensions())
    tools = [*_load_tools(verbose=verbose, kernel=kernel), *builtin_tools, *extended["tools"]]
    skills = _load_skills(os.getcwd(), verbose=verbose)
    system = root_system(
        _resolve_system(getattr(ns, "system_prompt", None), tools=tools, skills=skills),
        can_delegate=tree.allowance.max_depth > 0,
        playbook=True,
        sandboxed=tree.kernel_kind == "wasm",
        network=tree.network,
    )

    def _tools_loader():
        _take_extensions(reload_extensions())
        return [*_load_tools(kernel=kernel), *builtin_tools, *extended["tools"]]

    def _skills_loader():
        return _load_skills(os.getcwd())

    def _templates_loader():
        return _load_prompt_templates(os.getcwd())

    resume = getattr(ns, "resume", None)
    if resume:
        if sessions_dir is None:
            _die("Cannot use --resume together with --no-session.")
        problem = await _resume_folder_problem(ns, resume, sessions_dir)
        if problem:
            _die(problem)
        try:
            session = await CodingSession.resume(
                resume,
                provider=provider,
                provider_name=provider_name,
                model=model,
                system=system,
                tools=tools,
                sessions_dir=sessions_dir,
                tools_loader=_tools_loader,
                skills_loader=_skills_loader,
                templates_loader=_templates_loader,
                keep_model=bool(getattr(ns, "model", None)),
            )
        except FileNotFoundError:
            _die(f"Session not found: {resume}")
        if verbose:
            _info(f"Resumed {len(session.transcript)} messages from {resume}")
        if session.interrupted:
            _info("The last turn of this session did not finish; its work may be incomplete. "
                  "Send a message, or /continue, to carry on.")
    else:
        session = await CodingSession.create(
            session_id=session_id,
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            tools=tools,
            sessions_dir=sessions_dir,
            tools_loader=_tools_loader,
            skills_loader=_skills_loader,
            templates_loader=_templates_loader,
        )

    if getattr(ns, "engine", "session") == "rlmness":
        from rlmness.session import Session as BackendState
        provider.instructions = _resolve_system(
            getattr(ns, "system_prompt", None), tools=[], skills=skills)
        if sessions_dir is not None:
            provider.event_path = sessions_dir / f"{session.session_id}.backend.events.jsonl"
            provider.state = BackendState.load(
                sessions_dir / f"{session.session_id}.backend.json")
    if getattr(ns, "engine", "session") == "rlmness":
        session.backend = provider
    session.agent_tree = tree
    session.shadow = Shadow(session.cwd, rlm_agent_home())
    session.load_undo()
    session.tasks = tasks
    session.mcp = hub
    session.agentnet = net
    session.jobs = jobs
    session.remote_tools = getattr(tree, "remote_tools", None)
    session.policy = policy
    built["session"] = session
    session.autopilot = pilot
    pilot.path = sessions_dir / f"{session.session_id}.goal.json" if sessions_dir else None
    pilot.used_tokens = lambda: (session.usage.total_input_tokens
                                 + session.usage.total_output_tokens)
    pilot.load()
    net.me = f"s-{session.session_id[:8]}"
    if sessions_dir is not None:
        tree.tracer.persist(sessions_dir / f"{session.session_id}.requests.jsonl")
    net.register()

    def _progress(text: str) -> None:
        backend = getattr(session, "backend", None)
        if backend is not None and hasattr(backend, "record_event"):
            backend.record_event({"kind": "progress", "text": text})

    net.on_progress = _progress
    tasks.attach(sessions_dir / f"{session.session_id}.tasks.json" if sessions_dir else None)
    tree.attach(tree.root_id(), session)
    from rlmagent.playbook.improver import Improver
    from rlmagent.plugins import rlm_agent_home
    from rlmagent.refinement.controller import project_folder
    from rlmagent.refinement.evidence import EvidenceLog

    session.evidence = EvidenceLog(
        None if no_session else project_folder(rlm_agent_home(), session.cwd) / "evidence.jsonl"
    )
    local = sessions_dir / f"{session.session_id}.playbook.json" if sessions_dir else None
    session.improver = tree.improver = Improver(
        session, local_path=local, shared_path=rlm_agent_home() / "playbook.json"
    )
    session.improver.auto = getattr(ns, "auto_improve", True)
    session.improver.hold_auto = getattr(ns, "review_auto_improve", True)
    session.improver.every = getattr(ns, "improve_every", 5)
    session.improver.cooldown = getattr(ns, "improve_cooldown", 900.0)
    session.on_compacted = lambda: (tree.tracer.compacted("root"), session.improver.compacted())
    session.improver.show_digest()

    max_turns = getattr(ns, "max_turns", None)
    if max_turns is not None:
        session.harness.settings.max_turns = max_turns

    _install_hooks(session.harness, verbose=verbose)
    _install_safety(session.harness, ApprovalPolicy.AUTO)
    from rlmagent.refinement.controller import attach

    try:
        attach(session, rlm_agent_home(), persist=not no_session)
    except Exception:
        await session.shutdown()
        raise
    attach_logger(session)
    return session


async def _resume_folder_problem(ns, session_id: str, sessions_dir) -> str | None:
    """Why this session must not resume here: it was started for another project."""
    from pathlib import Path

    from rlmagent.harness.session.index import SessionCatalog

    meta = SessionCatalog(sessions_dir).get(session_id)
    if meta is None or not meta.cwd or getattr(ns, "resume_anywhere", False):
        return None
    if Path(meta.cwd).resolve() == Path(os.getcwd()).resolve():
        return None
    text = f"Session {session_id} belongs to {meta.cwd}, not {os.getcwd()}."
    if not _is_interactive(ns):
        return f"{text} Run it from that folder, or pass --resume-anywhere."
    import asyncio

    answer = await asyncio.get_running_loop().run_in_executor(
        None, input, f"{text} Resume it here anyway? [y/N] ")
    return None if answer.strip().lower().startswith("y") else "Resume cancelled."


def make_agent_tree(ns, provider, provider_name, model, sessions_dir, *, cwd, backend=False):
    """Build the tree of agents a run uses, with the limits from the command line."""
    from rlmness import Allowance

    from rlmagent.agents.prompts import child_first_message, child_system
    from rlmagent.agents.tree import AgentTree
    from rlmagent.cli.main import _resolve_system

    allowance = Allowance(
        max_depth=getattr(ns, "max_depth", 2),
        max_calls=getattr(ns, "max_calls", None) or (50 if str(model).startswith("acp:") else 200),
        max_cost=getattr(ns, "max_cost", 2.0),
        max_live=getattr(ns, "max_live", 8),
        max_seconds=getattr(ns, "max_seconds", None),
    )

    from rlmagent.agents.approval import Approver
    from rlmagent.kernel.wasm import available

    kind = getattr(ns, "kernel", "local")
    if kind == "wasm" and not available():
        print("rlm-agent: the WebAssembly sandbox is not set up (run `rlmness --setup`); "
              "using the local kernel.", file=sys.stderr)
        kind = "local"
    network = getattr(ns, "network", True)
    if getattr(ns, "auto_approve", False):
        policy = "auto"
    elif getattr(ns, "print_mode", False) or getattr(ns, "prompt", None) is not None:
        policy = "deny"
    else:
        policy = "ask"

    def system_for(node, tools):
        base = _resolve_system(None, tools=tools, skills=[])
        return child_system(base, node, node.depth < allowance.max_depth, sandboxed=kind == "wasm",
                            network=network)

    if backend:
        from rlmagent.backend_context import BackendContext
        tree_type = BackendContext
    else:
        tree_type = AgentTree
    tree = tree_type(
        provider=provider,
        provider_name=provider_name,
        model=model,
        cwd=cwd,
        allowance=allowance,
        sessions_dir=sessions_dir,
        system_for=system_for,
        first_message_for=child_first_message,
        cell_timeout=getattr(ns, "cell_timeout", 300.0),
        max_agents=getattr(ns, "max_agents", 50),
        kernel=kind,
        restore_kernel=getattr(ns, "restore_kernel", False),
        approver=Approver(policy),
        network=network,
    )
    tree.allow_private_fetch = getattr(ns, "allow_private_fetch", False)
    return tree


def _is_interactive(ns: argparse.Namespace) -> bool:
    """A human is at a terminal and this run is not one-shot or piped."""
    import sys

    if getattr(ns, "print_mode", False) or getattr(ns, "prompt", None) is not None:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


async def settle_project_trust(
    ns: argparse.Namespace, *, ask: Asker | None = None,
) -> TrustOutcome:
    """Decide whether this folder's project inputs load, before any are read."""
    from rlmagent.cli.main import _die, _info, _warn
    from rlmagent.config.store import load_config
    from rlmagent.trust import StartupCancelled, resolve_project_trust

    override: bool | None = None
    if getattr(ns, "approve", False):
        override = True
    elif getattr(ns, "no_approve", False):
        override = False
    if ask is None and _is_interactive(ns):
        ask = ask_on_console
    default = load_config().project_trust or "ask"

    try:
        outcome = await resolve_project_trust(
            os.getcwd(), override=override, default=default, ask=ask,  # type: ignore[arg-type]
        )
    except StartupCancelled:
        _die("No project trust decision made; exiting.")
    if outcome.notice:
        _warn(outcome.notice)
    if not outcome.trusted and outcome.inputs is not None and not outcome.inputs.empty:
        _info(
            f"Project inputs not loaded ({outcome.inputs.describe()}): {outcome.reason}. "
            "Run with --approve to load them this once."
        )
    return outcome


def trust_choices(request: TrustRequest) -> list[tuple[Choice, str]]:
    from rlmagent.trust import Choice

    menu = [(Choice.TRUST_FOLDER, "Trust this folder")]
    if request.parent is not None:
        menu.append((Choice.TRUST_PARENT, f"Trust parent folder ({request.parent})"))
    menu += [
        (Choice.TRUST_ONCE, "Trust for this run only"),
        (Choice.DISTRUST_FOLDER, "Do not trust this folder"),
        (Choice.DISTRUST_ONCE, "Do not trust for this run only"),
    ]
    return menu


def trust_question(request: TrustRequest) -> str:
    """The explanation shown above the choices, shared by every frontend."""
    lines = [
        f"{request.folder} contains project inputs: {request.inputs.describe()}.",
        "They shape the agent's instructions and can add plugins that run as code.",
        "This controls project inputs; it is not a sandbox.",
    ]
    if request.store_problem:
        lines.append(f"Saved decisions are unavailable: {request.store_problem}")
    return "\n".join(lines)


async def ask_on_console(request: TrustRequest) -> Choice | None:
    """Ask on the terminal; EOF or Ctrl+C cancels."""
    import asyncio
    import sys

    menu = trust_choices(request)
    sys.stderr.write(trust_question(request) + "\n")
    for number, (_choice, label) in enumerate(menu, start=1):
        sys.stderr.write(f"  {number}. {label}\n")
    loop = asyncio.get_running_loop()
    while True:
        try:
            answer = await loop.run_in_executor(None, input, f"Choose 1-{len(menu)}: ")
        except (EOFError, KeyboardInterrupt):
            return None
        answer = answer.strip()
        if answer.isdigit() and 1 <= int(answer) <= len(menu):
            return menu[int(answer) - 1][0]


def attach_logger(session: CodingSession) -> None:
    """Subscribe a ``RunLogger`` to the session's event stream."""
    from rlmagent.logs import RunLogger

    logger = RunLogger.create(
        session.session_id,
        provider=session.provider_name,
        model=session.model,
        cwd=session.cwd,
    )
    if logger is None:
        return
    logger.stats_source = lambda: session.usage
    session.harness.on_event(logger.record_event)


def git_branch(cwd: str | None = None) -> str | None:
    """Current git branch name, or ``None`` outside a repository."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    branch = out.stdout.strip()
    return branch or None


__all__ = [
    "ask_on_console",
    "build_session",
    "git_branch",
    "settle_project_trust",
    "trust_choices",
    "trust_question",
]
