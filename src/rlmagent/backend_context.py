"""Frontend metadata and policies for a backend-owned execution tree."""
from types import SimpleNamespace

from rlmagent.agents.approval import Approver
from rlmagent.agents.budget import BudgetedProvider
from rlmagent.agents.trace import TracedProvider, Tracer
from rlmagent.model.external import with_external


class BackendContext:
    def __init__(self, *, provider, provider_name, model, cwd, allowance,
                 sessions_dir, kernel="wasm", approver=None, network=False,
                 cell_timeout=300, max_agents=50, tracer=None, **metadata):
        self.provider = BudgetedProvider(with_external(provider), allowance)
        self.provider_name, self.model, self.cwd = provider_name, model, cwd
        self.allowance, self.sessions_dir = allowance, sessions_dir
        self.kernel_kind, self.network = kernel, network
        self.approver = approver or Approver("deny")
        self.cell_timeout, self.max_agents = cell_timeout, max_agents
        self.tracer = tracer or Tracer(None)
        self.improver = None
        self.records, self.backend_agents = {}, {}
        self.nodes = {"root": SimpleNamespace(context=None)}
        self._finished = False
        self.__dict__.update(metadata)

    async def start(self):
        pass

    def root_id(self):
        return "root"

    def root_kernel(self):
        # Tool metadata identifies the coding route. The solver owns execution.
        return self

    async def run(self, code, timeout=None):
        raise RuntimeError("submit code through the backend solver")

    def attach(self, agent, session):
        self.session = session

    def provider_for(self, agent):
        return TracedProvider(self, agent)

    def use_provider(self, provider, provider_name):
        self.provider = BudgetedProvider(with_external(provider), self.allowance)
        self.provider_name = provider_name
        return self.provider

    def _role(self, name):
        if self.improver is None:
            raise RuntimeError("playbook unavailable")
        entry = self.improver.entry(name)
        if entry['kind'] != 'role':
            raise ValueError("entry is not a role")
        return entry['content']

    def child_session_ids(self, parent):
        return []

    def finish_trace(self, complete=True):
        if self._finished:
            return None
        self._finished = True
        agents = [{"agent": key, "parent": value.get("parent_run_id"),
                   "task": value.get("instruction"), "status": value.get("status"),
                   "session": getattr(getattr(self, "session", None), "session_id", None)
                   if key == "root" else None}
                  for key, value in self.backend_agents.items()]
        return self.tracer.finish(agents, self.provider.refusal, complete)

    async def close(self):
        self.finish_trace()
