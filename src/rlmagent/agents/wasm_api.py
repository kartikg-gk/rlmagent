"""Code run in every sandbox before the model's first cell: the same functions the
local kernel has, reached through the `_rlmagent` bridge instead of a socket."""

CORE = r'''
async def improve(instructions=None, shared=False):
    """Ask for the playbook to be updated from this conversation when the turn ends."""
    return await _rlmagent("improve", {"instructions": instructions, "shared": bool(shared)})


async def playbook(id=None):
    """One playbook entry in full, or a list of all of them."""
    return await _rlmagent("playbook", {"id": id})


async def skill(id):
    """Load a playbook skill and return its function."""
    entry = await playbook(id)
    if entry["kind"] != "skill":
        raise ValueError(f"{id} is a {entry['kind']}, not a skill")
    namespace = {}
    exec(compile(entry["content"], f"<skill {id}>", "exec"), namespace)
    return namespace[entry["callable"]]
'''

DELEGATE = r'''
async def rlm(task, context=None, role=None, model=None):
    """Hand `task` to a sub-agent; `context` arrives in its sandbox as CONTEXT. Returns its FINAL value."""
    return await _rlmagent("rlm", {"task": str(task), "context": context, "role": role, "model": model})


async def gather_rlm(jobs):
    """Run several sub-agents at once. `jobs` is a list of (task, context) pairs or tasks. Results keep order."""
    pairs = [list(j) if isinstance(j, (tuple, list)) else [j, None] for j in jobs]
    return await _rlmagent("gather", {"jobs": [[str(t), c] for t, c in pairs]})


class _SubAgent:
    """A sub-agent started with spawn(). Losing this object does not stop it."""

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"<sub-agent {self.id}>"

    async def status(self):
        return await _rlmagent("status", {"id": self.id})

    async def result(self, timeout=None):
        return await _rlmagent("result", {"id": self.id, "timeout": timeout})

    async def cancel(self):
        return await _rlmagent("cancel", {"id": self.id})

    async def send(self, text):
        return await _rlmagent("send", {"id": self.id, "text": str(text)})

    async def messages(self):
        return await _rlmagent("messages", {"id": self.id})


async def spawn(task, data=None, keep=False, role=None, model=None):
    """Start a sub-agent and return at once. keep=True: it stays idle after FINAL, ready for send()."""
    return _SubAgent(await _rlmagent("spawn", {"task": str(task), "context": data,
                                               "keep": bool(keep), "role": role, "model": model}))


async def children():
    """Every sub-agent you started, in order."""
    return [_SubAgent(i) for i in await _rlmagent("children", {})]
'''

CHILD = r'''
async def tell_parent(text):
    """Leave a note the agent that started you can read with messages()."""
    await _rlmagent("tell_parent", {"text": str(text)})


CONTEXT = await _rlmagent("context", {})
'''


NETWORK = r'''
import sys as _sys
for _m in ("js", "pyodide_js", "pyodide.http", "pyodide.ffi.wrappers"):
    _sys.modules.pop(_m, None)
del _sys, _m
import pyodide_js
try:
    await pyodide_js.loadPackage("micropip")
    import micropip
    await micropip.install(["httpx"])
except Exception as _e:
    print(f"[network set-up failed: {_e}]")
'''


def wasm_api(can_delegate: bool, is_child: bool, network: bool = False) -> str:
    return ((NETWORK if network else "") + CORE + (DELEGATE if can_delegate else "")
            + (CHILD if is_child else ""))
