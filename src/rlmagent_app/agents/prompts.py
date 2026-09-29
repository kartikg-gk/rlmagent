"""What each agent is told about its kernel, its sub-agents and its limits."""

from __future__ import annotations

import json

from rlmagent_app.agents.tree import AgentNode

KERNEL_GUIDE = """## Your kernel
The python tool runs code in one kernel that stays alive for the whole task: \
variables, imports and functions you define remain available in later calls. \
Its working directory is the project root. Run shell commands with a leading `!` \
or with subprocess. Print summaries (lengths, counts, a few lines) rather than \
whole files or large objects; long output is cut to its start and end."""

DELEGATION_GUIDE = """## Sub-agents
Inside the kernel you can hand work to sub-agents, which are agents like you \
with their own kernel and the same tools:

    answer  = await rlm("task for the sub-agent", context)
    answers = await gather_rlm([("task A", data_a), ("task B", data_b)])

- `context` can be any JSON-compatible value. The sub-agent receives it as the \
variable CONTEXT in its own kernel, so large data never has to pass through a \
message. Slice or filter it first: send each sub-agent only what its task needs.
- Each call returns the value the sub-agent passed to FINAL, as a normal Python \
object you can keep working with.
- Delegate when pieces are independent (the same check across many files, \
several separate questions) or when reading everything yourself would crowd \
out the rest of the task. Do small things yourself; a sub-agent costs a full \
agent run.
- If the user asks you to use sub-agents, you must call rlm, gather_rlm or spawn; \
do not do the work yourself instead.
- gather_rlm runs its jobs at the same time and returns results in order. If \
one fails, the others are stopped and the call raises.

To keep working while a sub-agent runs, start it in the background:

    h = await spawn("run the test suite and report failures", data)
    ...                       # carry on with other cells
    report = await h.result() # waits; h.result(timeout=10) raises TimeoutError if not done
    await h.status()          # starting, running, idle, done, failed or cancelled
    await h.cancel()          # stops it and everything it started
    await h.messages()        # notes it left for you
    hs = await children()     # every sub-agent you started

- Use spawn for long work you do not need yet or may want to stop; use \
gather_rlm for independent pieces you need all of now.
- spawn(task, data, keep=True) keeps the sub-agent after its FINAL, idle, so \
h.send("next instruction") can give it more work; h.result() then returns its \
newest FINAL.
- When you finish, sub-agents you started are stopped; collect what you need \
first."""

LEAF_NOTE = "Sub-agents are not available to you: rlm, gather_rlm and spawn are not defined in your kernel. Do this task yourself."

CHILD_ROLE = """## Your role
You were started by another agent to do one task. Its data, if any, is in the \
variable CONTEXT in your kernel. When you are done, call FINAL(value) in the \
python tool with the result it asked for; that value is returned to it directly. \
Call FINAL exactly once. To report progress before then, await tell_parent("note")."""

RESUMED_NOTICE = (
    "This session was resumed. The kernel is new: variables, imports and functions "
    "from earlier in the conversation are gone. Rebuild anything you still need."
)

COMPACTION_NOTE = "The kernel is still running: variables defined before this summary are still available."


PLAYBOOK_GUIDE = """## Playbook
You keep a playbook of rules, memories, skills and sub-agent roles across \
sessions. When something is worth keeping (a correction from the user, a \
mistake you repeated, a procedure you did twice, a lasting fact), call \
`await improve("what to keep")` in the python tool; the playbook is updated \
when the turn ends. Use `shared=True` only for what every future project \
should know. `await playbook()` lists entries."""


def root_system(base: str, can_delegate: bool, playbook: bool = False) -> str:
    parts = [base, KERNEL_GUIDE, DELEGATION_GUIDE if can_delegate else LEAF_NOTE]
    if playbook:
        parts.append(PLAYBOOK_GUIDE)
    return "\n\n".join(parts)


def child_system(base: str, node: AgentNode, can_delegate: bool) -> str:
    parts = [base, KERNEL_GUIDE, CHILD_ROLE, DELEGATION_GUIDE if can_delegate else LEAF_NOTE]
    return "\n\n".join(parts)


def context_preview(value: object, limit: int = 600) -> str:
    if value is None:
        return "CONTEXT is None (no data was passed)."
    if isinstance(value, str):
        kind = f"a string of {len(value):,} characters"
    elif isinstance(value, (list, tuple)):
        kind = f"a list of {len(value):,} items"
    elif isinstance(value, dict):
        kind = f"a dict with keys {list(value)[:20]}"
    else:
        kind = f"a {type(value).__name__}"
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    head = text[:limit] + (" ..." if len(text) > limit else "")
    return f"CONTEXT is {kind}. It begins:\n{head}"


RETURN_REMINDER = (
    "Deliver your answer by calling FINAL(value) in the python tool. "
    "A plain text reply is not returned to the agent that asked."
)


def child_first_message(node: AgentNode) -> str:
    return f"{node.task}\n\n{context_preview(node.context)}\n\n{RETURN_REMINDER}"
