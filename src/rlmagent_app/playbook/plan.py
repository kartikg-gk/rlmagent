"""Ask the model what to change in the playbook, and read its answer."""

from __future__ import annotations

import json
import re

from rlmagent_app.playbook.store import Entry

CONVERSATION_CHARS = 60_000

SYSTEM = """You maintain an agent's playbook: short entries the agent reads in later \
turns and later sessions. Read the conversation and decide whether anything in it \
is worth keeping. Most of the time the right answer is no change.

Entry kinds:
- rule: a narrow instruction for how to work ("run the tests before saying done").
- memory: a fact, decision, preference or outcome worth remembering.
- skill: a Python function the agent can reuse; `content` is its full source and \
`callable` the name of the function it defines.
- role: instructions for a kind of sub-agent the agent keeps starting.

Only keep what the conversation shows: a repeated mistake, a correction from the \
user, a procedure done more than once, a lasting fact. Prefer one small edit to \
many; prefer updating an entry to adding a near-duplicate; delete entries shown to \
be wrong. Never propose changes to source files.

Answer with one JSON object and nothing else:
{"summary": "one sentence", "rationale": "the evidence", "expected": "what should improve",
 "edits": [{"action": "create|update|delete", "kind": "rule|memory|skill|role", "id": "short-id",
            "title": "...", "content": "...", "callable": "only for skills"}]}
An empty "edits" list is a good answer when nothing is worth keeping."""

SCOPES = {
    "local": "These edits apply to this session only. Shared entries are shown for context "
             "and are read-only: to change one for this session, create a local entry.",
    "global": "These edits are shared by every future session. Only keep what will still be "
              "true later: user preferences, reusable skills and roles, facts about tools "
              "or about a named project.",
}


CHECK_CHARS = 30_000

CHECK_SYSTEM = """You decide whether an agent's recent conversation holds anything worth \
keeping in its playbook: a correction from the user, a mistake made more than once, a \
procedure done twice, a lasting fact or preference. Routine work, one-off details and \
guesses are not worth keeping. Answer with one JSON object and nothing else:
{"worth_it": true or false, "why": "one sentence", "focus": "what to keep, if anything"}"""

AUTO_NOTE = (
    "This update was started automatically. No change is a good answer. Keep only what "
    "the conversation clearly shows. Nothing goes to the shared playbook unless the user "
    "asked for it."
)


def parse_check(text: str) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        answer = {}
    return bool(answer.get("worth_it")), str(answer.get("focus") or "")


def build_request(entries: list[Entry], history: list[dict], conversation: str, *,
                  scope: str, instructions: str | None) -> str:
    if len(conversation) > CONVERSATION_CHARS:
        conversation = "[earlier conversation omitted]\n" + conversation[-CONVERSATION_CHARS:]
    state = "\n".join(
        f"- [{e.kind}] {e.id} (v{e.version}, {e.scope}): {e.title}\n  {e.content}" for e in entries
    ) or "(empty)"
    past = "\n".join(f"- {h['id']}: {h.get('summary', '')}" for h in history[-10:]) or "(none)"
    parts = [
        f"<playbook>\n{state}\n</playbook>",
        f"<earlier_changes>\n{past}\n</earlier_changes>",
        f"<scope>\n{SCOPES[scope]}\n</scope>",
        f"<conversation>\n{conversation}\n</conversation>",
    ]
    if instructions:
        parts.append(f"<instructions>\n{instructions}\n</instructions>")
    return "\n\n".join(parts)


def parse_proposal(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("the planner's answer contained no JSON object")
    try:
        proposal = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"the planner's answer is not valid JSON: {exc}") from None
    if not isinstance(proposal.get("edits"), list):
        raise ValueError("the planner's JSON has no `edits` list")
    return proposal
