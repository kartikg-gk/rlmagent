"""An isolated model conversation with two non-executable harness operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from pydantic import Field

from rlmagent.harness.contracts.tooling import ToolOutcome, ToolSpec
from rlmagent.harness.contracts.transcript import HumanEntry, TextSegment, ToolOutcomeEntry
from rlmagent.harness.provider.wire import StreamCloseEvent, StreamFaultEvent
from rlmagent.refinement.store import ChangeNote, Guidance, Record, Snapshot

STEP_LIMIT = 12
EDIT_LIMIT = 8
INPUT_LIMIT = 80_000
POLICY = """Review the supplied task record for reusable guidance. You have no execution
environment and cannot modify the project, installed skills, or base instructions.
First read the editable harness with guidance_inspect. Propose narrowly supported
changes through guidance_edit. A workflow needs a complete Markdown document with
YAML name and description fields. Use prompt for concise behavioral guidance,
skill for a repeatable procedure, and subagent for delegation instructions.
Avoid preserving secrets, incidental task output, or guesses. An empty change set
is acceptable. Finish with a JSON object containing summary, evidence, and expected.
Expected describes a prediction, not a measured performance improvement.
For text-only model adapters, request one operation per reply as JSON:
{"tool":"guidance_inspect","arguments":{}} or
{"tool":"guidance_edit","arguments":{"action":"create","entry":
{"id":"check-build","kind":"prompt","title":"Build check","content":"..."}}}.
Use action update with a complete replacement entry; delete uses id instead.
Do not enclose JSON in a code fence. No other operations are available."""


class Verdict(Record):
    summary: str = Field(min_length=1, max_length=2000)
    evidence: str = Field(min_length=1, max_length=8000)
    expected: str = Field(min_length=1, max_length=2000)


@dataclass
class Candidate:
    baseline: Snapshot
    state: Snapshot
    changed: bool


class Draft:
    def __init__(self, baseline: Snapshot):
        self.baseline = baseline
        self.state = baseline.model_copy(deep=True)
        self.inspected = False
        self.edits: list[dict] = []
        self.attempts = 0

    def inspect(self, arguments: dict):
        if set(arguments) - {"id"}:
            raise ValueError("Inspection accepts only an optional id")
        self.inspected = True
        if "id" in arguments:
            return self.state.entries[arguments["id"]].model_dump()
        return {
            "revision": self.state.revision,
            "entries": [
                {"id": item.id, "kind": item.kind, "title": item.title}
                for item in self.state.entries.values()
            ],
            "recent_changes": [item.model_dump() for item in self.state.history[-5:]],
        }

    def edit(self, arguments: dict):
        if not self.inspected:
            raise ValueError("Inspect the editable state before staging changes")
        self.attempts += 1
        if self.attempts > EDIT_LIMIT:
            raise ValueError("The edit allowance for this review is exhausted")
        next_state = self.state.model_copy(deep=True)
        action = arguments.get("action")
        if action == "delete":
            if set(arguments) != {"action", "id"}:
                raise ValueError("Delete requires action and id")
            entry_id = arguments["id"]
            if entry_id not in next_state.entries:
                raise ValueError("Cannot remove absent guidance")
            del next_state.entries[entry_id]
        elif action in ("create", "update"):
            if set(arguments) != {"action", "entry"}:
                raise ValueError("Create and update require action and entry")
            entry = Guidance.model_validate(arguments["entry"])
            exists = entry.id in next_state.entries
            if exists != (action == "update"):
                raise ValueError("Create needs a new id; update needs an existing id")
            next_state.entries[entry.id] = entry
            entry_id = entry.id
        else:
            raise ValueError("Unsupported guidance operation")
        self.state = Snapshot.model_validate(next_state.model_dump())
        self.edits.append({"action": action, "id": entry_id})
        return {"staged": len(self.edits), "id": entry_id}

    def finish(self, value: dict) -> Candidate:
        verdict = Verdict.model_validate(value)
        if not self.inspected:
            raise ValueError("Review ended without inspecting the harness")
        changed = self.state.entries != self.baseline.entries
        if changed:
            self.state.revision += 1
            self.state.history.append(
                ChangeNote(id=uuid4().hex, **verdict.model_dump(), edits=self.edits)
            )
        return Candidate(self.baseline, self.state, changed)


def evidence_text(messages) -> str:
    """Keep recent structured tool calls and outcomes, not just visible chat text."""
    rows = []
    length = 0
    for message in reversed(messages):
        row = json.dumps(message.model_dump(mode="json"), ensure_ascii=False)
        if length + len(row) > INPUT_LIMIT:
            if not rows:
                rows.append(row[-INPUT_LIMIT:])
            break
        rows.append(row)
        length += len(row) + 1
    return "\n".join(reversed(rows))


async def review(provider, model, baseline, evidence, focus, signal, progress) -> Candidate:
    draft = Draft(baseline)

    async def unused(*args, **kwargs):
        return ToolOutcome(content="Operations are dispatched by the review controller")

    entry_schema = Guidance.model_json_schema()
    specs = [
        ToolSpec(
            name="guidance_inspect",
            label="Inspect guidance",
            description="Read the saved guidance or one entry.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "additionalProperties": False,
            },
            run=unused,
        ),
        ToolSpec(
            name="guidance_edit",
            label="Stage guidance",
            description="Stage one create, update, or delete against editable guidance.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "update", "delete"]},
                    "entry": entry_schema,
                    "id": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            run=unused,
        ),
    ]
    messages = [HumanEntry(content=json.dumps({"trajectory": evidence, "focus": focus}))]
    operations = {"guidance_inspect": draft.inspect, "guidance_edit": draft.edit}
    for step in range(STEP_LIMIT):
        progress(f"reviewing step {step + 1}/{STEP_LIMIT}")
        if signal.is_cancelled():
            raise RuntimeError("Review cancelled")
        reply = None
        async for event in provider.stream_response(
            model=model, system=POLICY, messages=messages, tools=specs, signal=signal
        ):
            if isinstance(event, StreamFaultEvent):
                raise RuntimeError(event.error.error_message or "Refinement provider failed")
            if isinstance(event, StreamCloseEvent):
                reply = event.message
        if reply is None:
            raise ValueError("Refinement provider returned no final response")
        messages.append(reply)
        calls = [(call.name, call.arguments, call.id) for call in reply.tool_calls]
        if not calls:
            value = json.loads(reply.text)
            if not isinstance(value, dict):
                raise ValueError("Review response must be a JSON object")
            if "tool" not in value:
                return draft.finish(value)
            if set(value) != {"tool", "arguments"} or not isinstance(value["arguments"], dict):
                raise ValueError("Malformed review operation")
            calls = [(value["tool"], value["arguments"], None)]
        for name, arguments, call_id in calls:
            if signal.is_cancelled():
                raise RuntimeError("Review cancelled")
            progress(f"running {name}")
            try:
                if name not in operations:
                    raise ValueError("This review cannot execute that operation")
                result = operations[name](arguments)
                content = json.dumps(result, ensure_ascii=False)
                failed = False
            except (ValueError, KeyError, TypeError) as error:
                content, failed = str(error), True
            if call_id is None:
                messages.append(
                    HumanEntry(
                        content=json.dumps({"operation": name, "error": failed, "result": content})
                    )
                )
            else:
                messages.append(
                    ToolOutcomeEntry(
                        tool_call_id=call_id,
                        tool_name=name,
                        content=[TextSegment(text=content)],
                        is_error=failed,
                    )
                )
    raise ValueError("Review reached its model-step limit without a final decision")
