"""ApplyPatch: several edits across files from one unified diff, all of them or none."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rlmagent.tools._shared import (
    CancelToken,
    JValue,
    ProgressNotifier,
    ToolOutcome,
    ToolSpec,
    fault,
    require_str,
    text_outcome,
)

_PARAMS: Mapping[str, JValue] = {
    "type": "object",
    "properties": {
        "patch": {
            "type": "string",
            "description": (
                "A unified diff with ---/+++ headers and @@ hunks. Paths are relative to the "
                "working directory; /dev/null creates or deletes a file."
            ),
        },
    },
    "required": ["patch"],
    "additionalProperties": False,
}


async def _run_patch(
    tool_call_id: str,
    arguments: Mapping[str, JValue],
    signal: CancelToken | None = None,
    on_update: ProgressNotifier | None = None,
) -> ToolOutcome:
    from rlmagent.agents.approval import Approver
    from rlmagent.agents.patch import PatchError
    from rlmagent.agents.workspace import Workspace

    try:
        diff = require_str(arguments, "patch")
        # Approval for this tool happens before it runs, like Write and Edit.
        return text_outcome(await Workspace(Path.cwd(), Approver("auto")).apply_patch(diff))
    except (PatchError, ValueError, OSError) as exc:
        return fault(f"Patch not applied: {exc}")


def make_patch_tool() -> ToolSpec:
    return ToolSpec(
        name="ApplyPatch",
        label="Apply Patch",
        description=(
            "Apply a unified diff that may change, create or delete several files. Every hunk "
            "is checked first; if any does not match, no file is changed."
        ),
        parameters=_PARAMS,
        run=_run_patch,
        parallelism="sequential",
        format_call=lambda a: f"{str(a.get('patch', '')).count('+++ ')} files",
    )


__all__ = ["make_patch_tool"]
