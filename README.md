# rlmagent

A terminal coding agent with a persistent Python workspace and recursive agents.
The application supplies the terminal interface, editor protocol, workspace tools,
and approvals. The `rlmness` backend runs the model loop, Python cells, recursive
delegation, and shared budget.

## Install

Python 3.12 or newer is required. The default sandbox also requires Node.js.

```sh
pip install rlmagent
rlmness --setup
```

This revision requires `rlmness>=0.2.2`. When installing unreleased local builds,
install the matching backend wheel before the application wheel.

## Run

Configure an API provider, or use an installed and authenticated external CLI:

```sh
rlm-agent --provider cli --model 'cli:claude-code?model=haiku' "Inspect this project"
rlm-agent --provider cli --model 'cli:codex' "Fix the failing check"
rlm-agent --provider cli --model 'cli:claude-code?model=haiku' -p "Summarize the project"
```

Run `rlm-agent --help` for provider, model, session, and budget settings. Keys are
read from provider configuration or environment variables. Keep keys out of prompts.

## Execution

`--engine rlmness` is the default. Recursive agents execute through the backend;
the frontend does not run another recursive model loop around each backend step.
`--engine session` selects the previous engine for compatibility.

`--kernel wasm` uses separate Pyodide interpreters in a shared Node host.
`--kernel local` uses host Python with the backend engine; it has machine access.
The compatibility engine uses its previous IPython kernel for local execution.
`fetch_url` refuses local and private addresses, including through redirects, unless
`--allow-private-fetch` is passed. Sandbox code using httpx directly and approved
shell commands are not covered by this check.
`--no-network` disables the sandbox network setup. With networking enabled, use
async httpx; blocking requests-based libraries are not supported by this host.

The backend exposes `rlm(piece, instruction=question)` and `gather_rlm(pieces, ...)`.
Background work uses `await spawn(task, data, keep=False, role=None, model=None)`.
Handles provide `result`, `status`, `send`, `messages`, `cancel`, and `id`.
Persistent agents use `keep=True`; they retain saved variables between questions.
`await children()` lists handles and `await tell_parent(text)` sends a child update.
Depth, call, cost, live-agent, and time limits apply across recursive execution.

## Workspace approvals

Sandbox code reaches the project through `read_file` (whole files or a line range),
`list_files`, `search_files` (regex search of file contents), `write_file`,
`edit_file`, `apply_patch`, `ask_user`, and `run`. File paths are confined to the configured workspace.
Writes, edits, and commands use the application's approval policy. Shell commands
run with the user's permissions; the workspace path restriction is not an OS shell
sandbox. Command cancellation and timeout handling terminate the process tree.

Use `/auto` in line mode or Shift+Tab in the terminal UI to toggle automatic
approval for the session. `--auto-approve` enables it at launch. Headless runs deny
mutations by default. Approval checks bind commands and edit payloads to their content.

## State and playbooks

Saved frontend sessions have a matching `.backend.json` state file. Resume loads
the backend's saved values and supported definitions; unsupported objects are
reported as dropped. Ephemeral sessions retain state only while the app is open.
Live background jobs end with the application session; they are not restarted
automatically after process exit. `/undo` restores its filesystem/transcript
checkpoint and rolls back saved backend variables, unless they are too large
to copy; effects outside the workspace are not undone. `rlm-agent session prune
--older-than DAYS` lists old sessions and snapshots; add `--yes` to remove them.

Use `await playbook(id)` to inspect a saved entry, `await skill(id)` to load a skill,
and `await improve(instructions, shared=False)` to queue a playbook update at the
turn boundary. Compaction also schedules the existing automatic improvement check,
as do every 5 finished tasks (`--improve-every`) and any task with 3 or more failed
tool calls. A memory saved automatically is dropped 3 finished tasks later unless a
later edit keeps it, since it describes one moment of the work.
Contracts select their execution engine explicitly with `"engine": "rlmness"`.
Existing contracts without this field retain the previous session engine. Contract
settings take precedence over the normal CLI engine default.

Contracts can select a playbook explicitly:

```json
"playbook": {"file": "./playbook.json", "update": false}
```

The file is relative to the contract workspace. `update: false` allows reading
entries while rejecting changes; `true` enables updates and automatic checks.
Contracts that omit this section have no playbook.

The shared Node host can stall on a runaway cell. Recovery may reset every sandbox
in that host. The backend state is a saved namespace, not a complete process clone;
external effects are not rolled back or replayed by variable restoration.

## Editor and external agents

`rlm-agent --acp` serves the Agent Client Protocol over stdio. Configure the editor
to launch this command with the desired provider and model arguments. Approvals
and project trust are requested through the editor.

`cli:` and `acp:` models use external agents as model providers. For supported ACP
bridges, run `rlm-agent acp install` and inspect `rlm-agent acp status`.
ACP model usage may be unreported; use call and time caps as well as a cost cap.

## Development

Run tests one file at a time with `bash scripts/test_each.sh`. Failing files print
their complete diagnostics. The backend integration tests use actual sandbox
interpreters with scripted model replies and require the sandbox setup.


### Project tools and execution records

The default backend provides `await instruction_docs()` to discover project
instructions and skills, and `await instruction_docs(path)` to read a selected
file. Discovery excludes build, environment and version-control folders.
`await fetch_url(url, max_chars=50000)` returns bounded HTTP(S) text, the final
URL and a truncation flag. It is unavailable with `--no-network`.

Ordinary workspace errors return text beginning with `Error:` so a cell can
inspect and correct a failed operation. Approval refusals still stop the operation.
Write and edit approvals show a unified diff. A file changed while approval is
pending must be read again before a new change can be approved.

Saved sessions write `<session-id>.backend.events.jsonl` alongside their state.
These records include generated code, outputs, host-tool results, child lifecycle
and per-call usage. Contract runs write `<trace-file>.backend.events.jsonl`.
Usage totals preserve input, output, cache reads/writes and reasoning counts from
all recursive model calls. Providers that omit counts cannot supply those totals.

`rlm-agent session view [id]` replays those steps as cards, and
`rlm-agent session steps [id] [--json]` prints steps, tokens, cost and time per agent.

### Terminal UI

With the default engine each agent step is a card: depth, a short agent id, the step
number and its state, then the code, the first lines of output and, once it ends,
tokens, cost and time. Sub-agent cards are indented under their parent. Enter or a
click opens a card in full; F2 shows the agent tree, F3 a timeline of steps and F4
the background workers with their heartbeats and scheduled jobs. The status bar shows steps,
running agents, depth and spend against the run's limits. Messages typed while a
run is going are queued and sent when it ends.

`/diff` shows what the session changed in the project (`/diff last` for the last
turn, `/diff stat` for a file summary). Agents keep a task list with
`await todo("add", text)` and `await todo("done", n)` (a `todo` tool in the session
engine); it is saved with the session, shown in the sidebar and by `/tasks`. The
session engine also has `ApplyPatch` for multi-file unified diffs, plus `Grep` and `Glob`.

### Remote models and cloud credentials

In the model picker, Ctrl+F fetches remote models, Ctrl+T checks credentials, and
Ctrl+E sets a compatible API base URL. The model list also accepts a custom model
identifier. The CLI offers `rlm-agent provider models openrouter` and
`rlm-agent provider test openrouter`. A successful model-list check establishes
access to that endpoint; it does not prove permission to generate with every model.

For Google Vertex, install `rlmagent[vertex]`. Vertex support is experimental: mocked authentication checks pass,
but a live completion with a configured cloud account has not been verified.
Configure Application Default Credentials, and set `GOOGLE_CLOUD_PROJECT` and optionally `GOOGLE_CLOUD_LOCATION`
(default `us-central1`). Select `--provider vertex --model google/MODEL_ID`.
Credentials refresh when needed and remain in memory. Vertex credential checks
verify ADC refresh; remote model discovery is unavailable for this endpoint, so
enter a compatible model identifier. Model access still requires a live completion.

Workspace commands execute through non-login Bash, with a default 120-second timeout and a maximum of 1,800 seconds.

## Workers

A worker is a named agent that keeps running after the terminal closes. It is an
ordinary saved session, so `--resume` works on it once it has stopped.

```text
rlm-agent worker start nightly --auto-approve
rlm-agent worker send nightly "check the build"
rlm-agent worker schedule nightly "review open issues" --in 3600
rlm-agent worker schedule nightly "run the tests" --at 2027-01-02T09:00 --every 86400
rlm-agent worker list
rlm-agent worker jobs nightly
rlm-agent worker log nightly --follow
rlm-agent worker attach nightly
rlm-agent worker stop nightly
```

Start flags are the normal run flags. The folder's trust is decided when you start
it, because a worker has nobody to ask. For the same reason anything needing
approval is refused unless it was started with `--auto-approve`, and `ask_user`
has no answer. A job that was running when a worker died is not run again; a job
due while a worker was down runs when it next starts. Workers do not restart on
their own after a reboot.

## MCP servers

List Model Context Protocol servers in `mcp.json` in the rlm-agent home, or in
`.rlm-agent/mcp.json` of a trusted project:

```json
{"servers": {
  "files": {"command": "npx", "args": ["-y", "some-mcp-server"], "env": {"TOKEN": "${MY_TOKEN}"}},
  "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${DOCS_KEY}"}}
}}
```

`${NAME}` is read from the environment when a server starts, so keys stay out of the
file. Servers start on first use. Cells call `await mcp_tools()` and
`await mcp(server, tool, arguments)`; the session engine has matching tools. Calls
need approval unless the server marks the tool read-only. `/mcp` lists servers,
their state and tools; `/mcp NAME` starts one.

## Project extensions

A Python file in `.rlm-agent/extensions/` (or `~/.rlm-agent/extensions/`) with a
`register()` function is loaded at startup and by `/reload`. Project files load
only in a trusted folder. `register()` may return a mapping of names to functions,
a `ToolSpec`, or a list of those:

```python
def word_count(path):
    """Count the words in a file."""
    with open(path) as handle:
        return len(handle.read().split())

def register():
    return {"word_count": word_count}
```

A function becomes `await word_count(...)` in a cell, and a tool under `--engine
session`. It runs in the host with your permissions, outside the sandbox and outside
approval, so enforce any rule it needs in its own code. It cannot replace a built-in
name. `/tools` lists what is really available; in a cell, `await capabilities()`
reports the same along with the limits in force.

## Manual harness refinement

Use `/refine` when a conversation contains a correction or procedure worth
retaining. It starts a bounded background review and returns immediately:

```text
/refine Remember the repository's focused validation command
/refine status
/refine cancel
```

Only one review may be pending per session. The refiner can inspect saved guidance
and stage create, update, or delete operations. It cannot execute Python, run
commands, modify project files, or replace the base system instructions. A review
may make no changes. Its limits are 12 model steps, 8 edit attempts, and 300 seconds;
the normal provider and tree budget also apply.

There are three entry kinds:

| Kind | Purpose |
| --- | --- |
| `prompt` | A focused behavioral note |
| `skill` | A Markdown workflow with YAML `name` and `description` metadata |
| `subagent` | Guidance for assigning and executing delegated tasks |

Workflow metadata must match its stable entry id. Individual entries are limited
to 32 KiB. Existing Python-function playbook skills keep their original meaning.
Installed skill files are never overwritten by refinement.

`/refine evidence` shows, per saved change in this project: when it was saved, how
many later tasks actually sent it to the model, failed tool calls per task before and
after, and your verdict from `/refine rate <change> better|same|worse`. The counts
are signals rather than proof, because tasks differ.

A proposal that changes guidance waits for you: `/refine show` lists each added,
changed or removed entry with a diff of its text, `/refine apply` saves it and
`/refine cancel` drops it. An approved proposal waits for a turn boundary before it
is saved or activated; if the session is idle it commits immediately.
`/refine status` distinguishes a running review, a proposal awaiting activation,
completion, cancellation, and failure. Beginning the next submission or finishing
the current turn services that boundary. Updated guidance then applies to the
root and newly created child agents. Running children keep their existing prompt.
Notes over 400 characters appear shortened in the prompt, with the rest available
through `await guidance("entry-id")`.
Workflow names and descriptions appear in the prompt; the full procedure is
available through `await guidance("entry-id")` in Python or `/skill:entry-id`.

Saved guidance is project-scoped under
`~/.rlm-agent/harness/<project-path-hash>/state.json`, so later sessions in the
same workspace inherit it. Read/modify/write operations are locked across
processes and reject proposals based on superseded state. Invalid state is
reported rather than replaced with an empty harness. The saved history includes
the evidence, expected benefit, and edits for each committed proposal.

`--no-session` keeps this harness in memory. Plan mode prevents changes, and abort,
rewind, inference-route changes, or shutdown discard pending work. Contract runs
do not load this ambient harness or enable `/refine`.

This mechanism runs only when explicitly requested. Existing `/improve` and its
automatic playbook scheduling are separate. Proposals from the automatic playbook
check are held: `/improve pending` lists them, `/improve accept N` saves one and
`/improve reject N` drops it. `--apply-auto-improve` saves them at once instead. A committed refinement means its
structure and persistence checks passed; its expected benefit has not been
established by a performance benchmark.
