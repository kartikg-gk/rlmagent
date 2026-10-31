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
`--no-network` disables the sandbox network setup. With networking enabled, use
async httpx; blocking requests-based libraries are not supported by this host.

The backend exposes `rlm(piece, instruction=question)` and `gather_rlm(pieces, ...)`.
Background work uses `await spawn(task, data, keep=False, role=None, model=None)`.
Handles provide `result`, `status`, `send`, `messages`, `cancel`, and `id`.
Persistent agents use `keep=True`; they retain saved variables between questions.
`await children()` lists handles and `await tell_parent(text)` sends a child update.
Depth, call, cost, live-agent, and time limits apply across recursive execution.

## Workspace approvals

Sandbox code reaches the project through `read_file`, `list_files`, `write_file`,
`edit_file`, and `run`. File paths are confined to the configured workspace.
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
checkpoint and does not rewind backend variables.

Use `await playbook(id)` to inspect a saved entry, `await skill(id)` to load a skill,
and `await improve(instructions, shared=False)` to queue a playbook update at the
turn boundary. Compaction also schedules the existing automatic improvement check.
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

### Remote models and cloud credentials

In the model picker, Ctrl+F fetches remote models, Ctrl+T checks credentials, and
Ctrl+E sets a compatible API base URL. The model list also accepts a custom model
identifier. The CLI offers `rlm-agent provider models openrouter` and
`rlm-agent provider test openrouter`. A successful model-list check establishes
access to that endpoint; it does not prove permission to generate with every model.

For Google Vertex, install `rlmagent[vertex]`, configure Application Default
Credentials, and set `GOOGLE_CLOUD_PROJECT` and optionally `GOOGLE_CLOUD_LOCATION`
(default `us-central1`). Select `--provider vertex --model google/MODEL_ID`.
Credentials refresh when needed and remain in memory. Vertex credential checks
verify ADC refresh; remote model discovery is unavailable for this endpoint, so
enter a compatible model identifier. Model access still requires a live completion.

Workspace commands execute through non-login Bash, with a default 120-second timeout and a maximum of 1,800 seconds.
