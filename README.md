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

Or install it as a tool with one line, which also sets up the sandbox when Node is
present (`RLMAGENT_VERSION` pins a version):

```sh
curl -fsSL https://raw.githubusercontent.com/kartikg-gk/rlmagent/main/scripts/install.sh | sh
```

```powershell
irm https://raw.githubusercontent.com/kartikg-gk/rlmagent/main/scripts/install.ps1 | iex
```

`rlm-agent config show` prints the saved provider and model, and any
`RLM_AGENT_PROVIDER`, `RLM_AGENT_MODEL`, `RLM_AGENT_API_KEY` or `RLM_AGENT_HOME`
set in the environment, which take precedence for a run. `rlm-agent config path`
prints where the file is. Providers include OpenAI, Anthropic, OpenRouter, Ollama,
Hugging Face, Google Gemini, Mistral, Azure OpenAI, Vertex and Codex subscription sign-in.

This revision requires `rlmness>=0.2.3`. When installing unreleased local builds,
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

For native subscription sign-in, install the optional credential-store and identity
verification dependencies, then complete browser consent:

```sh
pip install 'rlmagent[subscription]'
rlm-agent provider login codex
rlm-agent provider models codex
rlm-agent --provider codex --model MODEL_FROM_CATALOG "Inspect this project"
rlm-agent provider logout codex
```

Subscription access, refresh and identity tokens are kept in the platform credential
store, separately for each `RLM_AGENT_HOME`. Requests use the public Responses API
with server-side response storage disabled. Model availability and usage limits come
from the signed-in account. `provider test` checks the model endpoint; a completed
model response is required to verify inference access. `login --new-account` replaces
the selected account after validating its identity. Sign-out clears local tokens and
reports whether remote revocation succeeded.

The native Gemini provider reads `GOOGLE_API_KEY`; Mistral reads `MISTRAL_API_KEY`.
Azure reads `AZURE_OPENAI_API_KEY` and requires `AZURE_OPENAI_BASE_URL` ending in
`/openai/v1`. `/tier auto|default|flex|priority|off` changes subsequent requests when
the active provider supports a service tier.

`/fast on|off` selects the priority or default tier for the current model. Tier
preferences are saved per provider and model. Use `/model cycle add MODEL`,
`/model cycle remove MODEL`, and `/model cycle next` to maintain a provider-specific
rotation. `--offline` suppresses startup model discovery and update checks; an
explicitly requested model run still uses its provider.

Set `RLM_AGENT_CODEX_TRANSPORT=websocket` to reuse a Responses WebSocket per agent.
The default is HTTP. Socket cancellation closes that agent's connection. A lost
server context can be resent once before output begins; visible output is never
automatically replayed. `NODE_EXTRA_CA_CERTS` adds a PEM certificate bundle to
the platform and bundled TLS roots for HTTP and WebSocket connections.

This release focuses on Codex and Claude subscription accounts. For Claude, use
the installed official CLI:

```sh
rlm-agent provider login claude-subscription
rlm-agent --provider claude-subscription --model cli:claude-code "Inspect this project"
```

Claude authentication remains within that CLI. Additional provider integrations
are retained for compatibility; their release qualification is deferred:

- `pip install 'rlmagent[copilot]'` enables `--provider copilot`. Run
  `rlm-agent provider login copilot` through the official CLI first. This adapter
  uses the official SDK with its own tools and hooks disabled. Sign out with
  `/logout` in that CLI.
- `--provider claude-subscription --model cli:claude-code` uses the installed,
  unmodified CLI. `provider login claude-subscription` invokes its sign-in.
  This route does not extract subscription OAuth credentials.
- `--provider xai` reads `XAI_API_KEY`. The experimental `xai-subscription`
  route requires your own registered public OAuth client and compatible API
  entitlement. `provider login xai-subscription --client-registration FILE`
  accepts exactly `client_id`, `authorization_endpoint`, `token_endpoint`,
  `scope`, `resource`, and `redirect_port`. Endpoints must be provider-owned
  HTTPS URLs; `resource` is `null` or `https://api.x.ai/v1`, and `redirect_port`
  selects the registered loopback callback port (zero requests a dynamic port).
  There is no built-in registration or verified subscription entitlement for
  this experimental route. Keep client secrets and tokens out of this file.

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

End-to-end tasks give a real model real work and check the workspace it leaves, in
place, through a tool host over stdio, and over TCP. They cost money, so they run only
when asked:

```sh
RLM_AGENT_E2E=1 pytest tests/e2e -m e2e
```

`RLM_AGENT_E2E_PROVIDER` and `RLM_AGENT_E2E_MODEL` pick the model (default: the Claude
Code CLI with haiku).


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
rlm-agent worker schedule nightly "weekday build" --cron "0 9 * * 1-5" --timezone America/New_York
rlm-agent worker list
rlm-agent worker jobs nightly
rlm-agent worker log nightly --follow
rlm-agent worker attach nightly
rlm-agent worker recover
rlm-agent worker incidents nightly
rlm-agent worker watch
rlm-agent worker watch --stop
rlm-agent worker stop-all --wait 60
rlm-agent worker stop nightly
```

Cron schedules accept five numeric fields or `@hourly`, `@daily`, `@weekly`, and
`@monthly`. Lists, ranges, and steps work within fields. If both day fields are
set, both must match. The timezone defaults to UTC. Missing local times during a
daylight-saving jump are skipped, and repeated local times can run at both
occurrences. After downtime, one overdue run is taken and the next run follows
the wall-clock schedule.

`worker recover` restarts interrupted saved workers from their original launch
settings and session IDs. Pass a worker name to recover only that worker.
`worker incidents NAME` shows worker exits and fleet recovery attempts with the
saved session, last heartbeat, active turn, affected message and job IDs, and
restart outcome. A one-shot job interrupted by a worker exit remains visible in
`worker jobs` and is skipped on restart.
`worker watch` runs a fleet monitor in the foreground. It checks for interrupted
workers at startup and continues checking while it runs. Use `worker watch --stop`
from another terminal to stop the monitor.
`worker service install` registers and starts a watcher for the current app home
using Windows Task Scheduler, the Linux systemd user manager or a macOS launchd
agent. It starts again
when the user logs in. Use `worker service status`, `start`, `stop`, and
`uninstall` to manage it. The service uses the Python installation and session
directory from installation time. On Windows and macOS, watcher output goes to
`fleet-service.log` in the app home. On Linux, use the user journal. The macOS
agent is covered by tests of its definition and commands, not yet on a Mac.
`worker stop-all` lets the current turns finish and records the stop request
for interrupted workers, so recovery skips them.

Start a saved worker with `--idle-passivate 300` to close its process after
five idle minutes. With `worker watch` or the watcher service running, a queued
message or due job wakes the same saved session. `worker attach NAME` also wakes
it. An explicit stop suppresses automatic wakeup until you start the worker again.

Send `/goal <objective>` and then `/autonomous` to a saved worker session to
keep working after you detach. The worker saves its continuation and budgets
before starting the next turn. A queued turn survives a restart. If the worker
was killed during a turn, inspect the saved conversation and re-enable
`/autonomous` to continue.

Start flags are the normal run flags. The folder's trust is decided when you start
it, because a worker has nobody to ask. For the same reason anything needing
approval is refused unless it was started with `--auto-approve`, and `ask_user`
has no answer. A job that was running when a worker died is not run again; a job
due while a worker was down runs when it next starts. Workers do not restart on
their own after a reboot.

## Goals and autonomous mode

`/goal TEXT [--tokens N]` sets an objective that is put in front of the model on every
turn until it calls `await goal_done(summary)`; `/goal pause`, `resume` and `clear`
manage it. `/autonomous` then starts the next turn by itself while the goal is open,
and stops at its limits (`/autonomous turns N`, `tokens N`, `seconds N`), when the
goal is done, or when you press Esc. `/autonomous gate CMD` sets the default
verification command. Use `/autonomous gate add NAME CMD` for additional gates,
`/autonomous gate list` to see their latest status, and
`/autonomous gate remove NAME` to remove a named gate. Every configured command
must exit zero on the same `goal_done` attempt. A retry reruns all gates and
records each result separately.
Autonomous continuation also stops if the same failed operation, or the same
pair of failed turns, repeats three times. This check uses tool arguments and
root cell failures, so rephrasing the final reply does not reset it.

## Agents working together

Open sessions and workers can reach each other. In a cell: `await list_agents()`,
`await send_message(to, text)`, `await read_messages()`. A message to a worker runs as
its next task; a message to a session is shown to it before its next turn.
`await schedule(task, in_seconds=..., every=None, worker=None)` or
`await schedule(task, cron="0 9 * * 1-5", timezone="America/New_York")` queues work on a
worker, and `await heartbeat(text)` reports progress, shown in the terminal and in
`rlm-agent worker log`.

Other functions: `await web_search(query)` when `BRAVE_API_KEY`, `TAVILY_API_KEY` or
`SERPER_API_KEY` is set; `await context_usage()`; `await compact()` to summarize older
conversation when the turn ends. `/btw QUESTION` answers from the conversation so far; neither the
question nor the answer is kept in it.

`/image PATH` attaches a PNG, JPEG, GIF or WebP file (up to 5 MB) to your next
message. Use a model and provider that accept image input.

Press F1 for keyboard help. Focus a finished step and press Space to hide or
show its details, or Enter to open the full step.

To keep a saved session running between terminals, close the original terminal
session and start its detached process from the project folder:

```sh
rlm-agent session detach SESSION_ID
rlm-agent session attach SESSION_ID
rlm-agent session attach SESSION_ID --tui
rlm-agent worker stop session-SESSION_ID
```

Saved interactive sessions use the supervisor from the first turn.
Reconnect with `rlm-agent --resume SESSION_ID`. Use `--repl` for the line client,
`rlm-agent tui` to select Textual, or `--foreground` to run the interactive session
in the current process. `--supervised` explicitly selects the worker path.
A supervised worker shows approval requests and agent questions in
both clients. Follow the displayed `/answer REQUEST_ID VALUE` prompt to respond.
An unanswered request expires after 60 seconds. Approval responses are `once`,
`always`, `auto`, or `deny`. Questions accept text or a displayed option number.

Type `/detach` in the attached terminal to leave the process running. A detached
session uses the same unattended approval rules as a worker. Start detachment
between turns. A running turn cannot move between processes.
Saved sessions have a single active writer across terminal and worker processes.
The Textual attachment replays the saved conversation from the start of
detached execution and follows new prompts, streamed replies, and tool activity.
Multiple terminals can attach to the same worker and send prompts. Press Ctrl+D
to close the Textual client and leave the worker running. Ctrl+C interrupts the
current worker turn, or closes the client if the worker is idle.

A kept child in a saved session accepts `receipt = await handle.send(text)`
while it is busy. Use `await handle.delivery(receipt)` to inspect its status.
Kept children process these messages in order as follow-up turns. Queued
messages resume on the next parent run after reopening the session. A message
already running at the time of a crash is marked interrupted.
A detached supervisor restarts a crashed worker from its saved session, with
backoff. An interrupted turn is reported on resume.

On a saved session with the default engine, agents created with
`spawn(..., keep=True)` by the root or another kept agent retain their ID,
last completed result, and supported
kernel values across reopening. Find them with `children()` and send a new task
when the child is idle. In-flight child work is not replayed after a restart.

Use `await name_child(handle, "reviewer")` to name a child under its parent,
then `await child("reviewer")` to recover its handle. `await child_info(handle)`
shows status, pending deliveries, and recorded model usage without consuming the
result or messages. In the backend engine, names and usage for kept children
survive reopening the saved session.

Use `await steer_child(handle, text, message_id=None)` to add an instruction
at an active child's next model request. It returns a receipt, also available
through `child_info`. Reusing the message ID with the same text and child
returns the original receipt. If the active turn ends before another model
request, the receipt expires. Use `send()` to queue a separate follow-up turn.

For a protected HTTP MCP server, configure an `oauth` object beside its `url`
with `issuer`, a registered public `client_id`, and `redirect_port`. Register
`http://127.0.0.1:PORT/oauth/callback` with the issuer. Add `scope` if needed.
Use `/mcp login NAME` to sign in and `/mcp logout NAME` to disconnect locally.
Access tokens refresh when the server issues a refresh token. To reuse a login
after reopening, install `rlmagent[mcp-keyring]` and set `"persist": true` in
the OAuth object to use your platform credential store. With persistence
disabled, credentials live in the current process.

Use `"connector": "slack"` or `"connector": "notion"` instead of a server URL
to use a built-in endpoint. Supply your registered OAuth client details. For
confidential clients, set `client_secret_env` to the name of the environment
variable that holds the secret.

`rlm-agent factory workflow.json` runs a sequence of contract stages. Each
stage needs its own trace file. A failed stage stops the workflow. On success,
its final answer is included as data in the next stage's prompt.

```sh
rlm-agent factory init my-review contract.json review
rlm-agent factory list
rlm-agent factory export my-review workflow.json
rlm-agent factory import my-other-review workflow.json
rlm-agent factory run my-review --state review-state.json
```

The bundled recipes are `build`, `review`, and `release-review`. Run a saved
factory from a cell with `await factory_run("my-review")`, subject to workspace
command approval. Reuse the same `--state` file to continue after completed
stages. If a stage was interrupted or failed, review its workspace changes
before explicitly adding `--retry-stage`.
For a stage that needs to keep one agent and its kernel through several turns,
use `"resident": ["first prompt", "next prompt"]` in place of `prompt`. The
whole group shares one contract and counts as one checkpoint.

For a dependency graph, use `"format": "rlmagent.factory/2"` with a
`stages` object keyed by stage name. Each stage has a `contract`, a
`prompt`, and an optional `after` list. A stage normally requires all its
dependencies to succeed. Set `when` to `any_failed` for a recovery stage or
`always` to run after dependencies settle. `parallel` controls concurrency,
and stages with overlapping workspace paths are serialized.

Set a stage's `foreach` field to the name of a direct dependency that returns
a JSON array. The stage runs once for each item, and its dependents receive an
array of results in input order. A run can expand up to 128 items. On an explicit
retry, completed items are reused unless an upstream dependency is also rerun.

An optional `policy` object can set `max_transitions`, `max_live_agents`,
and `on_failure`. Each admitted stage or item consumes one transition,
including retries. The live-agent limit reserves capacity from the stage
contracts before starting them. Set `on_failure` to `stop` to finish active
stages and leave unstarted work waiting after a failure. The default
`continue` preserves the graph's dependency guards.

```sh
rlm-agent factory launch graph.json
rlm-agent factory status RUN_ID
rlm-agent factory view RUN_ID
rlm-agent factory cancel RUN_ID
rlm-agent factory resume RUN_ID --retry-stages
```

Editors can launch `rlm-agent --acp`, supply stdio or streamable HTTP MCP
servers for each session, and steer an active turn with text. Steering enters
at the next harness step. Closing a session stops its work and reports usage
and request counts. Saved sessions can be loaded again.

Saved sessions append a request ledger beside the transcript. It records how
model requests follow one another, delegate to children, receive child
returns, and follow compaction. HTTP model requests carry the corresponding
`X-Rlmagent-Request-Id`. Use `--engine session` when you need the same live
kernel across prompts and compaction.

```json
{
  "format": "rlmagent.factory/1",
  "stages": [
    {"contract": "inspect.json", "prompt": "Find the cause of the failing test."},
    {"contract": "fix.json", "prompt": "Fix the issue and verify the result."}
  ]
}
```

`rlm-agent update` shows the installed version. Use `rlm-agent update --apply`
to upgrade an installation from a package index.

`rlm-agent status` shows configuration, sandbox, git, MCP, search, workers and open
sessions; `rlm-agent doctor` reports problems and `doctor --fix` repairs the ones it
can (stale workers and sessions, a missing Pyodide).

## Background jobs and watches

`await job_start(cmd)` runs a command in the background; `job_read(id, cursor)` returns
what it wrote since `cursor`, `job_wait(id)` waits for it and `job_stop(id)` ends it and
everything it started. `await service_start(name, cmd, ready=regex)` keeps a dev server
or similar running and can wait for its ready line. `await watch(job=id)`,
`watch(path=p)` or `watch(messages=True)` waits until a job writes or ends, a path
changes, or another agent writes. Jobs belong to the session, so a sandbox restart
does not lose them, and they need the same approval as `run()`.

`await history(start, end)` reads every saved message of the session, including what
compaction summarized away; `history(agent=RUN_ID)` reads one agent's steps. When a
kernel process dies mid-run, a new one is started and the model is told which state
was lost, up to three times per kernel.

## Run policy

| Flag | Effect |
| --- | --- |
| `--deny-git` | Refuse git commands that touch other branches, remotes or history (push, fetch, checkout, branch, tag, reset, ...) |
| `--max-tool-output BYTES` | Most output one cell shows the model (default 20000; zero disables this cap; the start and end are kept) |
| `--max-compactions N` | Most compactions in the session |
| `--agent-max-tokens N` | Most completion tokens any one agent may spend |
| `--search-block DOMAIN` | Drop web search results from a domain, `*.domain` or origin (repeatable) |

A contract can set the same through an optional `policy` object with `allow_git`,
`max_tool_output_bytes`, `max_compactions`, `agent_max_tokens` and `search_block`
(every field given, `null` for the default). The summary a contract run prints ends
with `stop_reason`: `done`, `max_calls`, `max_cost`, `max_tokens`, `max_seconds`,
`agent_tokens`, `compaction_failed`, `no_answer`, `cancelled` or `error`.

Optional `policy.allowed_tools` limits both the model tool list and the host
functions available to Python, including in descendants. For example,
`["python", "read_file", "list_files"]` allows a Wasm Python cell to read and
list workspace files. Model tools still have to appear in the contract's
`tools` list. Omit the allowlist to use the original tool policy. Restricted
Python requires Wasm. Delegation and guest networking use the separate
`limits.max_depth` and `kernel.network` settings.

Optional `limits.max_new_tokens` meters reported uncached input and output
across the whole run, including compaction. The run admits one provider
request at a time with this limit enabled. Before each request it estimates
the request's size high (the whole prompt counted as uncached, plus the
provider's longest reply) and refuses a request that could pass the limit,
so a run stops short of it rather than over it. The run reports `max_tokens`
and refuses further requests after reaching the limit or losing usage coverage.

Cells that fail in a familiar way (a missing `await`, quotes nested in an f-string, a
missing module, a timeout, an empty reply) get a one-line hint with the error.

`--max-calls 0` admits no model requests. This differs from an omitted call budget,
which selects the default. In a policy, `max_tool_output_bytes: 0` explicitly
disables that output cap; `null` selects the default cap.

Contracts can use the two saved-account routes without exporting tokens. For Codex,
set the model to `{"provider":"codex","name":"MODEL","api_key_env":null,"base_url":null}`.
For Claude use provider `claude-subscription` and model name `cli:claude-code` or
`cli:claude-code?model=MODEL`, also with null key and endpoint fields. These routes
use the same signed-in accounts as the normal CLI and reject credential/endpoint
overrides. Codex contracts use the fixed Responses HTTP route. Claude runs through
the official CLI. Sign in before starting a contract; a contract never initiates login.

For inline ACP runtime contracts, use `"credential":{"account":"codex"}` or
`"credential":{"account":"claude-subscription"}` matching the contract provider.
Native API-key providers retain their explicit `api_key` and `headers` credential object.

ACP snapshots retain current-runtime `treeTotals` and additionally expose
`lifetimeTotals`, `lifetimeUsage`, and `lifetimeCoverage` for the persisted request
ledger plus current calls, deduplicated by request ID. Reopening a session reloads
these lifetime totals without consuming the new runtime's admission budget. Older
records with omitted usage fields or unknown pricing are counted in coverage;
missing usage is never presented as a complete measurement.

## Split execution

The agent loop and its keys can stay on one machine while the tools work somewhere
else. Start a tool host beside the files:

```sh
python -m rlmagent.toolhost --root /work                       # over stdio
RLM_AGENT_TOOLS_TOKEN=... python -m rlmagent.toolhost --root /work --tcp 0.0.0.0:7700
```

Then point a session at it with `--tools-at "python -m rlmagent.toolhost --root /work"`
or `--tools-at tcp://host:7700` (with the same `RLM_AGENT_TOOLS_TOKEN`). File reads,
search, edits, patches, commands and jobs then run in the tool host. Approvals and the
git guard are decided in the harness before anything is sent; the host only runs what
it is asked, confined to its root.

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
Use `/refine auto review` to stage a whole-harness proposal after compaction,
`/refine auto apply` to opt in to automatic activation, or
`/refine auto off` to cancel that mode for the current session.
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

`/refine scope session` selects guidance inherited along the saved conversation
branch. Rewinding restores the guidance attached to that ancestry; edits in a
sibling branch stay separate. `/refine scope project` returns to shared project
guidance. `/refine rollback N` restores an archived revision while idle and records
the restoration as a new revision. `/refine off`
sets a global preference that stops further review requests and removes supplemental
guidance from subsequent model requests, including requests in other processes.
An already submitted request cannot be recalled by changing this preference.

`--no-session` keeps this harness in memory. Plan mode prevents changes, and abort,
rewind, inference-route changes, or shutdown discard pending work. Contract runs
do not load this ambient harness or enable `/refine`.

This mechanism runs only when explicitly requested. Existing `/improve` and its
automatic playbook scheduling are separate. Proposals from the automatic playbook
check are held: `/improve pending` lists them, `/improve accept N` saves one and
`/improve reject N` drops it. `--apply-auto-improve` saves them at once instead. A committed refinement means its
structure and persistence checks passed; its expected benefit has not been
established by a performance benchmark.

## Portable sessions and delivery

`rlm-agent session export ID --format html` produces a standalone escaped HTML
transcript with tool arguments and branch and compaction summaries. JSONL keeps
the original records; `rlm-agent session import FILE` validates their references
and imports them under a fresh local identity.

`rlm-agent session share ID` previews the complete outgoing unlisted gist and its
SHA256 digest without uploading. To publish that exact payload, repeat with
`--publish --sha256 DIGEST` and a locally configured `GH_TOKEN` or `GITHUB_TOKEN`.
Unlisted links can be read by anyone who obtains them. A changed session requires
a fresh preview and digest.

The terminal copy action first tries the native clipboard locally. Over SSH or
tmux it emits OSC52, reporting delivery as unconfirmed because the terminal may
block it. Keymap overrides are validated for conflicts before they are applied.

Capability packages install into isolated stores with a resource manifest and
recorded distribution versions. Use `rlm-agent package install NAME SOURCE`,
`package update NAME --source REPLACEMENT`, and `package remove NAME`. Add
`--local` for a project store; project resources require project trust. Local
wheel installations also record their SHA256 digest.

Successful package changes automatically collect inactive cache directories older
than 30 days. Active installations, directories loaded by this process, and linked
directories are preserved. `package prune --older-than DAYS` previews eligible
directories; add `--apply` to remove them, or `--local` for the project store.

## Managed offline releases

`rlm-agent update --manifest /path/release.json` validates and previews a local
wheel bundle. Its manifest has `format: "rlmagent.release/1"`, an exact `version`,
a `channel` (`stable`, `preview`, or `nightly`), and `artifacts`, each containing
`filename` and `sha256`. Optional `extras` selects bundled optional dependencies.
Include the matching backend wheel and every dependency; installation uses
`--no-index` and rejects direct URL dependencies.

Add `--apply` to install into a fresh virtual environment and activate it only
after health checks. `--drain --wait 60 --restart-workers` lets current worker
tasks finish, restarts them with saved settings, and holds new work until fleet
activation completes. A drain timeout leaves processes alive. Failed activation
restores the prior installation; interrupted activation is recovered on the next
managed command. `update --managed --rollback --apply` selects the previous slot.
Use `rlm-agent update --run ...` to launch the active managed installation.

Managed channels describe a supplied, verified bundle; they do not publish it.
The index updater selects final releases for `stable`, non-development prereleases
for `preview`, and development releases for `nightly`. It skips yanked releases,
empty releases and artifacts incompatible with the current Python version. Preview
and nightly installs pin the selected version. If that channel has no eligible
published release, the updater reports this and leaves the installation unchanged.

### Execution continuity and failure handling

Each active WebAssembly kernel runs in its own Node worker. A timed-out cell
terminates that worker; sibling kernels retain their state. The default engine
keeps its interpreter across turns and conversation compaction, including imported
modules, object identities and asynchronous tasks. Resident retained children reuse
their interpreter on follow-up. Closing a session releases its interpreters.
After a process restart, recovery restores supported checkpoint values and
definitions; it cannot restore arbitrary live objects or unfinished Python tasks.

`--request-timeout SECONDS` bounds each complete model response, including nested
agent and summary requests (default 300). Contracts may set
`limits.request_timeout`; existing contracts use the same default. This is separate
from `--cell-timeout` and the whole-run `--max-seconds` limit. Invalid compaction
summaries get at most three attempts. A failed compaction keeps the original
history and does not consume the successful-compaction allowance. Session write
failures report the affected session and file instead of silently dropping records.

Execution processes inherit a small set of operating-system environment variables.
Host API-key and control variables are excluded from local kernels and shell jobs.
Approved commands and background jobs use an isolated `/workspace` mount by
default. Install Bubblewrap on Linux or in the default WSL distribution on Windows.
Use Linux executables and paths relative to the workspace for these commands.
`--shell-network` enables network access inside that workspace sandbox.
`--shell-access host` grants commands your host filesystem permissions. Contracts
can set `policy.shell_access` and `policy.shell_network`, and tool hosts accept the
same shell options. Select `--kernel local` explicitly for host Python execution.
A missing Wasm runtime stops with setup instructions.

Background job output is capped at 8 MiB on disk while the process output continues
to drain. `job_read` returns Unicode-safe byte cursors; job summaries include
stored bytes, total bytes and truncation state. Job metadata is stored under its
session, so reopening a session can recover its job records. `job_start` also
accepts `argv=[...]` for direct execution and explicit `env={...}` overrides.
Use `service_start(..., persist=True)` only for a service intentionally kept alive
after session shutdown. Reopen its owning session to read it or stop it by name
with `job_stop`. Ordinary services and jobs are stopped when their session closes.
