#!/bin/sh
# Install rlm-agent as a command-line tool, with its sandbox when Node is present.
#   curl -fsSL <url-of-this-file> | sh
# Set RLMAGENT_VERSION to pin a version, e.g. RLMAGENT_VERSION=0.2.0.
set -eu

spec="rlmness-cli"
if [ -n "${RLMAGENT_VERSION:-}" ]; then
    spec="rlmness-cli==${RLMAGENT_VERSION}"
fi

say() { printf '%s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
    if command -v pipx >/dev/null 2>&1; then
        say "Installing $spec with pipx"
        pipx install --force "$spec"
        installer=pipx
    else
        say "Installing uv (https://astral.sh/uv) to manage the tool"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        PATH="$HOME/.local/bin:$PATH"
        export PATH
    fi
fi

if [ "${installer:-}" != "pipx" ]; then
    say "Installing $spec with uv"
    uv tool install --force --python 3.12 "$spec"
    uv tool update-shell >/dev/null 2>&1 || true
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

if command -v node >/dev/null 2>&1; then
    say "Node found; installing the Pyodide sandbox"
    rlmness_bin="$(command -v rlmness || true)"
    if [ -z "$rlmness_bin" ] && command -v uv >/dev/null 2>&1; then
        uv tool run --from "$spec" rlmness --setup || say "Sandbox setup failed; run: rlm-agent doctor --fix"
    elif [ -n "$rlmness_bin" ]; then
        "$rlmness_bin" --setup || say "Sandbox setup failed; run: rlm-agent doctor --fix"
    fi
else
    say "Node.js 18+ was not found. The sandboxed kernel needs it; install Node, then run:"
    say "  rlm-agent doctor --fix"
fi

say ""
say "Done. Next:"
say "  rlm-agent config     # choose a provider and model"
say "  rlm-agent status     # check the setup"
say "  rlm-agent            # start in the current folder"
