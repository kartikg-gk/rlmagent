#!/bin/sh
# Install rlm-agent as a command-line tool, with its sandbox when Node is present.
#   curl --proto '=https' --tlsv1.2 -fsSL <url-of-this-file> | sh
# Set RLMAGENT_VERSION to pin a version, e.g. RLMAGENT_VERSION=0.2.2.
# Otherwise the latest release on PyPI is looked up and installed by exact version.
# Set NO_COLOR=1 for plain output.
set -eu

package="rlmness-cli"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != "dumb" ]; then
    bold=$(printf '\033[1m'); green=$(printf '\033[32m'); red=$(printf '\033[31m')
    yellow=$(printf '\033[33m'); reset=$(printf '\033[0m')
else
    bold=""; green=""; red=""; yellow=""; reset=""
fi

step_no=0
step() { step_no=$((step_no + 1)); printf '%s[%s/5]%s %s\n' "$bold" "$step_no" "$reset" "$*"; }
ok() { printf '      %sok%s %s\n' "$green" "$reset" "$*"; }
warn() { printf '      %snote%s %s\n' "$yellow" "$reset" "$*"; }
fail() {
    printf '%serror%s %s\n' "$red" "$reset" "$1" >&2
    if [ -n "${2:-}" ]; then printf '      %s\n' "$2" >&2; fi
    exit 1
}
fetch() { curl --proto '=https' --tlsv1.2 -fsSL "$@"; }

step "Checking this machine"
case "$(uname -s)" in
    Linux|Darwin) ok "$(uname -s) $(uname -m)" ;;
    MINGW*|MSYS*|CYGWIN*)
        fail "This script is for macOS, Linux and WSL." \
             "On Windows use PowerShell: irm <url>/install.ps1 | iex" ;;
    *) fail "Unsupported system: $(uname -s)." "Install with pip instead: pip install $package" ;;
esac
command -v curl >/dev/null 2>&1 || fail "curl was not found." "Install curl with your package manager and run this again."

step "Choosing the version"
if [ -n "${RLMAGENT_VERSION:-}" ]; then
    version="$RLMAGENT_VERSION"
    ok "$version (pinned by RLMAGENT_VERSION)"
else
    version=$(fetch "https://pypi.org/pypi/$package/json" 2>/dev/null \
        | tr ',' '\n' | sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' | head -n 1) || version=""
    if [ -z "$version" ]; then
        fail "Could not read the latest version from PyPI." \
             "Check your connection, or set RLMAGENT_VERSION=<version> and run this again."
    fi
    ok "$version (latest on PyPI)"
fi
spec="$package==$version"

step "Getting uv, which installs and isolates the tool"
if command -v uv >/dev/null 2>&1; then
    ok "found $(uv --version 2>/dev/null || echo uv)"
else
    fetch https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
        || fail "Installing uv failed." "See https://docs.astral.sh/uv/getting-started/installation/ and run this again."
    PATH="$HOME/.local/bin:$PATH"
    export PATH
    command -v uv >/dev/null 2>&1 || fail "uv was installed but is not on PATH." "Open a new terminal and run this again."
    ok "installed uv"
fi

step "Installing $spec"
# Builds published before the rename installed a tool named rlmagent that
# owns the same rlm-agent command; take it out so the two do not clash.
if uv tool list 2>/dev/null | grep -q '^rlmagent '; then
    uv tool uninstall rlmagent >/dev/null 2>&1 || true
    ok "removed the earlier rlmagent install"
fi
uv tool install --force --python 3.12 "$spec" \
    || fail "Installing $spec failed." "The output above says why. Check the version exists: https://pypi.org/project/$package/#history"
uv tool update-shell >/dev/null 2>&1 || true
PATH="$HOME/.local/bin:$PATH"
export PATH
ok "rlm-agent $version"

step "Setting up the sandbox for the agent's code"
if command -v node >/dev/null 2>&1; then
    if rlmness --setup >/dev/null 2>&1; then
        ok "Node $(node --version) and Pyodide"
    else
        warn "Sandbox setup failed. Run: rlm-agent doctor --fix"
    fi
else
    warn "Node.js 18+ was not found. Install it from https://nodejs.org, then run: rlm-agent doctor --fix"
    warn "Until then, rlm-agent --kernel local runs code directly on this machine."
fi

printf '\n%sDone.%s Open a new terminal, then:\n' "$bold" "$reset"
printf '  rlm-agent status     # check the setup\n'
printf '  rlm-agent            # start in a project folder; the first run picks a provider\n'
