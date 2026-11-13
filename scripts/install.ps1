# Install rlm-agent as a command-line tool, with its sandbox when Node is present.
#   irm <url-of-this-file> | iex
# Set $env:RLMAGENT_VERSION to pin a version, e.g. $env:RLMAGENT_VERSION = '0.2.0'.
$ErrorActionPreference = 'Stop'

$spec = 'rlmness-cli'
if ($env:RLMAGENT_VERSION) { $spec = "rlmness-cli==$($env:RLMAGENT_VERSION)" }

function Have($name) { $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }

if (-not (Have 'uv')) {
    Write-Host 'Installing uv (https://astral.sh/uv) to manage the tool'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$HOME\.local\bin;$env:Path"
}

Write-Host "Installing $spec with uv"
uv tool install --force --python 3.12 $spec
try { uv tool update-shell | Out-Null } catch { }
$env:Path = "$HOME\.local\bin;$env:Path"

if (Have 'node') {
    Write-Host 'Node found; installing the Pyodide sandbox'
    try {
        if (Have 'rlmness') { rlmness --setup } else { uv tool run --from $spec rlmness --setup }
    } catch {
        Write-Host 'Sandbox setup failed; run: rlm-agent doctor --fix'
    }
} else {
    Write-Host 'Node.js 18+ was not found. The sandboxed kernel needs it; install Node, then run:'
    Write-Host '  rlm-agent doctor --fix'
}

Write-Host ''
Write-Host 'Done. Open a new terminal, then:'
Write-Host '  rlm-agent config     # choose a provider and model'
Write-Host '  rlm-agent status     # check the setup'
Write-Host '  rlm-agent            # start in the current folder'
