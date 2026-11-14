# Install rlm-agent as a command-line tool, with its sandbox when Node is present.
#   irm <url-of-this-file> | iex
# Set $env:RLMAGENT_VERSION to pin a version, e.g. $env:RLMAGENT_VERSION = '0.2.2'.
# Otherwise the latest release on PyPI is looked up and installed by exact version.
# Set $env:NO_COLOR = '1' for plain output.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$package = 'rlmness-cli'
$color = -not $env:NO_COLOR
$script:stepNo = 0

function Have($name) { $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }
function Say($text, $fg) { if ($color -and $fg) { Write-Host $text -ForegroundColor $fg } else { Write-Host $text } }
function Step($text) { $script:stepNo++; Say "[$($script:stepNo)/4] $text" 'White' }
function Ok($text) { Say "      ok $text" 'Green' }
function Note($text) { Say "      note $text" 'Yellow' }
function Fail($text, $hint) {
    Say "error $text" 'Red'
    if ($hint) { Write-Host "      $hint" }
    throw $text
}

Step 'Choosing the version'
if ($env:RLMAGENT_VERSION) {
    $version = $env:RLMAGENT_VERSION
    Ok "$version (pinned by RLMAGENT_VERSION)"
} else {
    $version = ''
    try { $version = (Invoke-RestMethod "https://pypi.org/pypi/$package/json").info.version } catch { }
    if (-not $version) {
        Fail 'Could not read the latest version from PyPI.' 'Check your connection, or set $env:RLMAGENT_VERSION and run this again.'
    }
    Ok "$version (latest on PyPI)"
}
$spec = "$package==$version"

Step 'Getting uv, which installs and isolates the tool'
if (Have 'uv') {
    Ok 'found uv'
} else {
    try { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression | Out-Null } catch {
        Fail 'Installing uv failed.' 'See https://docs.astral.sh/uv/getting-started/installation/ and run this again.'
    }
    $env:Path = "$HOME\.local\bin;$env:Path"
    if (-not (Have 'uv')) { Fail 'uv was installed but is not on PATH.' 'Open a new terminal and run this again.' }
    Ok 'installed uv'
}

Step "Installing $spec"
# Builds published before the rename installed a tool named rlmagent that
# owns the same rlm-agent command; take it out so the two do not clash.
# With no tools installed uv says so on stderr, which Windows PowerShell turns
# into a stopping error here; read the list inside try so that case is just empty.
$listed = ''
try { $listed = (& uv tool list 2>$null | Out-String) } catch { }
if ($listed -match '(?m)^rlmagent ') {
    try { uv tool uninstall rlmagent | Out-Null } catch { }
    Ok 'removed the earlier rlmagent install'
}
# uv reports progress on stderr; let it through without stopping on it.
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
uv tool install --force --python 3.12 $spec
$code = $LASTEXITCODE; $ErrorActionPreference = $prev
if ($code -ne 0) {
    Fail "Installing $spec failed." "The output above says why. Check the version exists: https://pypi.org/project/$package/#history"
}
try { uv tool update-shell 2>$null | Out-Null } catch { }
$env:Path = "$HOME\.local\bin;$env:Path"
Ok "rlm-agent $version"

Step "Setting up the sandbox for the agent's code"
if (Have 'node') {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    rlmness --setup *> $null
    $code = $LASTEXITCODE; $ErrorActionPreference = $prev
    if ($code -eq 0) { Ok "Node $(node --version) and Pyodide" } else { Note 'Sandbox setup failed. Run: rlm-agent doctor --fix' }
} else {
    Note 'Node.js 18+ was not found. Install it from https://nodejs.org, then run: rlm-agent doctor --fix'
    Note 'Until then, rlm-agent --kernel local runs code directly on this machine.'
}

Write-Host ''
Write-Host 'Done. Open a new terminal, then:'
Write-Host '  rlm-agent status     # check the setup'
Write-Host '  rlm-agent            # start in a project folder; the first run picks a provider'
