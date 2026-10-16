[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9._-]*$')]
    [string]$Username,
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9.-]*$')]
    [string]$HostName = '10.96.189.12',
    [ValidateSet('Inventory', 'Setup', 'Collect')]
    [string]$Stage = 'Inventory',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$Qos = 'normal',
    [ValidatePattern('^[a-f0-9]{40}$')]
    [string]$Revision,
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    foreach ($toolName in @('git', 'ssh', 'scp')) { Get-Command $toolName -ErrorAction Stop | Out-Null }
    if (-not $Revision) { $Revision = (& git rev-parse HEAD).Trim() }
    else { & git cat-file -e "$Revision^{commit}" }
    if ($LASTEXITCODE -ne 0 -or $Revision -notmatch '^[a-f0-9]{40}$') { throw 'Cannot resolve source commit.' }
    $sourceFiles = @(& git ls-tree -r --name-only $Revision)
    foreach ($requiredFile in @('cluster/run_setup.sh', 'cluster/setup.sbatch', 'uv.lock')) {
        if ($sourceFiles -notcontains $requiredFile) { throw "Commit $Revision does not contain $requiredFile." }
    }
    $trackedChanges = @(& git status --porcelain --untracked-files=no)
    if ($trackedChanges.Count) { throw 'Commit tracked changes before deployment; only the recorded commit is uploaded.' }
    $localDirectory = Join-Path $projectRoot "outputs\tc2-setup\$Revision"
    New-Item -ItemType Directory -Path $localDirectory -Force | Out-Null
    $archiveName = "c2f-$Revision.tar.gz"
    $archivePath = Join-Path $localDirectory $archiveName
    $target = "$Username@$HostName"
    $mode = $Stage.ToLowerInvariant()
    $sshOptions = @('-o', 'StrictHostKeyChecking=ask', '-o', 'ConnectTimeout=15',
        '-o', 'NumberOfPasswordPrompts=2', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3')
    if ($Stage -eq 'Inventory') {
        & git archive --format=tar.gz "--output=$archivePath" $Revision
        if ($LASTEXITCODE -ne 0) { throw 'Source archive failed.' }
        $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $archiveHash = '' }
    $remoteScript = @'
set -euo pipefail
umask 077
release="$HOME/click2future/releases/__REVISION__"
__STAGING__
if [[ ! -f "$release/outputs/source-revision.txt" ]] || [[ "$(cat "$release/outputs/source-revision.txt")" != __REVISION__ ]]; then
    printf '%s\n' 'Source release is missing or incomplete. Run Inventory first.' >&2
    exit 2
fi
export C2F_PROJECT="$release" C2F_QOS='__QOS__'
bash "$release/cluster/run_setup.sh" __MODE__
'@
    $staging = @'
archive="$HOME/__ARCHIVE__"
printf '%s  %s\n' '__HASH__' "$archive" | sha256sum -c -
mkdir -p "$HOME/click2future/releases"
if [[ ! -e "$release" ]]; then
    mkdir "$release"
    tar -xzf "$archive" -C "$release" --no-same-owner
    mkdir -p "$release/outputs"
    printf '%s\n' '__REVISION__' > "$release/outputs/source-revision.txt"
else
    printf '%s\n' 'Reusing this source release; existing source will not be overwritten.'
fi
'@
    if ($Stage -ne 'Inventory') { $staging = '' }
    $remoteScript = $remoteScript.Replace('__STAGING__', $staging).Replace('__REVISION__', $Revision).
        Replace('__ARCHIVE__', $archiveName).Replace('__HASH__', $archiveHash).
        Replace('__QOS__', $Qos).Replace('__MODE__', $mode).Replace(([string][char]13), '')
    [IO.File]::WriteAllText((Join-Path $localDirectory "remote-$mode.sh"), $remoteScript, (New-Object Text.UTF8Encoding($false)))
    Write-Host "Target: $target | stage: $Stage | QoS: $Qos | source: $Revision"
    Write-Host 'Only committed source is transferred. No Git credentials, local environment, datasets, or keys are uploaded.'
    Write-Host 'Enter NTU credentials only in SSH/SCP prompts. No transcript or password storage is used.'
    if ($DryRun) {
        Write-Host "Dry run passed. Prepared commands are in $localDirectory; no connection or job submission occurred."
        return
    }
    if ($Stage -eq 'Inventory') {
        & scp @sshOptions $archivePath "$($target):$archiveName"
        if ($LASTEXITCODE -ne 0) { throw 'Source transfer failed.' }
    }
    $encodedScript = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript))
    $remoteCommand = "printf '%s' '$encodedScript' | base64 --decode | bash -l"
    $reportLines = New-Object 'System.Collections.Generic.List[string]'
    $bundleLines = New-Object 'System.Collections.Generic.List[string]'
    $readingBundle = $false
    $finishedBundle = $false
    & ssh @sshOptions $target $remoteCommand | ForEach-Object {
        $line = [string]$_
        if ($line -eq 'C2F_DIAGNOSTICS_BEGIN') { $readingBundle = $true }
        elseif ($line -eq 'C2F_DIAGNOSTICS_END') { $readingBundle = $false; $finishedBundle = $true }
        elseif ($readingBundle) { $bundleLines.Add($line) }
        else { Write-Host $line; $reportLines.Add($line) }
    }
    $remoteExitCode = $LASTEXITCODE
    $reportLines | Set-Content -LiteralPath (Join-Path $localDirectory "$mode.log") -Encoding UTF8
    if ($finishedBundle) {
        $bundlePath = Join-Path $localDirectory "diagnostics-$mode.tar.gz"
        [IO.File]::WriteAllBytes($bundlePath, [Convert]::FromBase64String(($bundleLines -join '')))
        Write-Host "Diagnostics saved: $bundlePath"
    }
    [ordered]@{ checked_at_utc = [DateTime]::UtcNow.ToString('o'); stage = $mode;
        source_revision = $Revision; remote_exit_code = $remoteExitCode;
        diagnostics_downloaded = $finishedBundle } | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $localDirectory "$mode-result.json") -Encoding UTF8
    if ($remoteExitCode -ne 0) { throw "Remote stage failed (exit $remoteExitCode). Inspect the saved diagnostics." }
    if (-not $finishedBundle) { throw 'No completed diagnostic bundle was returned; setup remains unverified.' }
} finally { Pop-Location }
