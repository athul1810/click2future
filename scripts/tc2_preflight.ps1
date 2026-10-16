[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9.-]*$')]
    [string]$HostName,
    [ValidatePattern('^[a-z0-9][a-z0-9._-]*$')]
    [string]$Username,
    [ValidateRange(500, 30000)]
    [int]$TimeoutMs = 5000,
    [switch]$CheckLogin,
    [string]$ReportPath = 'outputs/tc2-access.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($CheckLogin -and -not $Username) {
    throw 'CheckLogin requires your approved NTU network username.'
}
$accessReport = [ordered]@{
    checked_at_utc = [DateTime]::UtcNow.ToString('o')
    host = $HostName
    port = 22
    tcp_reachable = $false
    login_checked = $false
    authenticated = $false
    note = 'TCP reachability alone does not establish account approval or GPU access.'
}
$connection = New-Object System.Net.Sockets.TcpClient
try {
    $attempt = $connection.ConnectAsync($HostName, 22)
    $finished = $attempt.Wait($TimeoutMs)
    $accessReport.tcp_reachable = $finished -and $connection.Connected
} catch {
    $accessReport['connection_error'] = $_.Exception.GetBaseException().Message
} finally {
    $connection.Dispose()
}
if ($accessReport.tcp_reachable -and $CheckLogin) {
    $sshCommand = Get-Command ssh -ErrorAction Stop
    # Non-interactive key authentication only; never collect passwords or accept unknown keys.
    & $sshCommand.Source -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8 `
        "$Username@$HostName" 'command -v mytcinfo >/dev/null 2>&1 && mytcinfo; id -un'
    $accessReport.login_checked = $true
    $accessReport.authenticated = $LASTEXITCODE -eq 0
}
$destination = [IO.Path]::GetFullPath($ReportPath)
$parentDirectory = Split-Path -Parent $destination
New-Item -ItemType Directory -Path $parentDirectory -Force | Out-Null
$accessReport | ConvertTo-Json | Set-Content -LiteralPath $destination -Encoding UTF8
$accessReport | ConvertTo-Json
if (-not $accessReport.tcp_reachable) { exit 2 }
if ($CheckLogin -and -not $accessReport.authenticated) { exit 3 }
exit 0
