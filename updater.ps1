# Local transaction helper bundled by the application; NOT downloaded/executed from a release.
# Installed EXE may be replaced only after the user has stopped the macro.
param(
    [Parameter(Mandatory=$true)][string]$CurrentExe,
    [Parameter(Mandatory=$true)][string]$StagedExe,
    [Parameter(Mandatory=$true)][string]$HealthFile,
    [Parameter(Mandatory=$true)][string]$StatusFile,
    [Parameter(Mandatory=$true)][string]$ReleaseTag,
    [Parameter(Mandatory=$true)][string]$Digest
)
$ErrorActionPreference = 'Stop'
$backup = "$CurrentExe.rollback"
$started = $null
function Write-Status([string]$status, [string]$message) {
    try {
        $payload = @{ tag=$ReleaseTag; status=$status; digest=$Digest; details=$message }
        $payload | ConvertTo-Json -Compress | Set-Content -LiteralPath $StatusFile -Encoding UTF8
    } catch {}
}
try {
    if (-not (Test-Path -LiteralPath $CurrentExe) -or -not (Test-Path -LiteralPath $StagedExe)) {
        throw 'The original or staged executable is missing.'
    }
    if ($CurrentExe -notlike '*.exe' -or $StagedExe -notlike '*.new') {
        throw 'Refusing to update paths with unexpected extensions.'
    }
    # Wait for the old process to shut down and for Windows to release its EXE.
    $released = $false
    for ($attempt=0; $attempt -lt 90; $attempt++) {
        try {
            $fh = [System.IO.File]::Open($CurrentExe,'Open','ReadWrite','None')
            $fh.Close()
            $released = $true
            break
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $released) { throw 'The original executable is still locked.' }
    Copy-Item -LiteralPath $CurrentExe -Destination $backup -Force
    if (Test-Path -LiteralPath $HealthFile) {
        Remove-Item -LiteralPath $HealthFile -Force
    }
    Move-Item -LiteralPath $StagedExe -Destination $CurrentExe -Force
    $started = Start-Process -FilePath $CurrentExe -PassThru
    $healthy = $false
    for ($i=0; $i -lt 120; $i++) {
        if (Test-Path -LiteralPath $HealthFile) { $healthy=$true; break }
        $started.Refresh()
        if ($started.HasExited) { break }
        Start-Sleep -Seconds 1
    }
    if (-not $healthy) { throw 'The updated app did not pass its startup health check.' }
    Write-Status 'installed' 'New version launched and passed startup health check.'
} catch {
    $message = $_.Exception.Message
    try {
        if ($started) {
            $started.Refresh()
            if (-not $started.HasExited) { Stop-Process -Id $started.Id -Force }
            Start-Sleep -Seconds 2
        }
        if (Test-Path -LiteralPath $backup) {
            Copy-Item -LiteralPath $backup -Destination $CurrentExe -Force
            Write-Status 'rolled_back' $message
            Start-Process -FilePath $CurrentExe | Out-Null
        } else {
            Write-Status 'failed' $message
        }
    } catch {
        Write-Status 'failed' ("Could not restore previous EXE: " + $_.Exception.Message)
    }
} finally {
    try { Remove-Item -LiteralPath "$CurrentExe.updating" -Force -ErrorAction SilentlyContinue } catch {}
}
