# Local transaction helper bundled by the application; never downloaded from a release.
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
$backupCreated = $false
$replacementAttempted = $false
$healthName = [System.IO.Path]::GetFileName($HealthFile)
function Write-Status([string]$status, [string]$message) {
    $tmp = "$StatusFile.$PID.tmp"
    try {
        $payload = @{ tag=$ReleaseTag; status=$status; digest=$Digest; details=$message }
        $payload | ConvertTo-Json -Compress | Set-Content -LiteralPath $tmp -Encoding UTF8
        Move-Item -LiteralPath $tmp -Destination $StatusFile -Force
    } catch {} finally { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
}
function Wait-ExeReleased([string]$path) {
    for ($attempt=0; $attempt -lt 90; $attempt++) {
        try {
            $fh = [System.IO.File]::Open($path,'Open','ReadWrite','None')
            $fh.Close()
            return $true
        } catch { Start-Sleep -Seconds 1 }
    }
    return $false
}
try {
    if (-not (Test-Path -LiteralPath $CurrentExe) -or -not (Test-Path -LiteralPath $StagedExe)) {
        throw 'The original or staged executable is missing.'
    }
    if ($CurrentExe -notlike '*.exe' -or $StagedExe -ne "$CurrentExe.new") {
        throw 'Refusing to update paths with unexpected extensions.'
    }
    if ($healthName -notmatch '^update_health_[0-9a-f]{32}\.ok$') {
        throw 'Invalid transaction health marker.'
    }
    if ($Digest -notmatch '^sha256:[0-9a-fA-F]{64}$') { throw 'Invalid update digest.' }
    if (-not (Wait-ExeReleased $CurrentExe)) { throw 'The original executable is still locked.' }
    # Recheck immediately before replacement: staged bytes may have changed since download.
    if ((Get-FileHash -LiteralPath $StagedExe -Algorithm SHA256).Hash -ne $Digest.Substring(7)) {
        throw 'Staged EXE SHA-256 mismatch.'
    }
    Copy-Item -LiteralPath $CurrentExe -Destination $backup -Force
    $backupCreated = $true
    Remove-Item -LiteralPath $HealthFile -Force -ErrorAction SilentlyContinue
    $replacementAttempted = $true
    Move-Item -LiteralPath $StagedExe -Destination $CurrentExe -Force
    $env:THEBUMS_UPDATE_HEALTH = $healthName
    try { $started = Start-Process -FilePath $CurrentExe -PassThru }
    finally { Remove-Item Env:THEBUMS_UPDATE_HEALTH -ErrorAction SilentlyContinue }
    $healthy = $false
    for ($i=0; $i -lt 120; $i++) {
        $started.Refresh()
        if ($started.HasExited) { break }
        if ((Test-Path -LiteralPath $HealthFile) -and (Get-Content -LiteralPath $HealthFile -Raw) -eq 'ready') {
            $healthy=$true; break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $healthy) { throw 'The updated app did not pass its startup health check.' }
    Write-Status 'installed' 'New version launched and passed startup health check.'
} catch {
    $message = $_.Exception.Message
    try {
        if ($replacementAttempted -and $backupCreated) {
            if ($started) {
                $started.Refresh()
                if (-not $started.HasExited) { Stop-Process -Id $started.Id -Force }
            }
            if ((Test-Path -LiteralPath $CurrentExe) -and -not (Wait-ExeReleased $CurrentExe)) {
                throw 'Updated executable remains locked; backup retained for manual recovery.'
            }
            Copy-Item -LiteralPath $backup -Destination $CurrentExe -Force
            Write-Status 'rolled_back' $message
            Start-Process -FilePath $CurrentExe | Out-Null
        } else {
            # Never restore an older, stale backup when this transaction did not replace the EXE.
            Write-Status 'failed' $message
            if ((Test-Path -LiteralPath $CurrentExe) -and (Wait-ExeReleased $CurrentExe)) {
                Start-Process -FilePath $CurrentExe | Out-Null
            }
        }
    } catch {
        Write-Status 'failed' ("Could not recover previous EXE: " + $_.Exception.Message)
    }
} finally {
    Remove-Item Env:THEBUMS_UPDATE_HEALTH -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $HealthFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "$CurrentExe.updating" -Force -ErrorAction SilentlyContinue
}
