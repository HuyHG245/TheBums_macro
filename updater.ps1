# Local transaction helper bundled by the application; never downloaded from a release.
param(
    [Parameter(Mandatory=$true)][string]$CurrentExe,
    [Parameter(Mandatory=$true)][string]$StagedExe,
    [Parameter(Mandatory=$true)][string]$HealthFile,
    [Parameter(Mandatory=$true)][string]$StatusFile,
    [Parameter(Mandatory=$true)][string]$ReleaseTag,
    [Parameter(Mandatory=$true)][string]$Digest,
    [Parameter(Mandatory=$true)][int]$ParentPid,
    [Parameter(Mandatory=$true)][string]$ReadyFile,
    [Parameter(Mandatory=$true)][string]$CommitFile
)
$ErrorActionPreference = 'Stop'
$backup = "$CurrentExe.rollback"
$started = $null
$backupCreated = $false
$replacementAttempted = $false
$authorized = $false
$parentExited = $false
$parentProcess = $null
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
function Start-Macro([bool]$withHealth) {
    # Even if a helper was launched outside the Python launcher, never let a
    # restarted one-file EXE inherit an exiting PyInstaller bootloader session.
    Get-ChildItem Env: | Where-Object { $_.Name -like '_PYI_*' -or $_.Name -eq '_MEIPASS2' } |
        ForEach-Object { Remove-Item -LiteralPath ('Env:'+ $_.Name) -ErrorAction SilentlyContinue }
    Remove-Item Env:THEBUMS_UPDATE_HEALTH -ErrorAction SilentlyContinue
    $env:PYINSTALLER_RESET_ENVIRONMENT = '1'
    if ($withHealth) { $env:THEBUMS_UPDATE_HEALTH = $healthName }
    try { return (Start-Process -FilePath $CurrentExe -WorkingDirectory ([System.IO.Path]::GetDirectoryName($CurrentExe)) -PassThru) }
    finally {
        Remove-Item Env:THEBUMS_UPDATE_HEALTH -ErrorAction SilentlyContinue
        Remove-Item Env:PYINSTALLER_RESET_ENVIRONMENT -ErrorAction SilentlyContinue
    }
}
try {
    Write-Output "Updater $PID started; validating transaction."
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
    if ($ReadyFile -ne "$HealthFile.ready" -or $CommitFile -ne "$HealthFile.commit") {
        throw 'Invalid transaction handshake paths.'
    }
    $lockFile = "$CurrentExe.updating"
    if ((Get-Content -LiteralPath $lockFile -Raw).Trim() -ne [string]$ParentPid) {
        throw 'Update reservation is not owned by the parent process.'
    }
    $parentProcess = Get-Process -Id $ParentPid -ErrorAction Stop
    # Verify ability to write in the EXE directory before telling the app to close.
    $probe = "$CurrentExe.$PID.writeprobe"
    try { [System.IO.File]::WriteAllText($probe, 'ready') }
    finally { Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue }
    [System.IO.File]::WriteAllText($ReadyFile, [string]$PID)
    Write-Output 'Ready acknowledged; waiting for explicit install authorization.'
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ((Test-Path -LiteralPath $CommitFile) -and
            (Get-Content -LiteralPath $CommitFile -Raw) -eq [string]$PID) {
            $authorized = $true; break
        }
        if ($parentProcess.HasExited) { throw 'Parent exited without authorizing installation.' }
        Start-Sleep -Milliseconds 100
    }
    if (-not $authorized) { throw 'Install authorization timed out.' }
    Write-Output 'Install authorized; waiting for the parent and EXE file lock.'
    if (-not $parentProcess.WaitForExit(90000)) { throw 'Parent process did not exit.' }
    $parentExited = $true
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
    Write-Output 'Verified replacement installed; launching a fresh process.'
    $started = Start-Macro $true
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
    Write-Output 'Installation passed startup health check.'
} catch {
    $message = $_.Exception.Message
    Write-Output "Updater failed: $message"
    # Before authorization, Python owns the verified download and the lock.
    # Do not remove them or start a second copy of the still-running app.
    if (-not $authorized) { exit 1 }
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
            Start-Macro $false | Out-Null
        } else {
            # Never restore an older, stale backup when this transaction did not replace the EXE.
            Write-Status 'failed' $message
            if ($parentExited -and (Test-Path -LiteralPath $CurrentExe) -and (Wait-ExeReleased $CurrentExe)) {
                Start-Macro $false | Out-Null
            }
        }
    } catch {
        Write-Output ("Could not recover previous EXE: " + $_.Exception.Message)
        Write-Status 'failed' ("Could not recover previous EXE: " + $_.Exception.Message)
    }
} finally {
    Remove-Item Env:THEBUMS_UPDATE_HEALTH -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $HealthFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ReadyFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $CommitFile -Force -ErrorAction SilentlyContinue
    if ($authorized -and $parentExited) {
        # Release only this transaction's reservation, never a replacement lock.
        try {
            if ((Get-Content -LiteralPath "$CurrentExe.updating" -Raw).Trim() -eq [string]$ParentPid) {
                Remove-Item -LiteralPath "$CurrentExe.updating" -Force -ErrorAction SilentlyContinue
            }
        } catch { Write-Output 'Could not clear update reservation; see status.' }
    }
}
