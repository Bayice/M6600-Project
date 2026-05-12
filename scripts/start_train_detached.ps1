# Start overnight training as an independent background PowerShell process.
# Similar to tmux detach behavior:
#   - You may close this launcher terminal after it starts.
#   - Training keeps running as long as Windows does not sleep / hibernate.
#   - Output is saved to launcher logs and overnight logs.

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\hexyw\Desktop\M6600_Project"
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force logs | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = "logs\launcher_$Timestamp.stdout.log"
$StderrLog = "logs\launcher_$Timestamp.stderr.log"
$PidFile = "logs\overnight_train_$Timestamp.pid"

$ScriptPath = Join-Path $ProjectRoot "scripts\run_all_train_overnight.ps1"

$ArgsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$ScriptPath`""
)

$Process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $ArgsList `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Minimized `
    -PassThru

$Process.Id | Out-File -FilePath $PidFile -Encoding ascii

Write-Host "Started overnight training."
Write-Host "PID: $($Process.Id)"
Write-Host "PID file: $PidFile"
Write-Host "Launcher stdout log: $StdoutLog"
Write-Host "Launcher stderr log: $StderrLog"
Write-Host ""
Write-Host "You can close this terminal now."
Write-Host ""
Write-Host "Check whether it is still running:"
Write-Host "  Get-Process -Id $($Process.Id)"
Write-Host ""
Write-Host "Find latest overnight log directory:"
Write-Host "  Get-ChildItem logs\overnight_* | Sort-Object LastWriteTime | Select-Object -Last 1"
Write-Host ""
Write-Host "Watch launcher stdout:"
Write-Host "  Get-Content $StdoutLog -Wait"
Write-Host ""
Write-Host "Watch launcher stderr:"
Write-Host "  Get-Content $StderrLog -Wait"
Write-Host ""
Write-Host "After the overnight directory appears, watch master log:"
Write-Host "  Get-Content logs\overnight_YYYYMMDD_HHMMSS\MASTER_overnight_train.log -Wait"