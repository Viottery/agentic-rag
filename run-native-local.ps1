$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $projectRoot ".runtime"
$logsDir = Join-Path $projectRoot "logs"
$pidFile = Join-Path $runtimeDir "native-app.pid"
$stdoutPath = Join-Path $logsDir "native-app.out.log"
$stderrPath = Join-Path $logsDir "native-app.err.log"

New-Item -ItemType Directory -Force $runtimeDir | Out-Null
New-Item -ItemType Directory -Force $logsDir | Out-Null

$envPath = Join-Path $projectRoot ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or ($line -notmatch "=")) {
            return
        }
        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($key) {
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

if (-not (Test-Path $python)) {
    throw "Virtual environment Python not found: $python"
}

if (Test-Path $pidFile) {
    $oldPidText = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    $oldPid = 0
    if ([int]::TryParse($oldPidText, [ref]$oldPid)) {
        $oldProcess = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($oldProcess) {
            Stop-Process -Id $oldPid -Force
        }
    }
}

Push-Location $projectRoot
try {
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    Set-Content -Path $pidFile -Value $process.Id -Encoding ASCII
    Write-Host "Started PID $($process.Id)"
    Write-Host "API http://127.0.0.1:8000"
    Write-Host "stdout $stdoutPath"
    Write-Host "stderr $stderrPath"
} finally {
    Pop-Location
}
