#Requires -Version 5.1

param(
    [string]$RepoUrl = "git@github.com:Viottery/agentic-rag.git",
    [string]$HttpsRepoUrl = "https://github.com/Viottery/agentic-rag.git",
    [string]$Branch = "master",
    [string]$InstallDir = "$HOME\agentic-rag",
    [string]$OpenAIApiKey = "",
    [string]$OpenAIBaseUrl = "https://api.openai.com/v1",
    [string]$OpenAIModel = "gpt-4o-mini",
    [string]$OpenAITemperature = "0",
    [string]$TavilyApiKey = "",
    [string]$PipIndexUrl = "https://mirrors.nju.edu.cn/pypi/web/simple",
    [string]$RequirementsFile = "requirements-windows-agent.txt",
    [switch]$UseHttps,
    [switch]$InstallPrereqs,
    [switch]$SkipDependencyInstall,
    [switch]$NoStart,
    [switch]$RecreateEnv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "WARN: $Message" -ForegroundColor Yellow
}

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Add-CommonToolPaths {
    $paths = @(
        "$Env:ProgramFiles\Git\cmd",
        "$Env:LocalAppData\Programs\Python\Python311",
        "$Env:LocalAppData\Programs\Python\Python311\Scripts",
        "$Env:ProgramFiles\Python311",
        "$Env:ProgramFiles\Python311\Scripts"
    )

    foreach ($path in $paths) {
        if ($path -and (Test-Path $path) -and ($env:Path -notlike "*$path*")) {
            $env:Path = "$env:Path;$path"
        }
    }
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    Write-Host "+ $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage Exit code: $LASTEXITCODE"
    }
}

function Install-WithWinget {
    param(
        [string]$Id,
        [string]$Name
    )

    if (-not (Test-CommandExists "winget")) {
        throw "winget is not available. Install App Installer from Microsoft Store or install $Name manually."
    }

    Write-Step "Installing $Name with winget"
    Invoke-Checked "winget" @(
        "install",
        "--id", $Id,
        "-e",
        "--accept-package-agreements",
        "--accept-source-agreements"
    ) "Failed to install $Name."
}

function Test-Python311 {
    if (Test-CommandExists "py") {
        & py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    if (Test-CommandExists "python") {
        & python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    return $false
}

function Invoke-SystemPython {
    param(
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    if (Test-CommandExists "py") {
        Write-Host "+ py -3.11 $($Arguments -join ' ')" -ForegroundColor DarkGray
        & py -3.11 @Arguments
    } else {
        Write-Host "+ python $($Arguments -join ' ')" -ForegroundColor DarkGray
        & python @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage Exit code: $LASTEXITCODE"
    }
}

function Get-VenvPython {
    return Join-Path $InstallDir ".venv\Scripts\python.exe"
}

function Invoke-VenvPython {
    param(
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    $venvPython = Get-VenvPython
    Write-Host "+ $venvPython $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage Exit code: $LASTEXITCODE"
    }
}

function Ensure-Prerequisites {
    if (-not $InstallPrereqs) {
        if (-not (Test-CommandExists "git")) {
            throw "Git is not installed. Re-run with -InstallPrereqs or install Git manually."
        }
        if (-not (Test-Python311)) {
            throw "Python 3.11 is not installed. Re-run with -InstallPrereqs or install Python 3.11 manually."
        }
        return
    }

    if (-not (Test-CommandExists "git")) {
        Install-WithWinget -Id "Git.Git" -Name "Git"
        Add-CommonToolPaths
    }

    if (-not (Test-Python311)) {
        Install-WithWinget -Id "Python.Python.3.11" -Name "Python 3.11"
        Add-CommonToolPaths
    }

    if (-not (Test-Python311)) {
        throw "Python 3.11 was installed but is not visible in this shell. Open a new PowerShell window and run again."
    }
}

function Sync-Repository {
    $selectedRepoUrl = if ($UseHttps) { $HttpsRepoUrl } else { $RepoUrl }

    if (-not (Test-Path $InstallDir)) {
        Write-Step "Cloning repository"
        $parent = Split-Path -Parent $InstallDir
        if ($parent) {
            New-Item -ItemType Directory -Force $parent | Out-Null
        }

        & git clone --branch $Branch $selectedRepoUrl $InstallDir
        if ($LASTEXITCODE -ne 0 -and -not $UseHttps) {
            Write-Warn "SSH clone failed. Falling back to HTTPS."
            & git clone --branch $Branch $HttpsRepoUrl $InstallDir
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to clone repository."
        }
    } else {
        Write-Step "Updating existing repository"
        Push-Location $InstallDir
        try {
            Invoke-Checked "git" @("fetch", "origin") "git fetch failed."
            Invoke-Checked "git" @("checkout", $Branch) "git checkout failed."
            Invoke-Checked "git" @("pull", "--ff-only", "origin", $Branch) "git pull failed."
        } finally {
            Pop-Location
        }
    }
}

function Ensure-CacheDirectories {
    Write-Step "Creating local runtime directories"
    New-Item -ItemType Directory -Force "$InstallDir\data\memory" | Out-Null
    New-Item -ItemType Directory -Force "$InstallDir\logs" | Out-Null
    New-Item -ItemType Directory -Force "$InstallDir\.runtime" | Out-Null
}

function Ensure-EnvFile {
    Push-Location $InstallDir
    try {
        if ((Test-Path ".env") -and -not $RecreateEnv) {
            Write-Step "Keeping existing .env"
            return
        }

        if (-not $script:OpenAIApiKey) {
            $script:OpenAIApiKey = Read-Host "Enter OPENAI_API_KEY (leave empty to create placeholder)"
        }

        $envBody = @"
APP_NAME=Agentic RAG
APP_ENV=windows-native
DEBUG=true
OPENAI_API_KEY=${script:OpenAIApiKey}
OPENAI_BASE_URL=$OpenAIBaseUrl
OPENAI_MODEL=$OpenAIModel
OPENAI_TEMPERATURE=$OpenAITemperature
TAVILY_API_KEY=$TavilyApiKey
TAVILY_SEARCH_DEPTH=basic
TAVILY_MAX_RESULTS=5
LOCAL_RAG_TRANSPORT=tcp
LOCAL_RAG_HOST=127.0.0.1
LOCAL_RAG_PORT=8765
LOCAL_RAG_ENABLED=false
SHELL_PROVIDER=powershell
SHELL_WORKSPACE_ROOT=
CONVERSATION_STORE_PATH=data/memory/conversations.db
RERANKER_ENABLED=false
"@

        Write-Step "Writing .env for native Windows"
        Set-Content -Path ".env" -Value $envBody -Encoding UTF8
    } finally {
        Pop-Location
    }
}

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
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

function Ensure-VenvAndDependencies {
    if ($SkipDependencyInstall) {
        Write-Step "Skipping Python dependency install"
        return
    }

    Push-Location $InstallDir
    try {
        if (-not (Test-Path ".venv\Scripts\python.exe")) {
            Write-Step "Creating Python virtual environment"
            Invoke-SystemPython @("-m", "venv", ".venv") "Failed to create virtual environment."
        }

        Write-Step "Installing Python dependencies"
        Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip") "Failed to upgrade pip."
        Invoke-VenvPython @("-m", "pip", "install", "-i", $PipIndexUrl, "-r", $RequirementsFile) "Failed to install requirements."
    } finally {
        Pop-Location
    }
}

function Stop-ExistingNativeApp {
    $pidFile = Join-Path $InstallDir ".runtime\native-app.pid"
    if (-not (Test-Path $pidFile)) {
        return
    }

    $oldPidText = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $oldPidText) {
        return
    }

    $oldPid = 0
    if (-not [int]::TryParse($oldPidText, [ref]$oldPid)) {
        return
    }

    $process = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($process) {
        Write-Step "Stopping existing native app process $oldPid"
        Stop-Process -Id $oldPid -Force
    }
}

function Start-NativeApp {
    if ($NoStart) {
        Write-Step "Skipping native app start"
        return
    }

    Push-Location $InstallDir
    try {
        Import-DotEnv -Path ".env"
        Stop-ExistingNativeApp

        $venvPython = Get-VenvPython
        if (-not (Test-Path $venvPython)) {
            throw "Virtual environment was not found: $venvPython"
        }

        $stdoutPath = Join-Path $InstallDir "logs\native-app.out.log"
        $stderrPath = Join-Path $InstallDir "logs\native-app.err.log"

        Write-Step "Starting native FastAPI app"
        $process = Start-Process `
            -FilePath $venvPython `
            -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
            -WorkingDirectory $InstallDir `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru

        Set-Content -Path ".runtime\native-app.pid" -Value $process.Id -Encoding ASCII
        Write-Host "Started PID $($process.Id)" -ForegroundColor Green
        Write-Host "stdout: $stdoutPath" -ForegroundColor Green
        Write-Host "stderr: $stderrPath" -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

function Wait-HttpHealth {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 180
    )

    if ($NoStart) {
        return
    }

    Write-Step "Waiting for $Url"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = ""
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Method Get -Uri $Url -TimeoutSec 5
            Write-Host ($response | ConvertTo-Json -Depth 8) -ForegroundColor Green
            return
        } catch {
            $lastError = $_.Exception.Message
            Start-Sleep -Seconds 3
        }
    }

    throw "Health check timed out: $Url. Last error: $lastError"
}

function Check-LocalRagHealth {
    Write-Step "Skipping local RAG health check because LOCAL_RAG_ENABLED=false in native agent mode."
}

function Send-SmokeChat {
    if ($NoStart) {
        return
    }
    if (-not $script:OpenAIApiKey) {
        Write-Warn "OPENAI_API_KEY is empty. Skipping /chat smoke request."
        return
    }

    Write-Step "Sending /chat smoke request"
    $body = @{
        question = "command: Write-Output agentic-rag-native-ok"
    } | ConvertTo-Json -Depth 8

    try {
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "http://127.0.0.1:8000/chat" `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 180
        Write-Host ($response | ConvertTo-Json -Depth 12) -ForegroundColor Green
    } catch {
        Write-Warn "/chat smoke request failed: $($_.Exception.Message)"
    }
}

Write-Step "Agentic RAG native Windows bootstrap"
Write-Warn "This path does not use Docker or WSL."
Write-Warn "Native agent mode disables local RAG/Qdrant/indexing first, so chat, shell/action, and web-search paths can run."
Write-Warn "Make sure the latest commits have been pushed to origin/$Branch before running this on a new machine."

Add-CommonToolPaths
Ensure-Prerequisites
Sync-Repository
Ensure-CacheDirectories
Ensure-EnvFile
Ensure-VenvAndDependencies
Start-NativeApp
Wait-HttpHealth -Url "http://127.0.0.1:8000/health"
Check-LocalRagHealth
Send-SmokeChat

Write-Step "Done"
Write-Host "Project: $InstallDir" -ForegroundColor Green
Write-Host "API:     http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Stop:    Stop-Process -Id (Get-Content '$InstallDir\.runtime\native-app.pid')" -ForegroundColor Green
