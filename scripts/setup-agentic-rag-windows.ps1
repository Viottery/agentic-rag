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
    [switch]$UseHttps,
    [switch]$InstallPrereqs,
    [switch]$SkipBuild,
    [switch]$IndexData,
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
        "$Env:ProgramFiles\Docker\Docker\resources\bin"
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

function Ensure-Prerequisites {
    if (-not $InstallPrereqs) {
        if (-not (Test-CommandExists "git")) {
            throw "Git is not installed. Re-run with -InstallPrereqs or install Git manually."
        }
        if (-not (Test-CommandExists "docker")) {
            throw "Docker is not installed. Re-run with -InstallPrereqs or install Docker Desktop manually."
        }
        return
    }

    if (-not (Test-CommandExists "git")) {
        Install-WithWinget -Id "Git.Git" -Name "Git"
        Add-CommonToolPaths
    }

    if (-not (Test-CommandExists "docker")) {
        Install-WithWinget -Id "Docker.DockerDesktop" -Name "Docker Desktop"
        Add-CommonToolPaths
    }

    if (Test-CommandExists "wsl") {
        Write-Step "Checking WSL"
        & wsl --status | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "WSL status check failed. Trying wsl --install. A reboot may be required."
            & wsl --install
            Write-Warn "If WSL was just installed, reboot Windows and run this script again."
        }
    } else {
        Write-Warn "wsl.exe not found. Install WSL from Windows Features or run this script on a recent Windows 10/11 build."
    }
}

function Start-DockerDesktop {
    if (-not (Test-CommandExists "docker")) {
        throw "docker command is not available."
    }

    Write-Step "Starting Docker Desktop if needed"
    $dockerOk = $false
    try {
        & docker info *> $null
        $dockerOk = ($LASTEXITCODE -eq 0)
    } catch {
        $dockerOk = $false
    }

    if (-not $dockerOk) {
        $candidates = @(
            "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
            "${Env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe"
        )
        $dockerDesktop = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
        if ($dockerDesktop) {
            Start-Process $dockerDesktop
        } else {
            Write-Warn "Docker Desktop executable was not found. Start Docker Desktop manually if the next wait fails."
        }
    }

    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline) {
        try {
            & docker info *> $null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Docker is ready." -ForegroundColor Green
                return
            }
        } catch {
        }
        Start-Sleep -Seconds 5
    }

    throw "Docker did not become ready within 5 minutes."
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
OPENAI_API_KEY=${script:OpenAIApiKey}
OPENAI_BASE_URL=$OpenAIBaseUrl
OPENAI_MODEL=$OpenAIModel
OPENAI_TEMPERATURE=$OpenAITemperature
TAVILY_API_KEY=$TavilyApiKey
TAVILY_SEARCH_DEPTH=basic
TAVILY_MAX_RESULTS=5
"@

        Write-Step "Writing .env"
        Set-Content -Path ".env" -Value $envBody -Encoding UTF8
    } finally {
        Pop-Location
    }
}

function Ensure-CacheDirectories {
    Write-Step "Creating model cache directories"
    New-Item -ItemType Directory -Force "$HOME\.cache\huggingface" | Out-Null
    New-Item -ItemType Directory -Force "$HOME\.cache\sentence_transformers" | Out-Null
    $env:HOME = $HOME
}

function Start-Services {
    Push-Location $InstallDir
    try {
        if (-not $SkipBuild) {
            Write-Step "Building app image"
            Invoke-Checked "docker" @("compose", "build", "app") "docker compose build failed."
        }

        Write-Step "Starting qdrant and app"
        Invoke-Checked "docker" @("compose", "up", "-d", "qdrant", "app") "docker compose up failed."

        Write-Step "Current containers"
        Invoke-Checked "docker" @("compose", "ps") "docker compose ps failed."
    } finally {
        Pop-Location
    }
}

function Wait-HttpHealth {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 180
    )

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
    Push-Location $InstallDir
    try {
        Write-Step "Checking local RAG service health"
        Invoke-Checked "docker" @(
            "compose",
            "exec",
            "-T",
            "app",
            "python",
            "-m",
            "app.agent.services.local_rag_client",
            "health"
        ) "local RAG health check failed."
    } finally {
        Pop-Location
    }
}

function Index-LocalData {
    if (-not $IndexData) {
        Write-Step "Skipping data/raw indexing. Re-run with -IndexData when you want to build the local KB index."
        return
    }

    Push-Location $InstallDir
    try {
        if (-not (Test-Path "data\raw")) {
            Write-Warn "data\raw does not exist. Skipping index."
            return
        }

        Write-Step "Indexing data/raw"
        Invoke-Checked "docker" @(
            "compose",
            "exec",
            "-T",
            "app",
            "python",
            "-m",
            "app.rag.cli",
            "index-dir",
            "data/raw"
        ) "index-dir failed."

        Write-Step "Checking Qdrant index"
        Invoke-Checked "docker" @(
            "compose",
            "exec",
            "-T",
            "app",
            "python",
            "-m",
            "app.rag.cli",
            "check"
        ) "rag check failed."
    } finally {
        Pop-Location
    }
}

function Send-SmokeChat {
    if (-not $script:OpenAIApiKey) {
        Write-Warn "OPENAI_API_KEY is empty. Skipping /chat smoke request."
        return
    }

    Write-Step "Sending /chat smoke request"
    $body = @{
        question = "Give a brief overview of this project's goal and architecture from the knowledge base."
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

Write-Step "Agentic RAG Windows Docker bootstrap"
Write-Warn "Make sure the latest commits have been pushed to origin/$Branch before running this on a new machine."

Add-CommonToolPaths
Ensure-Prerequisites
Start-DockerDesktop
Sync-Repository
Ensure-EnvFile
Ensure-CacheDirectories
Start-Services
Wait-HttpHealth -Url "http://127.0.0.1:8000/health"
Check-LocalRagHealth
Index-LocalData
Send-SmokeChat

Write-Step "Done"
Write-Host "Project: $InstallDir" -ForegroundColor Green
Write-Host "API:     http://127.0.0.1:8000" -ForegroundColor Green
