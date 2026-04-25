# Windows Deployment

This project is expected to run on Windows through Docker Desktop by default.
The app container is still Linux-based, which keeps Python, model, and system
package behavior close to the existing development environment.

If the target Windows machine cannot use Docker or WSL, use
[windows-native-agent.md](windows-native-agent.md) first. That native mode
skips Qdrant, indexing, and local KB retrieval so the agent, chat, shell/action,
and web-search paths can run.

Native Windows execution is being prepared through the runtime abstraction layer:

- shell provider: `SHELL_PROVIDER=auto` chooses PowerShell on Windows
- local RAG transport: `LOCAL_RAG_TRANSPORT=auto` chooses TCP on native Windows
- Docker-on-Windows still runs inside a Linux container, so the local RAG service
  normally reports `transport: unix`

## Prerequisites

- Windows 10/11
- PowerShell 5.1 or newer
- Docker Desktop with WSL2 backend
- Git

The setup script can install Git and Docker Desktop with `winget` when run with
`-InstallPrereqs`. If WSL2 is installed for the first time, reboot Windows and
run the script again.

## One-Shot Bootstrap

Make sure the latest commits have been pushed to `origin/master` before using
the raw GitHub URL on a fresh machine.

Download and run the bootstrap script:

```powershell
Invoke-WebRequest `
  -UseBasicParsing `
  -Uri "https://raw.githubusercontent.com/Viottery/agentic-rag/master/scripts/setup-agentic-rag-windows.ps1" `
  -OutFile "$HOME\setup-agentic-rag-windows.ps1"

powershell -ExecutionPolicy Bypass -File "$HOME\setup-agentic-rag-windows.ps1" `
  -InstallPrereqs `
  -UseHttps `
  -OpenAIApiKey "<OPENAI_API_KEY>"
```

The default install directory is:

```powershell
$HOME\agentic-rag
```

The script does the following:

- checks or installs Git and Docker Desktop
- starts Docker Desktop and waits until it is ready
- clones or updates the repository
- writes `.env` unless it already exists
- creates Hugging Face and SentenceTransformers cache folders
- builds the `app` image unless `-SkipBuild` is supplied
- starts `qdrant` and `app`
- checks `http://127.0.0.1:8000/health`
- checks local RAG service health
- sends a `/chat` smoke request when an API key is available

The script does not index `data/raw` by default.

## Running From A Local Clone

If the repository is already cloned:

```powershell
cd $HOME\agentic-rag
Set-ExecutionPolicy -Scope Process Bypass -Force

.\scripts\setup-agentic-rag-windows.ps1 `
  -UseHttps `
  -OpenAIApiKey "<OPENAI_API_KEY>"
```

To recreate `.env`:

```powershell
.\scripts\setup-agentic-rag-windows.ps1 `
  -UseHttps `
  -RecreateEnv `
  -OpenAIApiKey "<OPENAI_API_KEY>"
```

To skip rebuild and only update/start services:

```powershell
.\scripts\setup-agentic-rag-windows.ps1 `
  -UseHttps `
  -SkipBuild
```

## Optional Indexing

Indexing is intentionally manual. Run it only when you want to build or refresh
the local knowledge base from `data/raw`.

Through the setup script:

```powershell
.\scripts\setup-agentic-rag-windows.ps1 -IndexData
```

Direct Docker command:

```powershell
docker compose exec app python -m app.rag.cli index-dir data/raw
docker compose exec app python -m app.rag.cli check
```

## Daily Commands

Start services:

```powershell
cd $HOME\agentic-rag
docker compose up -d qdrant app
```

Stop services:

```powershell
cd $HOME\agentic-rag
docker compose down
```

Check containers:

```powershell
cd $HOME\agentic-rag
docker compose ps
```

Check API health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Check local RAG service health:

```powershell
docker compose exec app python -m app.agent.services.local_rag_client health
```

Send a chat request:

```powershell
$body = @{
  question = "Give a brief overview of this project's goal and architecture."
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body $body
```

Follow logs:

```powershell
docker compose logs -f app
```

Rebuild after dependency or Dockerfile changes:

```powershell
docker compose build app
docker compose up -d app
```

## Environment

The setup script writes this `.env` shape:

```dotenv
OPENAI_API_KEY=<OPENAI_API_KEY>
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0
TAVILY_API_KEY=
TAVILY_SEARCH_DEPTH=basic
TAVILY_MAX_RESULTS=5
```

Docker Compose also sets model-cache paths and Hugging Face mirror settings
inside the container.

## Troubleshooting

If Docker is not ready, start Docker Desktop manually and rerun the script:

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

If `docker compose build app` takes a long time, it is usually downloading
Python ML dependencies and model-related packages. The first build is expected
to be the slowest.

If clone over SSH fails, use HTTPS:

```powershell
.\scripts\setup-agentic-rag-windows.ps1 -UseHttps
```

If `.env` already exists and you need to change API settings:

```powershell
notepad .env
docker compose up -d app
```

If the app health check passes but chat fails, inspect logs:

```powershell
docker compose logs --tail 200 app
```

If local RAG health reports `transport: unix` on Docker Desktop, that is normal:
the service is running inside the Linux app container. Native Windows Python
deployment is the path where `LOCAL_RAG_TRANSPORT=auto` selects TCP.
