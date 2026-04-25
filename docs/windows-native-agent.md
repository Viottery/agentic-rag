# Windows Native Agent Mode

This mode runs on Windows without Docker and without WSL.

It intentionally disables the local KB/RAG path first, so the system can bring
up the agent runtime, chat endpoint, shell/action execution, conversation store,
and optional web search before dealing with Qdrant, indexing, and model-heavy
local retrieval.

## What Works First

- FastAPI app on `127.0.0.1:8000`
- `/health`
- `/chat`
- direct-answer path
- explicit shell/action path through PowerShell
- Tavily web search path when `TAVILY_API_KEY` is configured
- conversation persistence in `data/memory/conversations.db`

## What Is Disabled

- local RAG service startup
- Qdrant usage
- `data/raw` indexing
- embedding model preload
- reranker model preload

The script writes:

```dotenv
LOCAL_RAG_ENABLED=false
RERANKER_ENABLED=false
SHELL_PROVIDER=powershell
CONVERSATION_STORE_PATH=data/memory/conversations.db
```

## One-Shot Bootstrap

Download and run:

```powershell
Invoke-WebRequest `
  -UseBasicParsing `
  -Uri "https://raw.githubusercontent.com/Viottery/agentic-rag/master/scripts/setup-agentic-rag-windows-native.ps1" `
  -OutFile "$HOME\setup-agentic-rag-windows-native.ps1"

powershell -ExecutionPolicy Bypass -File "$HOME\setup-agentic-rag-windows-native.ps1" `
  -InstallPrereqs `
  -UseHttps `
  -OpenAIApiKey "<OPENAI_API_KEY>"
```

The script installs or checks:

- Git
- Python 3.11

It does not install Docker, WSL, or Qdrant.

## Local Clone Usage

```powershell
cd $HOME\agentic-rag
Set-ExecutionPolicy -Scope Process Bypass -Force

.\scripts\setup-agentic-rag-windows-native.ps1 `
  -UseHttps `
  -OpenAIApiKey "<OPENAI_API_KEY>"
```

Daily restart without reinstalling dependencies:

```powershell
.\scripts\setup-agentic-rag-windows-native.ps1 `
  -UseHttps `
  -SkipDependencyInstall
```

Recreate `.env`:

```powershell
.\scripts\setup-agentic-rag-windows-native.ps1 `
  -UseHttps `
  -RecreateEnv `
  -OpenAIApiKey "<OPENAI_API_KEY>"
```

## Manual Commands

Activate the virtual environment:

```powershell
cd $HOME\agentic-rag
.\.venv\Scripts\Activate.ps1
```

Run the app in the foreground:

```powershell
$env:OPENAI_API_KEY = "<OPENAI_API_KEY>"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
$env:OPENAI_TEMPERATURE = "0"
$env:LOCAL_RAG_ENABLED = "false"
$env:RERANKER_ENABLED = "false"
$env:SHELL_PROVIDER = "powershell"
$env:CONVERSATION_STORE_PATH = "data/memory/conversations.db"

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Stop the background app started by the setup script:

```powershell
Stop-Process -Id (Get-Content "$HOME\agentic-rag\.runtime\native-app.pid")
```

View logs:

```powershell
Get-Content "$HOME\agentic-rag\logs\native-app.out.log" -Tail 100 -Wait
Get-Content "$HOME\agentic-rag\logs\native-app.err.log" -Tail 100 -Wait
```

## Smoke Tests

API health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Shell/action path:

```powershell
$body = @{
  question = "command: Write-Output agentic-rag-native-ok"
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body $body
```

Direct-answer path:

```powershell
$body = @{
  question = "What is HTTP?"
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body $body
```

## Enabling Local RAG Later

Native local RAG is a separate step. It should be enabled only after the agent
path is healthy.

The intended follow-up work is:

- install model-heavy dependencies from `requirements.txt`
- choose either embedded `qdrant-client` local persistence or a real Qdrant
  server process
- set `LOCAL_RAG_ENABLED=true`
- configure `QDRANT_MODE`
- run indexing manually

Until then, local KB questions should return a controlled degraded result instead
of failing app startup.
