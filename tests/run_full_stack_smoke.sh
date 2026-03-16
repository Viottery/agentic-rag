#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Starting Docker services in background"
docker compose up -d --build

echo "[2/5] Waiting for API health"
python - <<'PY'
import json
import time
import urllib.request

base_url = "http://127.0.0.1:8000"
deadline = time.time() + 120
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5.0) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
        if response.status == 200 and data.get("status") == "ok":
            print("[ok] health check passed")
            break
        last_error = f"unexpected response: {response.status} {body}"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(2)
else:
    raise SystemExit(f"[error] API not ready: {last_error}")
PY

echo "[3/5] Indexing local knowledge base"
docker compose exec app python -m app.rag.cli index-dir data/raw

echo "[4/5] Checking Qdrant collection state"
docker compose exec app python -m app.rag.cli check

echo "[5/5] Running end-to-end chat flow verification"
python tests/e2e_chat_flow.py --base-url "http://127.0.0.1:8000"

echo
echo "[done] Full stack smoke test finished successfully"
