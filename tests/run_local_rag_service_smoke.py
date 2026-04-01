from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]

_INVOKE_SNIPPET = r"""
import asyncio
import json
import sys

from app.agent.services.local_rag_client import invoke_local_rag_client

payload = json.load(sys.stdin)
result = asyncio.run(invoke_local_rag_client(payload))
print(json.dumps(result, ensure_ascii=False))
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test the local RAG socket service from the host via docker compose.",
    )
    parser.add_argument(
        "--query",
        default="检索干员真理的技能信息",
        help="Query to send to the local RAG service.",
    )
    parser.add_argument(
        "--conversation-id",
        default="",
        help="Optional conversation id. Defaults to a generated smoke id.",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=int,
        default=30,
        help="Time budget passed to the local RAG service.",
    )
    parser.add_argument(
        "--service",
        default="app",
        help="Docker compose service name. Defaults to app.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full raw JSON response instead of a summary.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=12,
        help="Retry times when the dev server is reloading and the local socket is temporarily unavailable.",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=0.5,
        help="Seconds to wait between retries.",
    )
    return parser


def _run_docker_compose_exec(
    *,
    service: str,
    payload: dict[str, object],
    retries: int,
    retry_interval: float,
) -> dict[str, object]:
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        service,
        "python",
        "-c",
        _INVOKE_SNIPPET,
    ]
    last_error = ""
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            stdout = completed.stdout.strip()
            if not stdout:
                raise RuntimeError("local rag smoke returned empty stdout")
            return json.loads(stdout)

        details = completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}"
        last_error = details
        transient = "FileNotFoundError" in details or "No such file or directory" in details
        if attempt < attempts and transient:
            time.sleep(max(0.0, retry_interval))
            continue
        break

    raise RuntimeError(f"local rag smoke failed: {last_error}")


def _print_summary(response: dict[str, object]) -> None:
    trace = response.get("trace", [])
    evidence = response.get("evidence", [])
    route = {
        "source": response.get("routed_source_name", ""),
        "group": response.get("routed_top_level_group", ""),
        "scope": response.get("routed_hierarchy_scope", ""),
    }
    summary = {
        "status": response.get("status", ""),
        "question": response.get("question", ""),
        "rewritten_query": response.get("rewritten_query", ""),
        "route": route,
        "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        "sources": response.get("sources", []),
        "trace_actions": [item.get("action", "") for item in trace] if isinstance(trace, list) else [],
        "result_preview": str(response.get("result", ""))[:400],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if isinstance(evidence, list) and evidence:
        preview = []
        for item in evidence[:2]:
            preview.append(
                {
                    "source_id": item.get("source_id", ""),
                    "title": item.get("title", ""),
                    "score": item.get("score", 0.0),
                    "content_preview": str(item.get("content", ""))[:220],
                }
            )
        print("\nEvidence preview:")
        print(json.dumps(preview, ensure_ascii=False, indent=2))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = {
        "question": args.query,
        "conversation_id": args.conversation_id or f"smoke-{uuid4().hex[:8]}",
        "max_duration_seconds": args.max_duration_seconds,
    }

    response = _run_docker_compose_exec(
        service=args.service,
        payload=payload,
        retries=args.retries,
        retry_interval=args.retry_interval,
    )

    if args.raw:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        _print_summary(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
