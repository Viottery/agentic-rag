from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_QUESTIONS = [
    "从知识库检索，详细介绍干员真理的技能信息，重点说明技能1和技能2。",
    "继续基于上一轮的知识库结果，补充她的职业分支、定位和剧情设定，不要重复太多技能细节。",
    "基于前两轮已经检索到的信息，把真理总结成一段简介，并说明她和凛冬的关系线索。",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe multi-turn /chat behavior with one conversation_id and retrieval-heavy questions.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI service base URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-turn timeout in seconds",
    )
    parser.add_argument(
        "--conversation-id",
        default="",
        help="Reuse a fixed conversation id. If omitted, one will be generated.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Turn question. Repeat to override the default 3-turn retrieval conversation.",
    )
    parser.add_argument(
        "--db-path",
        default="data/memory/conversations.db",
        help="Conversation SQLite path on the host filesystem.",
    )
    parser.add_argument(
        "--mode",
        choices=["wait"],
        default="wait",
        help="Request mode. Multi-turn probe currently uses wait mode for determinism.",
    )
    return parser


def _extract_output_text(payload: dict[str, Any]) -> str:
    for key in (
        "answer",
        "grounded_answer",
        "answer_draft",
        "result",
        "observation",
        "aggregated_context",
        "error",
    ):
        value = payload.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _request_json(url: str, *, data: dict[str, Any] | None = None, timeout: float = 180.0) -> tuple[int, dict[str, Any]]:
    encoded = None
    headers = {"Accept": "application/json"}
    if data is not None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=encoded, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(getattr(response, "status", 200))
            body = response.read().decode("utf-8")
            return status_code, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:  # noqa: BLE001
            payload = {"error": body}
        return int(exc.code), payload


def _read_db_snapshot(db_path: Path, conversation_id: str) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "db_exists": False,
            "conversation_id": conversation_id,
            "rolling_summary": "",
            "turn_count": 0,
            "recent_turns": [],
        }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conversation_row = conn.execute(
            """
            SELECT rolling_summary
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        turn_rows = conn.execute(
            """
            SELECT turn_id, user_question, assistant_answer, output_text, status, trace_summary, turn_summary
            FROM turns
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    return {
        "db_exists": True,
        "conversation_id": conversation_id,
        "rolling_summary": str(conversation_row["rolling_summary"] if conversation_row else "").strip(),
        "turn_count": len(turn_rows),
        "recent_turns": [
            {
                "turn_id": str(row["turn_id"] or ""),
                "user_question": str(row["user_question"] or ""),
                "assistant_answer": str(row["assistant_answer"] or row["output_text"] or ""),
                "status": str(row["status"] or ""),
                "trace_summary": str(row["trace_summary"] or ""),
                "turn_summary": str(row["turn_summary"] or ""),
            }
            for row in turn_rows[-5:]
        ],
    }


def _run_turn(base_url: str, *, question: str, conversation_id: str, timeout: float) -> dict[str, Any]:
    status_code, body = _request_json(
        f"{base_url}/chat",
        data={
            "question": question,
            "conversation_id": conversation_id,
            "mode": "wait",
        },
        timeout=timeout,
    )
    output_full = _extract_output_text(body)
    return {
        "status_code": status_code,
        "ok": status_code < 400,
        "response_status": str(body.get("status", "")),
        "question": question,
        "output_full": output_full,
        "answer_preview": output_full[:200],
        "raw_response": body,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    questions = args.question or list(DEFAULT_QUESTIONS)
    conversation_id = (args.conversation_id or "").strip() or f"multi-turn-probe-{uuid4().hex[:8]}"
    db_path = Path(args.db_path)

    log_dir = Path(tempfile.mkdtemp(prefix="multi-turn-chat-probe-"))
    log_path = log_dir / "multi_turn_probe.json"

    turns: list[dict[str, Any]] = []
    started_at = time.monotonic()

    for index, question in enumerate(questions, start=1):
        turn_started_at = time.monotonic()
        result = _run_turn(
            base_url,
            question=question,
            conversation_id=conversation_id,
            timeout=args.timeout,
        )
        db_snapshot = _read_db_snapshot(db_path, conversation_id)
        turn_finished_at = time.monotonic()
        turns.append(
            {
                "turn_index": index,
                "duration_seconds": round(turn_finished_at - turn_started_at, 3),
                **result,
                "db_snapshot": db_snapshot,
            }
        )

    finished_at = time.monotonic()
    payload = {
        "conversation_id": conversation_id,
        "turn_count": len(turns),
        "wall_time_seconds": round(finished_at - started_at, 3),
        "log_file": str(log_path),
        "turns": turns,
        "final_db_snapshot": _read_db_snapshot(db_path, conversation_id),
    }

    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
