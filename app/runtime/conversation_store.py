from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any

from app.core.config import get_settings


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _extract_output_text(result: dict[str, Any]) -> str:
    for key in (
        "answer",
        "grounded_answer",
        "answer_draft",
        "result",
        "observation",
        "aggregated_context",
        "error",
    ):
        value = result.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_turn_summary(question: str, result: dict[str, Any]) -> str:
    output_text = _extract_output_text(result)
    trace_summary = str(result.get("trace_summary", "")).strip()
    status = str(result.get("status", "")).strip() or "unknown"
    actions: list[str] = []
    for item in result.get("intermediate_steps", []):
        action = str(item.get("action", "")).strip()
        if action and action not in actions:
            actions.append(action)

    parts = [
        f"Q: {question.strip()}",
        f"status={status}",
    ]
    if actions:
        parts.append(f"actions={','.join(actions[:8])}")
    if trace_summary:
        parts.append(trace_summary)
    degraded_items = [
        str(item.get("degraded_reason", "")).strip()
        for item in result.get("execution_results", [])
        if str(item.get("degraded_reason", "")).strip()
    ]
    if degraded_items:
        parts.append(f"limits={'; '.join(degraded_items[:2])}")
    if output_text:
        compact_output = " ".join(output_text.split())
        parts.append(f"answer={compact_output[:240]}")
    return " | ".join(parts)[:600]


def _tokenize_recall_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _infer_memory_kind(question: str, result: dict[str, Any], content: str) -> str:
    combined = f"{question} {content}".lower()
    if any(keyword in combined for keyword in ["偏好", "prefer", "喜欢", "不喜欢"]):
        return "preference"
    if any(keyword in combined for keyword in ["约束", "必须", "不要", "禁止", "constraint", "requirement"]):
        return "constraint"
    if any(keyword in combined for keyword in ["决定", "采用", "选择", "decision"]):
        return "decision"
    if any(keyword in combined for keyword in ["todo", "待办", "后续", "下一步", "open question"]):
        return "todo"
    if any(keyword in combined for keyword in ["项目", "架构", "仓库", "代码", "project", "architecture"]):
        return "project_context"
    if str(result.get("status", "")).strip() in {"failed", "degraded"}:
        return "open_question"
    return "fact"


def _extract_memory_candidates(
    *,
    turn_id: str,
    question: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    settings = get_settings()
    output_text = " ".join(_extract_output_text(result).split()).strip()
    if not output_text:
        return []

    candidates: list[dict[str, Any]] = []
    kind = _infer_memory_kind(question, result, output_text)
    content = output_text[: settings.conversation_memory_note_max_chars]
    title = " ".join(question.split())[:80]
    importance = 3 if kind in {"decision", "constraint", "project_context"} else 2
    candidates.append(
        {
            "memory_id": f"{turn_id}:answer",
            "scope": "conversation",
            "kind": kind,
            "title": title or "Conversation note",
            "content": content,
            "tags": [kind],
            "source_turn_ids": [turn_id],
            "importance": importance,
        }
    )

    checker_feedback = str(result.get("checker_result", {}).get("feedback", "")).strip()
    if checker_feedback:
        candidates.append(
            {
                "memory_id": f"{turn_id}:followup",
                "scope": "conversation",
                "kind": "open_question",
                "title": f"Follow-up from {title or turn_id}"[:80],
                "content": checker_feedback[: settings.conversation_memory_note_max_chars],
                "tags": ["open_question", "checker_feedback"],
                "source_turn_ids": [turn_id],
                "importance": 2,
            }
        )

    degraded_reasons = [
        str(item.get("degraded_reason", "")).strip()
        for item in result.get("execution_results", [])
        if str(item.get("degraded_reason", "")).strip()
    ]
    if degraded_reasons:
        candidates.append(
            {
                "memory_id": f"{turn_id}:constraint",
                "scope": "conversation",
                "kind": "constraint",
                "title": f"Runtime limits for {title or turn_id}"[:80],
                "content": "; ".join(degraded_reasons)[: settings.conversation_memory_note_max_chars],
                "tags": ["constraint", "runtime"],
                "source_turn_ids": [turn_id],
                "importance": 3,
            }
        )

    return candidates[: max(1, settings.conversation_memory_candidate_limit)]


def _compress_rolling_summary(lines: list[str], *, max_chars: int) -> str:
    if not lines:
        return ""

    selected: list[str] = []
    seen: set[str] = set()
    current_length = 0
    for line in reversed(lines):
        cleaned = " ".join(str(line).split()).strip()
        if not cleaned:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        candidate = f"- {cleaned}"
        projected = current_length + len(candidate) + (1 if selected else 0)
        if selected and projected > max_chars:
            break
        seen.add(normalized)
        selected.append(candidate)
        current_length = projected
    return "\n".join(reversed(selected))


def _memory_relevance(note_text: str, query: str) -> float:
    query_tokens = set(_tokenize_recall_text(query))
    if not query_tokens:
        return 0.0
    note_tokens = set(_tokenize_recall_text(note_text))
    if not note_tokens:
        return 0.0
    return len(query_tokens & note_tokens) / len(query_tokens)


@dataclass
class ConversationContextBundle:
    conversation_id: str
    messages: list[dict[str, Any]]
    conversation_summary: str
    recent_turn_summaries: list[str]
    memory_notes: list[str]


class ConversationStore:
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        recent_turn_limit: int | None = None,
        summary_turn_limit: int | None = None,
    ) -> None:
        settings = get_settings()
        self._path = Path(path or settings.conversation_store_path)
        self._recent_turn_limit = max(1, recent_turn_limit or settings.conversation_recent_turns)
        self._summary_turn_limit = max(1, summary_turn_limit or settings.conversation_summary_turns)
        self._lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        rolling_summary TEXT NOT NULL DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        turn_id TEXT NOT NULL UNIQUE,
                        conversation_id TEXT NOT NULL,
                        job_id TEXT NOT NULL DEFAULT '',
                        user_question TEXT NOT NULL,
                        assistant_answer TEXT NOT NULL DEFAULT '',
                        output_text TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT '',
                        trace_summary TEXT NOT NULL DEFAULT '',
                        turn_summary TEXT NOT NULL DEFAULT '',
                        intermediate_steps_json TEXT NOT NULL DEFAULT '[]',
                        raw_result_json TEXT NOT NULL DEFAULT '{}',
                        started_at TEXT NOT NULL DEFAULT '',
                        finished_at TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_turns_conversation_id_id
                    ON turns(conversation_id, id DESC);

                    CREATE TABLE IF NOT EXISTS memory_notes (
                        memory_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        scope TEXT NOT NULL DEFAULT 'conversation',
                        kind TEXT NOT NULL DEFAULT 'fact',
                        title TEXT NOT NULL DEFAULT '',
                        content TEXT NOT NULL DEFAULT '',
                        tags_json TEXT NOT NULL DEFAULT '[]',
                        source_turn_ids_json TEXT NOT NULL DEFAULT '[]',
                        importance INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_memory_notes_conversation_id_updated_at
                    ON memory_notes(conversation_id, updated_at DESC);
                    """
                )
                conn.commit()
            self._initialized = True

    def load_context_bundle(
        self,
        conversation_id: str,
        query: str = "",
    ) -> ConversationContextBundle:
        self.initialize()
        cleaned_conversation_id = conversation_id.strip()
        if not cleaned_conversation_id:
            return ConversationContextBundle(
                conversation_id="",
                messages=[],
                conversation_summary="",
                recent_turn_summaries=[],
                memory_notes=[],
            )

        with self._connect() as conn:
            conversation_row = conn.execute(
                """
                SELECT rolling_summary
                FROM conversations
                WHERE conversation_id = ?
                """,
                (cleaned_conversation_id,),
            ).fetchone()

            turn_rows = conn.execute(
                """
                SELECT user_question, assistant_answer, output_text, turn_summary
                FROM turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (cleaned_conversation_id, self._recent_turn_limit),
            ).fetchall()

            memory_rows = conn.execute(
                """
                SELECT kind, title, content, importance, updated_at
                FROM memory_notes
                WHERE conversation_id = ?
                ORDER BY updated_at DESC
                LIMIT 24
                """,
                (cleaned_conversation_id,),
            ).fetchall()

        messages: list[dict[str, Any]] = []
        recent_turn_summaries: list[str] = []
        for row in reversed(turn_rows):
            user_question = str(row["user_question"] or "").strip()
            assistant_answer = str(row["assistant_answer"] or row["output_text"] or "").strip()
            turn_summary = str(row["turn_summary"] or "").strip()
            if user_question:
                messages.append({"role": "user", "content": user_question})
            if assistant_answer:
                messages.append({"role": "assistant", "content": assistant_answer})
            if turn_summary:
                recent_turn_summaries.append(turn_summary)

        ranked_memory_notes: list[tuple[float, str]] = []
        for row in memory_rows:
            rendered = (
                f"[{str(row['kind'] or 'fact').strip() or 'fact'}] "
                f"{str(row['title'] or '').strip()}: "
                f"{str(row['content'] or '').strip()}"
            ).strip()
            if not rendered:
                continue
            score = float(row["importance"] or 1)
            if query.strip():
                score += _memory_relevance(rendered, query) * 10.0
            ranked_memory_notes.append((score, rendered))

        ranked_memory_notes.sort(key=lambda item: item[0], reverse=True)
        memory_notes = [
            note
            for _, note in ranked_memory_notes[: get_settings().conversation_memory_recall_limit]
        ]

        return ConversationContextBundle(
            conversation_id=cleaned_conversation_id,
            messages=messages,
            conversation_summary=str(conversation_row["rolling_summary"] if conversation_row else "").strip(),
            recent_turn_summaries=recent_turn_summaries,
            memory_notes=memory_notes,
        )

    def save_turn_result(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        job_id: str,
        question: str,
        result: dict[str, Any],
    ) -> None:
        self.initialize()
        cleaned_conversation_id = conversation_id.strip()
        if not cleaned_conversation_id or not turn_id.strip():
            return

        created_at = _iso_now()
        output_text = _extract_output_text(result)
        assistant_answer = str(result.get("answer", "")).strip() or output_text
        trace_summary = str(result.get("trace_summary", "")).strip()
        turn_summary = _build_turn_summary(question, result)
        status = str(result.get("status", "")).strip() or "unknown"
        intermediate_steps_json = json.dumps(result.get("intermediate_steps", []), ensure_ascii=False)
        raw_result_json = json.dumps(result, ensure_ascii=False)
        started_at = str(result.get("started_at", "")).strip()
        finished_at = str(result.get("finished_at", "")).strip()
        title = " ".join(question.strip().split())[:80]

        with self._lock:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                    (cleaned_conversation_id,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO conversations (
                            conversation_id, title, created_at, updated_at, rolling_summary
                        ) VALUES (?, ?, ?, ?, '')
                        """,
                        (cleaned_conversation_id, title, created_at, created_at),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE conversations
                        SET updated_at = ?, title = CASE WHEN title = '' THEN ? ELSE title END
                        WHERE conversation_id = ?
                        """,
                        (created_at, title, cleaned_conversation_id),
                    )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO turns (
                        turn_id,
                        conversation_id,
                        job_id,
                        user_question,
                        assistant_answer,
                        output_text,
                        status,
                        trace_summary,
                        turn_summary,
                        intermediate_steps_json,
                        raw_result_json,
                        started_at,
                        finished_at,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_id,
                        cleaned_conversation_id,
                        job_id,
                        question,
                        assistant_answer,
                        output_text,
                        status,
                        trace_summary,
                        turn_summary,
                        intermediate_steps_json,
                        raw_result_json,
                        started_at,
                        finished_at,
                        created_at,
                    ),
                )

                memory_candidates = _extract_memory_candidates(
                    turn_id=turn_id,
                    question=question,
                    result=result,
                )
                for item in memory_candidates:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO memory_notes (
                            memory_id,
                            conversation_id,
                            scope,
                            kind,
                            title,
                            content,
                            tags_json,
                            source_turn_ids_json,
                            importance,
                            created_at,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["memory_id"],
                            cleaned_conversation_id,
                            item["scope"],
                            item["kind"],
                            item["title"],
                            item["content"],
                            json.dumps(item["tags"], ensure_ascii=False),
                            json.dumps(item["source_turn_ids"], ensure_ascii=False),
                            item["importance"],
                            created_at,
                            created_at,
                        ),
                    )

                recent_summary_rows = conn.execute(
                    """
                    SELECT turn_summary
                    FROM turns
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (cleaned_conversation_id, self._summary_turn_limit),
                ).fetchall()
                rolling_summary = _compress_rolling_summary(
                    [
                        str(row["turn_summary"]).strip()
                        for row in recent_summary_rows
                        if str(row["turn_summary"]).strip()
                    ],
                    max_chars=get_settings().conversation_summary_max_chars,
                )

                conn.execute(
                    """
                    UPDATE conversations
                    SET rolling_summary = ?, updated_at = ?
                    WHERE conversation_id = ?
                    """,
                    (rolling_summary, created_at, cleaned_conversation_id),
                )
                conn.commit()


_STORE: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    global _STORE
    if _STORE is None:
        _STORE = ConversationStore()
    return _STORE
