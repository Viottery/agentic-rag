from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "pydantic_settings",
    types.SimpleNamespace(BaseSettings=object, SettingsConfigDict=lambda **kwargs: kwargs),
)

from app.runtime.conversation_store import ConversationStore


def test_conversation_store_persists_turns_and_loads_recent_context(tmp_path) -> None:
    store = ConversationStore(
        path=tmp_path / "conversations.db",
        recent_turn_limit=3,
        summary_turn_limit=5,
    )

    store.save_turn_result(
        conversation_id="conv-a",
        turn_id="turn-1",
        job_id="job-1",
        question="真理是谁？",
        result={
            "status": "finished",
            "answer": "真理是《明日方舟》中的辅助干员。",
            "trace_summary": "iterations=0 | tasks=1",
            "intermediate_steps": [{"action": "fast_answer"}],
            "started_at": "2026-04-01T11:00:00+00:00",
            "finished_at": "2026-04-01T11:00:01+00:00",
        },
    )

    bundle = store.load_context_bundle("conv-a")

    assert bundle.conversation_id == "conv-a"
    assert bundle.messages == [
        {"role": "user", "content": "真理是谁？"},
        {"role": "assistant", "content": "真理是《明日方舟》中的辅助干员。"},
    ]
    assert bundle.recent_turn_summaries
    assert "真理是谁？" in bundle.recent_turn_summaries[0]
    assert "真理是《明日方舟》中的辅助干员。" in bundle.conversation_summary
    assert bundle.memory_notes
    assert "真理" in bundle.memory_notes[0]


def test_conversation_store_updates_rolling_summary_from_multiple_turns(tmp_path) -> None:
    store = ConversationStore(
        path=tmp_path / "conversations.db",
        recent_turn_limit=4,
        summary_turn_limit=4,
    )

    store.save_turn_result(
        conversation_id="conv-b",
        turn_id="turn-1",
        job_id="job-1",
        question="HTTP是什么？",
        result={
            "status": "finished",
            "answer": "HTTP 是超文本传输协议。",
            "trace_summary": "iterations=0 | tasks=0",
            "intermediate_steps": [{"action": "fast_answer"}],
        },
    )
    store.save_turn_result(
        conversation_id="conv-b",
        turn_id="turn-2",
        job_id="job-2",
        question="TCP是什么？",
        result={
            "status": "finished",
            "answer": "TCP 是传输控制协议。",
            "trace_summary": "iterations=0 | tasks=0",
            "intermediate_steps": [{"action": "fast_answer"}],
        },
    )

    bundle = store.load_context_bundle("conv-b")

    assert len(bundle.messages) == 4
    assert len(bundle.recent_turn_summaries) == 2
    assert "HTTP是什么？" in bundle.conversation_summary
    assert "TCP是什么？" in bundle.conversation_summary


def test_conversation_store_recalls_query_relevant_memory_notes(tmp_path) -> None:
    store = ConversationStore(
        path=tmp_path / "conversations.db",
        recent_turn_limit=4,
        summary_turn_limit=4,
    )

    store.save_turn_result(
        conversation_id="conv-c",
        turn_id="turn-1",
        job_id="job-1",
        question="这个项目的架构是什么？",
        result={
            "status": "finished",
            "answer": "这个项目采用 fast gate、planner、execution agent 和 conversation queue。",
            "trace_summary": "iterations=1 | tasks=1",
            "intermediate_steps": [{"action": "planner"}],
        },
    )
    store.save_turn_result(
        conversation_id="conv-c",
        turn_id="turn-2",
        job_id="job-2",
        question="真理是谁？",
        result={
            "status": "finished",
            "answer": "真理是《明日方舟》中的辅助干员。",
            "trace_summary": "iterations=0 | tasks=0",
            "intermediate_steps": [{"action": "fast_answer"}],
        },
    )

    bundle = store.load_context_bundle("conv-c", query="项目架构怎么设计？")

    assert bundle.memory_notes
    assert "架构" in bundle.memory_notes[0]
