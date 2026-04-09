from __future__ import annotations

from datetime import datetime
import time

from app.agent.skill_runtime import task_type_to_executor
from app.agent.state import AgentState, ExecutionResult, SubTask, SubtaskState


def build_initial_agent_state(
    question: str,
    *,
    max_iterations: int,
    max_duration_seconds: int,
    conversation_id: str = "",
    turn_id: str = "",
    job_id: str = "",
    messages: list[dict] | None = None,
    conversation_summary: str = "",
    recent_turn_summaries: list[str] | None = None,
    memory_notes: list[str] | None = None,
) -> AgentState:
    """构造 /chat 主图的初始运行态。"""
    started_at_ts = time.time()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    return {
        # user input
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "job_id": job_id,
        "question": question,
        "messages": list(messages or []),
        "conversation_summary": conversation_summary,
        "recent_turn_summaries": list(recent_turn_summaries or []),
        "memory_notes": list(memory_notes or []),

        # planner / orchestration
        "fast_path_decision": {
            "mode": "planner_loop",
            "reason": "",
            "executor": "",
            "question": "",
            "success_criteria": "",
        },
        "thought": "",
        "subtasks": [],
        "planner_control": {
            "decision": "dispatch",
            "selected_task_id": "",
            "planner_note": "",
            "checker_feedback": "",
            "force_answer_reason": "",
        },
        "current_task": {},
        "intermediate_steps": [],
        "execution_results": [],
        "skill_results": [],

        # aggregated execution context
        "kb_structure_summary": "",
        "aggregated_context": "",
        "evidence": [],
        "retrieved_docs": [],
        "retrieved_sources": [],
        "used_tools": [],
        "observation": "",

        # answer generation / validation
        "answer_draft": "",
        "grounded_answer": "",
        "citations": [],
        "verification_result": {
            "needs_revision": False,
            "citation_coverage": 0.0,
            "confidence": 0.0,
            "supported_paragraphs": 0,
            "total_paragraphs": 0,
            "unsupported_claims": [],
            "degraded_citations": [],
            "summary": "",
        },
        "checker_result": {
            "passed": False,
            "feedback": "",
            "pass_reason": "quality_fail",
        },
        "trace_summary": "",

        # runtime control
        "started_at": started_at,
        "started_at_ts": started_at_ts,
        "finished_at": "",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "max_duration_seconds": max_duration_seconds,
        "error": "",

        # final output
        "answer": "",
        "status": "running",
    }


def build_subtask_initial_state(parent_state: AgentState, task: SubTask) -> SubtaskState:
    """为 execution agent 构造单任务子图的初始状态。"""
    normalized_task: SubTask = {
        **task,
        "executor": task.get("executor", "").strip() or task_type_to_executor(task.get("task_type", "rag")),
        "status": "running",
        "result": task.get("result", ""),
        "evidence": [],
        "sources": [],
        "error": "",
        "degraded": False,
        "degraded_reason": "",
        "rewritten_query": task.get("rewritten_query", ""),
        "sub_queries": task.get("sub_queries", []),
        "rewrite_reason": task.get("rewrite_reason", ""),
        "routed_source_name": task.get("routed_source_name", ""),
        "routed_top_level_group": task.get("routed_top_level_group", ""),
        "routed_hierarchy_scope": task.get("routed_hierarchy_scope", ""),
        "route_reason": task.get("route_reason", ""),
    }
    empty_result: ExecutionResult = {
        "task_id": normalized_task.get("task_id", ""),
        "executor": normalized_task.get("executor", ""),
        "status": "failed",
        "summary": "",
        "evidence_count": 0,
        "source_count": 0,
        "error": "",
        "evidence": [],
        "sources": [],
        "retrieved_docs": [],
        "retrieved_sources": [],
        "used_tools": [],
        "degraded": False,
        "degraded_reason": "",
        "trace": [],
    }

    return {
        "question": parent_state.get("question", ""),
        "thought": "",
        "current_task": normalized_task,
        "subtasks": [normalized_task],
        "intermediate_steps": [],
        "kb_structure_summary": parent_state.get("kb_structure_summary", ""),
        "aggregated_context": "",
        "evidence": [],
        "retrieved_docs": [],
        "retrieved_sources": [],
        "used_tools": [],
        "observation": "",
        "error": "",
        "started_at": parent_state.get("started_at", ""),
        "started_at_ts": parent_state.get("started_at_ts", time.time()),
        "finished_at": "",
        "max_duration_seconds": parent_state.get("max_duration_seconds", 90),
        "execution_result": empty_result,
        "status": "running",
    }
