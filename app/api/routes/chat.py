from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel, Field
import time

from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.core.config import get_settings

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """聊天接口请求体。"""

    question: str = Field(..., min_length=1)


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    """
    Chat API 入口。

    负责构造初始状态并调用 LangGraph。
    """
    settings = get_settings()

    started_at_ts = time.time()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    initial_state: AgentState = {
        # user input
        "question": req.question,
        "messages": [],

        # planner / orchestration
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
        "max_iterations": settings.agent_max_iterations,
        "max_duration_seconds": settings.agent_max_duration_seconds,
        "error": "",

        # final output
        "answer": "",
        "status": "running",
    }

    result = agent_graph.invoke(initial_state)
    return result
