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
        "aggregated_context": "",
        "evidence": [],
        "retrieved_docs": [],
        "retrieved_sources": [],
        "used_tools": [],
        "observation": "",
 
        # answer generation / validation
        "answer_draft": "",
        "checker_result": {
            "passed": False,
            "feedback": "",
            "pass_reason": "quality_fail",
        },
        "trace_summary": "",

        # runtime control
        "started_at": time.time(),
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
