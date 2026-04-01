from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.graph import main_agent_graph
from app.agent.state_factory import build_initial_agent_state
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

    initial_state = build_initial_agent_state(
        req.question,
        max_iterations=settings.agent_max_iterations,
        max_duration_seconds=settings.agent_max_duration_seconds,
    )

    result = main_agent_graph.invoke(initial_state)
    return result
