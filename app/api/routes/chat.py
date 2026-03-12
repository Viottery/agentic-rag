from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.graph import agent_graph
from app.agent.state import AgentState

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
    initial_state: AgentState = {
        "question": req.question,
        "messages": [],

        "thought": "",
        "next_action": "",
        "action_input": "",
        "observation": "",

        "retrieved_docs": [],
        "tool_result": "",

        "intermediate_steps": [],

        "answer": "",
        "status": "running",
    }

    result = agent_graph.invoke(initial_state)
    return result