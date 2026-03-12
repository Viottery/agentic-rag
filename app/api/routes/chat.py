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

    当前职责仅包括：
    - 接收用户问题
    - 构造初始 AgentState
    - 调用 LangGraph
    - 返回执行结果
    """
    initial_state: AgentState = {
        "question": req.question,
        "messages": [],

        "thought": "",
        "next_action": "",
        "action_input": "",
        "observation": "",

        "intermediate_steps": [],

        "answer": "",
        "status": "running",
    }

    result = agent_graph.invoke(initial_state)
    return result