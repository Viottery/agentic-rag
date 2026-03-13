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
        # user input
        "question": req.question,
        "messages": [],

        # react core
        "thought": "",
        "next_action": "",
        "action_input": "",
        "observation": "",

        # retrieval / tool raw outputs
        "retrieved_docs": [],
        "tool_result": "",

        # structured evidence
        "evidence": [],
        "retrieved_sources": [],
        "used_tools": [],

        # trace
        "intermediate_steps": [],
        "trace_summary": "",

        # runtime control
        "iteration_count": 0,
        "max_iterations": 3,
        "error": "",

        # final output
        "answer": "",
        "status": "running",
    }

    result = agent_graph.invoke(initial_state)
    return result