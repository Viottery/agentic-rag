from typing import Any, Dict, List, TypedDict


class AgentStep(TypedDict):
    """单步 ReAct 轨迹。"""

    thought: str
    action: str
    action_input: str
    observation: str


class AgentState(TypedDict):
    """
    Agent 运行时共享状态。

    当前字段已经覆盖 ReAct 最小闭环，并为后续 RAG / Tool 扩展预留接口。
    """

    question: str
    messages: List[Dict[str, Any]]

    thought: str
    next_action: str
    action_input: str
    observation: str

    retrieved_docs: List[str]
    tool_result: str

    intermediate_steps: List[AgentStep]

    answer: str
    status: str