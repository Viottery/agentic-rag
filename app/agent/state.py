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

    设计上为后续 ReAct 扩展预留字段：
    - thought / next_action / action_input / observation
    - intermediate_steps 用于记录中间轨迹
    """

    question: str
    messages: List[Dict[str, Any]]

    thought: str
    next_action: str
    action_input: str
    observation: str

    intermediate_steps: List[AgentStep]

    answer: str
    status: str