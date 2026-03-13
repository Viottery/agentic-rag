from typing import Any, Dict, List, Literal, TypedDict


class AgentStep(TypedDict):
    """单步 ReAct 轨迹。"""

    thought: str
    action: str
    action_input: str
    observation: str


class EvidenceItem(TypedDict, total=False):
    """结构化证据项。"""

    source_type: Literal["local_kb", "tool"]
    source_name: str
    source_id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]


class AgentState(TypedDict):
    """
    Agent 运行时共享状态。

    当前字段覆盖：
    - ReAct 最小闭环
    - 基础 RAG 结果传递
    - 基础调试与错误控制
    """

    # user input
    question: str
    messages: List[Dict[str, Any]]

    # react core
    thought: str
    next_action: str
    action_input: str
    observation: str

    # retrieval / tool raw outputs
    retrieved_docs: List[str]
    tool_result: str

    # structured evidence
    evidence: List[EvidenceItem]
    retrieved_sources: List[str]
    used_tools: List[str]

    # trace
    intermediate_steps: List[AgentStep]
    trace_summary: str

    # runtime control
    iteration_count: int
    max_iterations: int
    error: str

    # final output
    answer: str
    status: str