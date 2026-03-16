from typing import Any, Dict, List, Literal, TypedDict


class AgentStep(TypedDict):
    """单步编排轨迹。"""

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


class CitationItem(TypedDict, total=False):
    """答案段落映射出的引用项。"""

    paragraph_index: int
    label: str
    source_type: Literal["local_kb", "tool"]
    source_name: str
    source_id: str
    title: str
    snippet: str
    score: float
    degraded: bool


class SubTask(TypedDict, total=False):
    """planner 拆分出的子任务。"""

    task_id: str
    task_type: Literal["rag", "search", "action"]
    question: str
    status: Literal["pending", "running", "done", "failed"]
    result: str
    evidence: List[EvidenceItem]
    sources: List[str]
    error: str
    degraded: bool
    degraded_reason: str


class PlannerControl(TypedDict, total=False):
    """planner 当前轮次的控制信号。"""

    decision: Literal["dispatch", "answer", "finish"]
    selected_task_id: str
    planner_note: str
    checker_feedback: str
    force_answer_reason: str


class CheckerResult(TypedDict, total=False):
    """回答检查结果。"""

    passed: bool
    feedback: str
    pass_reason: Literal["quality_pass", "forced_budget_pass", "quality_fail"]


class VerificationResult(TypedDict, total=False):
    """引用与支持度校验结果。"""

    needs_revision: bool
    citation_coverage: float
    confidence: float
    supported_paragraphs: int
    total_paragraphs: int
    unsupported_claims: List[str]
    degraded_citations: List[str]
    summary: str


class AgentState(TypedDict):
    """
    Agent 运行时共享状态。

    当前字段覆盖：
    - planner 驱动的子任务编排
    - 基础 RAG / mock search / mock action 执行
    - answer generator + checker 闭环
    """

    # user input
    question: str
    messages: List[Dict[str, Any]]

    # planner / orchestration
    thought: str
    subtasks: List[SubTask]
    planner_control: PlannerControl
    current_task: SubTask
    intermediate_steps: List[AgentStep]

    # aggregated execution context
    aggregated_context: str
    evidence: List[EvidenceItem]
    retrieved_docs: List[str]
    retrieved_sources: List[str]
    used_tools: List[str]
    observation: str

    # answer generation / validation
    answer_draft: str
    grounded_answer: str
    citations: List[CitationItem]
    verification_result: VerificationResult
    checker_result: CheckerResult
    trace_summary: str

    # runtime control
    started_at: str
    started_at_ts: float
    finished_at: str
    iteration_count: int
    max_iterations: int
    max_duration_seconds: int
    error: str

    # final output
    answer: str
    status: str
