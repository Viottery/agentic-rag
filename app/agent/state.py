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


class FastPathDecisionState(TypedDict, total=False):
    """fast gate 的运行时决策。"""

    mode: Literal["direct_answer", "single_skill", "planner_loop"]
    reason: str
    executor: str
    question: str
    success_criteria: str


class SkillResult(TypedDict, total=False):
    """单次 skill 执行后的结构化结果。"""

    task_id: str
    executor: str
    status: Literal["done", "failed"]
    summary: str
    evidence_count: int
    source_count: int
    error: str


class ExecutionResult(TypedDict, total=False):
    """单次 execution agent 运行后的结构化结果。"""

    task_id: str
    executor: str
    status: Literal["done", "failed"]
    summary: str
    evidence_count: int
    source_count: int
    error: str
    evidence: List["EvidenceItem"]
    sources: List[str]
    retrieved_docs: List[str]
    retrieved_sources: List[str]
    used_tools: List[str]
    degraded: bool
    degraded_reason: str
    trace: List["AgentStep"]


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
    executor: str
    question: str
    success_criteria: str
    status: Literal["pending", "running", "done", "failed"]
    result: str
    evidence: List[EvidenceItem]
    sources: List[str]
    error: str
    degraded: bool
    degraded_reason: str
    rewritten_query: str
    sub_queries: List[str]
    rewrite_reason: str
    routed_source_name: str
    routed_top_level_group: str
    routed_hierarchy_scope: str
    route_reason: str


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
    conversation_id: str
    turn_id: str
    job_id: str
    question: str
    messages: List[Dict[str, Any]]
    conversation_summary: str
    recent_turn_summaries: List[str]
    memory_notes: List[str]

    # planner / orchestration
    fast_path_decision: FastPathDecisionState
    thought: str
    subtasks: List[SubTask]
    planner_control: PlannerControl
    current_task: SubTask
    intermediate_steps: List[AgentStep]
    execution_results: List[ExecutionResult]
    skill_results: List[SkillResult]

    # aggregated execution context
    kb_structure_summary: str
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


class SubtaskState(TypedDict):
    """
    execution agent 的子任务运行态。

    该状态只覆盖单个原子任务的执行图，方便未来单独调度、
    单独持久化，或切换到异步调用。
    """

    question: str
    thought: str
    current_task: SubTask
    subtasks: List[SubTask]
    intermediate_steps: List[AgentStep]
    kb_structure_summary: str
    aggregated_context: str
    evidence: List[EvidenceItem]
    retrieved_docs: List[str]
    retrieved_sources: List[str]
    used_tools: List[str]
    observation: str
    error: str
    started_at: str
    started_at_ts: float
    finished_at: str
    max_duration_seconds: int
    execution_result: ExecutionResult
    status: str
