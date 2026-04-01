from typing import Literal

from pydantic import BaseModel, Field


class TaskItem(BaseModel):
    """planner 输出的单个子任务。"""

    task_id: str = Field(..., description="子任务唯一 ID")
    task_type: Literal["rag", "search", "action"] = Field(
        default="rag",
        description="子任务类型",
    )
    question: str = Field(..., description="交给子 agent 执行的问题")
    executor: str = Field(
        default="",
        description="执行该子任务的 skill / executor 名称；若留空则由运行时根据 task_type 推断。",
    )
    success_criteria: str = Field(
        default="",
        description="该子任务完成后应满足的简短成功标准。",
    )


class FastPathDecision(BaseModel):
    """fast gate 的分流决策。"""

    mode: Literal["direct_answer", "single_skill", "planner_loop"] = Field(
        ...,
        description="当前请求应走的主路径。",
    )
    reason: str = Field(
        default="",
        description="为什么这样分流。",
    )
    executor: str = Field(
        default="",
        description="当 mode=single_skill 时，直接执行的 skill 名称。",
    )
    question: str = Field(
        default="",
        description="当 mode=single_skill 时，交给该 skill 的任务问题。",
    )
    success_criteria: str = Field(
        default="",
        description="single_skill 路径下的成功标准。",
    )


class PlannerDecision(BaseModel):
    """规划节点的结构化决策结果。"""

    thought: str = Field(..., description="当前轮次的简要思考")
    decision: Literal["dispatch", "answer", "finish"] = Field(
        ...,
        description="planner 的当前决策",
    )
    selected_task_id: str = Field(
        default="",
        description="当前准备执行的子任务 ID；若无需执行则返回空字符串",
    )
    planner_note: str = Field(
        default="",
        description="给后续节点或下一轮 planner 的简短备注",
    )
    subtasks: list[TaskItem] = Field(
        default_factory=list,
        description="当前轮次生成或更新后的子任务列表",
    )


class CheckerDecision(BaseModel):
    """checker 节点的结构化输出。"""

    passed: bool = Field(..., description="当前答案是否通过检查")
    feedback: str = Field(
        ...,
        description="若未通过，说明缺失信息、逻辑问题或建议补充的任务方向",
    )


class QueryRewritePlan(BaseModel):
    """query 重写与拆分结果。"""

    rewritten_query: str = Field(
        ...,
        description="更适合检索或搜索的主查询",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="若问题较复杂，可拆出的 0-3 个子查询",
    )
    rewrite_reason: str = Field(
        default="",
        description="重写或拆分的简短原因说明",
    )


class SearchResultSelection(BaseModel):
    """搜索结果二次选择结果。"""

    selected_indices: list[int] = Field(
        default_factory=list,
        description="从候选结果中选出的结果序号列表，使用 1-based index",
    )
    rationale: str = Field(
        default="",
        description="简短说明为什么这些结果更值得抽取",
    )


class RAGRoutePlan(BaseModel):
    """本地 RAG 路由结果。"""

    source_name: str = Field(
        default="",
        description="优先命中的知识源名称；不确定时可留空",
    )
    top_level_group: str = Field(
        default="",
        description="优先检索的顶层分组；不确定时可留空",
    )
    hierarchy_scope: str = Field(
        default="",
        description="更细的层级范围；不确定时可留空",
    )
    rationale: str = Field(
        default="",
        description="简短说明为什么这样选择检索范围",
    )
