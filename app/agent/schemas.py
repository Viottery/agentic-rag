from typing import Literal

from pydantic import BaseModel, Field


class TaskItem(BaseModel):
    """planner 输出的单个子任务。"""

    task_id: str = Field(..., description="子任务唯一 ID")
    task_type: Literal["rag", "search", "action"] = Field(
        ...,
        description="子任务类型",
    )
    question: str = Field(..., description="交给子 agent 执行的问题")


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
