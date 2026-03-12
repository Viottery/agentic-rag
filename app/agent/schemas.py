from typing import Literal

from pydantic import BaseModel, Field


class PlannerDecision(BaseModel):
    """规划节点的结构化决策结果。"""

    thought: str = Field(..., description="当前轮次的简要思考")
    next_action: Literal["respond", "retrieve", "tool"] = Field(
        ...,
        description="下一步动作",
    )
    action_input: str = Field(
        ...,
        description="传递给下一动作的输入",
    )