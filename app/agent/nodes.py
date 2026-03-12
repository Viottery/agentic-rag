from app.agent.prompt_loader import load_prompt
from app.agent.state import AgentState, AgentStep


def planner(state: AgentState) -> AgentState:
    """
    规划节点。

    当前阶段先实现最小可运行逻辑：
    - 基于问题生成简要 thought
    - 默认选择 respond
    - 记录一条 ReAct 中间轨迹

    后续这里会替换为真正的 LLM planner。
    """
    question = state["question"].strip()
    _ = load_prompt("planner.md")

    thought = "当前阶段采用最小闭环策略，先直接进入回答节点。"
    next_action = "respond"
    action_input = question

    step: AgentStep = {
        "thought": thought,
        "action": next_action,
        "action_input": action_input,
        "observation": "",
    }

    intermediate_steps = [*state["intermediate_steps"], step]

    return {
        **state,
        "thought": thought,
        "next_action": next_action,
        "action_input": action_input,
        "intermediate_steps": intermediate_steps,
    }


def responder(state: AgentState) -> AgentState:
    """
    回答节点。

    当前阶段不接入真实模型，先根据已有状态生成可验证的最终结果。
    后续可替换为基于 prompt 模板 + LLM 的回答生成逻辑。
    """
    question = state["question"].strip()
    thought = state["thought"].strip()
    _ = load_prompt("responder.md")

    answer = (
        "这是当前 Agent 工作流生成的初步回复。\n"
        f"- 用户问题：{question}\n"
        f"- 当前思考：{thought}"
    )

    return {
        **state,
        "answer": answer,
        "status": "finished",
    }