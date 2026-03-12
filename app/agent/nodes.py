from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_model
from app.agent.prompt_loader import load_prompt
from app.agent.state import AgentState, AgentStep


def planner(state: AgentState) -> AgentState:
    """
    规划节点。

    当前阶段先保留规则式实现：
    - 生成简要 thought
    - 默认进入 respond
    - 记录一条中间轨迹

    后续再替换为基于 LLM 的 planner。
    """
    question = state["question"].strip()
    _ = load_prompt("planner.md")

    thought = "当前阶段先验证 LLM 驱动的回答链路，默认进入回答节点。"
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

    当前阶段由该节点直接调用 LLM 生成最终回复。
    """
    question = state["question"].strip()
    thought = state["thought"].strip()
    prompt_template = load_prompt("responder.md")

    llm = get_chat_model()

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"当前规划思考：{thought}\n\n"
                "请基于以上信息给出最终回答。"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "answer": response.content,
        "status": "finished",
    }