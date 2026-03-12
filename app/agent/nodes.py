from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_model
from app.agent.prompt_loader import load_prompt
from app.agent.schemas import PlannerDecision
from app.agent.state import AgentState, AgentStep


def planner(state: AgentState) -> AgentState:
    """
    规划节点。

    使用 LLM 生成结构化决策，输出：
    - thought
    - next_action
    - action_input
    """
    question = state["question"].strip()
    prompt_template = load_prompt("planner.md")

    llm = get_chat_model().with_structured_output(PlannerDecision)

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(content=f"用户问题：{question}"),
    ]

    decision = llm.invoke(messages)

    step: AgentStep = {
        "thought": decision.thought,
        "action": decision.next_action,
        "action_input": decision.action_input,
        "observation": "",
    }

    intermediate_steps = [*state["intermediate_steps"], step]

    return {
        **state,
        "thought": decision.thought,
        "next_action": decision.next_action,
        "action_input": decision.action_input,
        "intermediate_steps": intermediate_steps,
    }


def retrieve(state: AgentState) -> AgentState:
    """
    检索节点。

    当前阶段使用 mock 检索结果，目标是验证：
    - 条件路由
    - observation 写回
    - responder 能消费检索结果
    """
    action_input = state["action_input"].strip()
    _ = load_prompt("retrieve.md")

    retrieved_docs = [
        f"Mock retrieval result for query: {action_input}",
        "Project focus: FastAPI + LangGraph + ReAct + RAG + tool calling.",
    ]
    observation = "已完成模拟检索，并生成检索上下文。"

    intermediate_steps = [*state["intermediate_steps"]]
    last_step = intermediate_steps[-1]
    last_step["observation"] = observation

    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "observation": observation,
        "intermediate_steps": intermediate_steps,
    }


def tool_executor(state: AgentState) -> AgentState:
    """
    工具执行节点。

    当前阶段使用 mock 工具结果，目标是验证：
    - planner -> tool 分支
    - tool result 写回 state
    """
    action_input = state["action_input"].strip()
    _ = load_prompt("tool_executor.md")

    tool_result = f"Mock tool executed successfully with input: {action_input}"
    observation = "已完成模拟工具调用，并返回工具结果。"

    intermediate_steps = [*state["intermediate_steps"]]
    last_step = intermediate_steps[-1]
    last_step["observation"] = observation

    return {
        **state,
        "tool_result": tool_result,
        "observation": observation,
        "intermediate_steps": intermediate_steps,
    }


def responder(state: AgentState) -> AgentState:
    """
    回答节点。

    汇总问题、规划结果、检索结果、工具结果，生成最终回答。
    """
    question = state["question"].strip()
    thought = state["thought"].strip()
    observation = state["observation"].strip()
    retrieved_docs = state["retrieved_docs"]
    tool_result = state["tool_result"].strip()

    prompt_template = load_prompt("responder.md")
    llm = get_chat_model()

    retrieved_text = "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "None"
    tool_text = tool_result if tool_result else "None"
    observation_text = observation if observation else "None"

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"当前思考：{thought}\n\n"
                f"观察结果：{observation_text}\n\n"
                f"检索结果：\n{retrieved_text}\n\n"
                f"工具结果：\n{tool_text}\n\n"
                "请基于以上状态生成最终回答。"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "answer": response.content,
        "status": "finished",
    }