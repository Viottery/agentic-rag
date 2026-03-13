from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_model
from app.agent.prompt_loader import load_prompt
from app.agent.schemas import PlannerDecision
from app.agent.state import AgentState, AgentStep, EvidenceItem


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

    当前阶段仍使用 mock 检索结果，但会返回：
    - retrieved_docs
    - evidence
    - retrieved_sources
    - observation

    这样后续替换为真实 RAG 时，不需要再重构 state 流转。
    """
    action_input = state["action_input"].strip()
    _ = load_prompt("retrieve.md")

    retrieved_docs = [
        f"Mock retrieval result for query: {action_input}",
        "Project focus: FastAPI + LangGraph + ReAct + RAG + tool calling.",
    ]

    evidence: list[EvidenceItem] = [
        {
            "source_type": "local_kb",
            "source_name": "mock_kb",
            "source_id": "mock_doc_1",
            "title": "Mock Retrieval Document 1",
            "content": retrieved_docs[0],
            "score": 0.95,
            "metadata": {"query": action_input, "rank": 1},
        },
        {
            "source_type": "local_kb",
            "source_name": "mock_kb",
            "source_id": "mock_doc_2",
            "title": "Mock Retrieval Document 2",
            "content": retrieved_docs[1],
            "score": 0.89,
            "metadata": {"query": action_input, "rank": 2},
        },
    ]

    retrieved_sources = [item["source_id"] for item in evidence]
    observation = "已完成模拟检索，并生成结构化证据。"

    intermediate_steps = [*state["intermediate_steps"]]
    if intermediate_steps:
        last_step = intermediate_steps[-1]
        last_step["observation"] = observation

    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "evidence": evidence,
        "retrieved_sources": retrieved_sources,
        "observation": observation,
        "intermediate_steps": intermediate_steps,
    }


def tool_executor(state: AgentState) -> AgentState:
    """
    工具执行节点。

    当前阶段仍使用 mock 工具结果，但会返回：
    - tool_result
    - used_tools
    - 可选 evidence
    - observation
    """
    action_input = state["action_input"].strip()
    _ = load_prompt("tool_executor.md")

    tool_name = "mock_tool"
    tool_result = f"Mock tool executed successfully with input: {action_input}"
    observation = "已完成模拟工具调用，并返回工具结果。"

    tool_evidence: EvidenceItem = {
        "source_type": "tool",
        "source_name": tool_name,
        "source_id": "mock_tool_result_1",
        "title": "Mock Tool Result",
        "content": tool_result,
        "score": 1.0,
        "metadata": {"tool_input": action_input},
    }

    intermediate_steps = [*state["intermediate_steps"]]
    if intermediate_steps:
        last_step = intermediate_steps[-1]
        last_step["observation"] = observation

    used_tools = [*state["used_tools"], tool_name]
    evidence = [*state["evidence"], tool_evidence]

    return {
        **state,
        "tool_result": tool_result,
        "used_tools": used_tools,
        "evidence": evidence,
        "observation": observation,
        "intermediate_steps": intermediate_steps,
    }


def responder(state: AgentState) -> AgentState:
    """
    回答节点。

    汇总问题、规划结果、检索结果、工具结果、结构化证据，生成最终回答。
    """
    question = state["question"].strip()
    thought = state["thought"].strip()
    observation = state["observation"].strip()
    retrieved_docs = state["retrieved_docs"]
    tool_result = state["tool_result"].strip()
    evidence = state["evidence"]
    retrieved_sources = state["retrieved_sources"]
    used_tools = state["used_tools"]

    prompt_template = load_prompt("responder.md")
    llm = get_chat_model()

    retrieved_text = "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "None"
    tool_text = tool_result if tool_result else "None"
    observation_text = observation if observation else "None"

    evidence_text = (
        "\n".join(
            f"- [{item.get('source_type', 'unknown')}] "
            f"{item.get('source_name', 'unknown')} / "
            f"{item.get('source_id', 'unknown')}: "
            f"{item.get('content', '')}"
            for item in evidence
        )
        if evidence
        else "None"
    )

    source_text = ", ".join(retrieved_sources) if retrieved_sources else "None"
    tool_list_text = ", ".join(used_tools) if used_tools else "None"

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"当前思考：{thought}\n\n"
                f"观察结果：{observation_text}\n\n"
                f"检索结果：\n{retrieved_text}\n\n"
                f"工具结果：\n{tool_text}\n\n"
                f"结构化证据：\n{evidence_text}\n\n"
                f"检索来源：{source_text}\n"
                f"已使用工具：{tool_list_text}\n\n"
                "请基于以上状态生成最终回答。"
            )
        ),
    ]

    response = llm.invoke(messages)

    trace_parts = []
    if thought:
        trace_parts.append(f"thought={thought}")
    if state["next_action"]:
        trace_parts.append(f"action={state['next_action']}")
    if retrieved_sources:
        trace_parts.append(f"sources={len(retrieved_sources)}")
    if used_tools:
        trace_parts.append(f"tools={','.join(used_tools)}")
    if observation:
        trace_parts.append(f"observation={observation}")

    trace_summary = " | ".join(trace_parts)

    return {
        **state,
        "trace_summary": trace_summary,
        "answer": response.content,
        "status": "finished",
    }