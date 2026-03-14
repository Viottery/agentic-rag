from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_model
from app.agent.prompt_loader import load_prompt
from app.agent.schemas import PlannerDecision
from app.agent.state import AgentState, AgentStep, EvidenceItem
from app.rag.retriever import retrieve_as_context


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

    当前阶段接入基础版真实向量检索，返回：
    - retrieved_docs
    - evidence
    - retrieved_sources
    - observation

    当前仍保持单轮、基础向量搜索的朴素实现。
    后续如果 retrieval subgraph 变复杂，这个节点可以继续保留，
    只需把内部调用切换到更完整的 retrieval workflow。
    """
    action_input = state["action_input"].strip()
    _ = load_prompt("retrieve.md")
    error = state["error"]

    try:
        retrieval_context = retrieve_as_context(action_input)
        retrieved_docs = retrieval_context["retrieved_docs"]
        retrieved_sources = retrieval_context["retrieved_sources"]
        evidence: list[EvidenceItem] = retrieval_context["evidence"]

        if retrieved_docs:
            observation = f"已完成知识库检索，命中 {len(retrieved_docs)} 条相关片段。"
        else:
            # 当前基础版检索命不中时，仍然让 agent 正常进入 responder，
            # 由 responder 基于“无可用检索结果”生成保守回答。
            observation = "已完成知识库检索，但未命中相关内容。"
    except Exception as exc:
        # 当前阶段先做节点级兜底，避免外部依赖异常直接打断整条 agent 链路。
        # 后续如果引入更完善的错误分类/监控，可以在这里细化异常类型。
        retrieved_docs = []
        retrieved_sources = []
        evidence = []
        observation = "知识库检索执行失败，已回退为无检索结果。"
        error = str(exc)

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
        "error": error,
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
