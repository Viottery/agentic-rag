from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_model
from app.agent.prompt_loader import load_prompt
from app.agent.schemas import CheckerDecision, PlannerDecision
from app.agent.state import AgentState, AgentStep, EvidenceItem, SubTask
from app.rag.retriever import retrieve_as_context


def _format_subtasks(subtasks: list[SubTask]) -> str:
    if not subtasks:
        return "None"

    lines = []
    for task in subtasks:
        lines.append(
            (
                f"- id={task.get('task_id', '')} "
                f"type={task.get('task_type', '')} "
                f"status={task.get('status', '')} "
                f"question={task.get('question', '')} "
                f"result={task.get('result', '')}"
            )
        )
    return "\n".join(lines)


def _merge_subtasks(existing: list[SubTask], proposed: list[dict]) -> list[SubTask]:
    existing_by_id = {task.get("task_id", ""): task for task in existing if task.get("task_id")}
    merged: list[SubTask] = []
    seen_ids: set[str] = set()

    for item in proposed:
        task_id = item.get("task_id", "").strip()
        if not task_id:
            continue
        seen_ids.add(task_id)

        prior = existing_by_id.get(task_id, {})
        merged.append(
            {
                "task_id": task_id,
                "task_type": item.get("task_type", prior.get("task_type", "rag")),
                "question": item.get("question", prior.get("question", "")),
                "status": prior.get("status", "pending"),
                "result": prior.get("result", ""),
                "evidence": prior.get("evidence", []),
                "sources": prior.get("sources", []),
                "error": prior.get("error", ""),
                "degraded": prior.get("degraded", False),
                "degraded_reason": prior.get("degraded_reason", ""),
            }
        )

    for task in existing:
        task_id = task.get("task_id", "")
        if task_id and task_id not in seen_ids:
            merged.append(task)

    if merged:
        return merged

    return existing


def _find_selected_task(subtasks: list[SubTask], task_id: str) -> SubTask:
    for task in subtasks:
        if task.get("task_id") == task_id:
            return task
    return {}


def _set_task_status(state: AgentState, task_id: str, status: str) -> list[SubTask]:
    updated: list[SubTask] = []
    for task in state["subtasks"]:
        new_task = {**task}
        if task.get("task_id") == task_id:
            new_task["status"] = status
        updated.append(new_task)
    return updated


def _replace_task(state: AgentState, replacement: SubTask) -> list[SubTask]:
    updated: list[SubTask] = []
    replaced = False
    for task in state["subtasks"]:
        if task.get("task_id") == replacement.get("task_id"):
            updated.append(replacement)
            replaced = True
        else:
            updated.append(task)

    if not replaced and replacement.get("task_id"):
        updated.append(replacement)

    return updated


def _append_step(state: AgentState, action: str, action_input: str, observation: str) -> list[AgentStep]:
    step: AgentStep = {
        "thought": state["thought"],
        "action": action,
        "action_input": action_input,
        "observation": observation,
    }
    return [*state["intermediate_steps"], step]


def _build_aggregated_context(subtasks: list[SubTask]) -> str:
    completed = [task for task in subtasks if task.get("status") == "done"]
    if not completed:
        return ""

    return "\n\n".join(
        (
            f"[{task.get('task_type', 'unknown')}] {task.get('question', '')}\n"
            f"{task.get('result', '')}"
        )
        for task in completed
    )


def planner(state: AgentState) -> AgentState:
    """
    planner 节点。

    负责：
    - 初次拆分子任务
    - 根据已完成任务决定是否继续 dispatch
    - 根据 checker 反馈决定是否回补信息
    """
    question = state["question"].strip()
    prompt_template = load_prompt("planner.md")
    llm = get_chat_model().with_structured_output(PlannerDecision)

    next_iteration = state["iteration_count"] + 1
    elapsed_seconds = max(0.0, time.time() - state["started_at"])
    checker_feedback = state["checker_result"].get("feedback", "")
    subtasks_text = _format_subtasks(state["subtasks"])
    current_answer = state["answer_draft"].strip() or "None"
    current_context = state["aggregated_context"].strip() or "None"

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"当前子任务：\n{subtasks_text}\n\n"
                f"已汇总上下文：\n{current_context}\n\n"
                f"当前答案草稿：\n{current_answer}\n\n"
                f"Checker 反馈：{checker_feedback or 'None'}\n\n"
                f"当前迭代：{next_iteration} / {state['max_iterations']}\n"
                f"已用时：{elapsed_seconds:.1f} / {state['max_duration_seconds']} 秒\n"
            )
        ),
    ]

    decision = llm.invoke(messages)
    merged_subtasks = _merge_subtasks(
        state["subtasks"],
        [item.model_dump() for item in decision.subtasks],
    )

    selected_task = _find_selected_task(merged_subtasks, decision.selected_task_id)
    planner_control = {
        "decision": decision.decision,
        "selected_task_id": decision.selected_task_id,
        "planner_note": decision.planner_note,
        "checker_feedback": checker_feedback,
        "force_answer_reason": "",
    }

    if elapsed_seconds >= state["max_duration_seconds"] and decision.decision == "dispatch":
        planner_control["decision"] = "answer"
        planner_control["planner_note"] = "达到最大思考时长，转入回答生成。"
        planner_control["force_answer_reason"] = (
            f"已达到最大思考时长限制（{state['max_duration_seconds']} 秒），"
            "系统需要基于当前信息给出最佳努力回答。"
        )
    elif next_iteration >= state["max_iterations"] and decision.decision == "dispatch":
        planner_control["decision"] = "answer"
        planner_control["planner_note"] = "达到最大迭代次数，转入回答生成。"
        planner_control["force_answer_reason"] = (
            f"已达到最大思考轮数限制（{state['max_iterations']} 轮），"
            "系统需要基于当前信息给出最佳努力回答。"
        )

    return {
        **state,
        "thought": decision.thought,
        "subtasks": merged_subtasks,
        "planner_control": planner_control,
        "current_task": selected_task,
        "iteration_count": next_iteration,
        "intermediate_steps": _append_step(
            state,
            "planner",
            decision.selected_task_id or decision.decision,
            decision.planner_note,
        ),
    }


def dispatcher(state: AgentState) -> AgentState:
    """
    dispatcher 节点。

    当前先做串行调度，但接口保留为 task-based，
    方便未来平滑升级为并行分发。
    """
    task_id = state["planner_control"].get("selected_task_id", "")
    selected_task = _find_selected_task(state["subtasks"], task_id)

    if not selected_task:
        observation = "planner 未选出可执行子任务，回退到回答阶段。"
        planner_control = {
            **state["planner_control"],
            "decision": "answer",
            "planner_note": observation,
        }
        return {
            **state,
            "planner_control": planner_control,
            "observation": observation,
            "current_task": {},
            "intermediate_steps": _append_step(state, "dispatcher", task_id, observation),
        }

    subtasks = _set_task_status(state, task_id, "running")

    return {
        **state,
        "subtasks": subtasks,
        "current_task": {**selected_task, "status": "running"},
        "observation": f"已分发子任务 {task_id} 到 {selected_task.get('task_type', 'unknown')} agent。",
        "intermediate_steps": _append_step(
            state,
            "dispatcher",
            f"{task_id}:{selected_task.get('task_type', '')}",
            "task dispatched",
        ),
    }


def rag_agent(state: AgentState) -> AgentState:
    """本地知识库检索 agent。"""
    task = state["current_task"]
    query = task.get("question", "").strip()
    try:
        retrieval_context = retrieve_as_context(query)
        retrieved_docs = retrieval_context["retrieved_docs"]
        retrieved_sources = retrieval_context["retrieved_sources"]
        evidence: list[EvidenceItem] = retrieval_context["evidence"]
        result = "\n".join(retrieved_docs) if retrieved_docs else "未命中相关知识库内容。"
        observation = f"RAG 子任务完成，命中 {len(retrieved_docs)} 条片段。"
        task_error = ""
        status = "done"
    except Exception as exc:
        retrieved_docs = []
        retrieved_sources = []
        evidence = []
        result = "知识库检索失败，已降级为空结果。"
        observation = "RAG 子任务执行失败。"
        task_error = str(exc)
        status = "failed"

    updated_task: SubTask = {
        **task,
        "status": status,
        "result": result,
        "evidence": evidence,
        "sources": retrieved_sources,
        "error": task_error,
    }
    subtasks = _replace_task(state, updated_task)

    return {
        **state,
        "subtasks": subtasks,
        "current_task": updated_task,
        "retrieved_docs": retrieved_docs,
        "retrieved_sources": retrieved_sources,
        "evidence": [*state["evidence"], *evidence],
        "aggregated_context": _build_aggregated_context(subtasks),
        "observation": observation,
        "error": task_error,
        "intermediate_steps": _append_step(state, "rag_agent", query, observation),
    }


def search_agent(state: AgentState) -> AgentState:
    """信息获取类 agent，当前为 mock 搜索。"""
    task = state["current_task"]
    query = task.get("question", "").strip()

    tool_name = "mock_search"
    result = (
        f"Search findings for query: {query}\n"
        "1. OpenAI 近期主要产品线集中在通用对话模型、推理模型和轻量化模型。\n"
        "2. 代表性方向包括：高质量通用模型、强调多模态能力的模型，以及强调推理效率的小型模型。\n"
        "3. 对外描述通常会围绕响应质量、工具使用能力和成本效率展开。\n"
        "4. 当前结果为搜索代理的模拟回传，用于验证多 agent 编排与总结链路。"
    )
    observation = "搜索子任务已通过 mock 搜索执行完成，并返回结构化摘要。"

    evidence: EvidenceItem = {
        "source_type": "tool",
        "source_name": tool_name,
        "source_id": f"{task.get('task_id', 'search')}_result",
        "title": "Mock Search Result",
        "content": result,
        "score": 1.0,
        "metadata": {"query": query},
    }

    updated_task: SubTask = {
        **task,
        "status": "done",
        "result": result,
        "evidence": [evidence],
        "sources": [tool_name],
        "error": "",
        "degraded": True,
        "degraded_reason": "search_agent 当前为 mock，实现用于验证编排链路而非真实联网搜索。",
    }
    subtasks = _replace_task(state, updated_task)

    return {
        **state,
        "subtasks": subtasks,
        "current_task": updated_task,
        "used_tools": [*state["used_tools"], tool_name],
        "evidence": [*state["evidence"], evidence],
        "aggregated_context": _build_aggregated_context(subtasks),
        "observation": observation,
        "intermediate_steps": _append_step(state, "search_agent", query, observation),
    }


def action_agent(state: AgentState) -> AgentState:
    """执行类 agent，当前为 mock action。"""
    task = state["current_task"]
    action_input = task.get("question", "").strip()

    tool_name = "mock_action"
    result = f"Mock action executed successfully for task: {action_input}"
    observation = "执行子任务已通过 mock action 完成。"

    evidence: EvidenceItem = {
        "source_type": "tool",
        "source_name": tool_name,
        "source_id": f"{task.get('task_id', 'action')}_result",
        "title": "Mock Action Result",
        "content": result,
        "score": 1.0,
        "metadata": {"action_input": action_input},
    }

    updated_task: SubTask = {
        **task,
        "status": "done",
        "result": result,
        "evidence": [evidence],
        "sources": [tool_name],
        "error": "",
        "degraded": True,
        "degraded_reason": "action_agent 当前为 mock，实现用于验证编排链路而非真实执行。",
    }
    subtasks = _replace_task(state, updated_task)

    return {
        **state,
        "subtasks": subtasks,
        "current_task": updated_task,
        "used_tools": [*state["used_tools"], tool_name],
        "evidence": [*state["evidence"], evidence],
        "aggregated_context": _build_aggregated_context(subtasks),
        "observation": observation,
        "intermediate_steps": _append_step(state, "action_agent", action_input, observation),
    }


def answer_generator(state: AgentState) -> AgentState:
    """汇总已有任务结果，生成答案草稿。"""
    question = state["question"].strip()
    prompt_template = load_prompt("answer_generator.md")
    llm = get_chat_model()

    subtasks_text = _format_subtasks(state["subtasks"])
    evidence_text = (
        "\n".join(
            f"- [{item.get('source_type', 'unknown')}] "
            f"{item.get('source_name', 'unknown')}: {item.get('content', '')}"
            for item in state["evidence"]
        )
        if state["evidence"]
        else "None"
    )

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"子任务执行情况：\n{subtasks_text}\n\n"
                f"已汇总上下文：\n{state['aggregated_context'] or 'None'}\n\n"
                f"结构化证据：\n{evidence_text}\n"
            )
        ),
    ]

    response = llm.invoke(messages)
    force_reason = state["planner_control"].get("force_answer_reason", "").strip()
    answer_text = response.content
    if force_reason:
        answer_text = f"{force_reason}\n\n{answer_text}"
    trace_summary = (
        f"iterations={state['iteration_count']} | "
        f"tasks={len(state['subtasks'])} | "
        f"done={sum(1 for task in state['subtasks'] if task.get('status') == 'done')} | "
        f"tools={','.join(state['used_tools']) if state['used_tools'] else 'None'} | "
        f"elapsed={max(0.0, time.time() - state['started_at']):.1f}s"
    )

    return {
        **state,
        "answer_draft": answer_text,
        "trace_summary": trace_summary,
        "observation": "已生成答案草稿，等待 checker 校验。",
        "intermediate_steps": _append_step(state, "answer_generator", question, "draft generated"),
    }


def checker(state: AgentState) -> AgentState:
    """检查答案草稿是否可以输出，否则返回 planner 补充信息。"""
    question = state["question"].strip()
    force_reason = state["planner_control"].get("force_answer_reason", "").strip()

    if force_reason:
        return {
            **state,
            "checker_result": {
                "passed": True,
                "feedback": force_reason,
                "pass_reason": "forced_budget_pass",
            },
            "answer": state["answer_draft"].strip(),
            "status": "finished",
            "observation": "因达到流程限制条件，checker 直接放行当前最佳努力答案。",
            "intermediate_steps": _append_step(
                state,
                "checker",
                question,
                f"forced pass: {force_reason}",
            ),
        }

    prompt_template = load_prompt("checker.md")
    llm = get_chat_model().with_structured_output(CheckerDecision)

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"答案草稿：\n{state['answer_draft'] or 'None'}\n\n"
                f"子任务执行情况：\n{_format_subtasks(state['subtasks'])}\n\n"
                f"汇总上下文：\n{state['aggregated_context'] or 'None'}\n"
            )
        ),
    ]

    decision = llm.invoke(messages)
    status = "finished" if decision.passed else "running"
    answer = state["answer_draft"] if decision.passed else ""

    return {
        **state,
        "checker_result": {
            "passed": decision.passed,
            "feedback": decision.feedback,
            "pass_reason": "quality_pass" if decision.passed else "quality_fail",
        },
        "answer": answer,
        "status": status,
        "observation": "checker 通过，准备输出最终答案。" if decision.passed else "checker 未通过，返回 planner。",
        "intermediate_steps": _append_step(
            state,
            "checker",
            question,
            decision.feedback,
        ),
    }
