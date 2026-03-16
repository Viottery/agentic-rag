from __future__ import annotations

from datetime import datetime
import re
import time
from collections import OrderedDict

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_chat_model
from app.agent.prompt_loader import load_prompt
from app.agent.schemas import CheckerDecision, PlannerDecision, QueryRewritePlan
from app.agent.state import AgentState, AgentStep, CitationItem, EvidenceItem, SubTask
from app.core.config import get_settings
from app.rag.retriever import retrieve_as_context
from app.tools.tavily_search import tavily_search


def _format_subtasks(subtasks: list[SubTask]) -> str:
    if not subtasks:
        return "None"

    lines = []
    for task in subtasks:
        degraded_text = ""
        if task.get("degraded", False):
            degraded_text = f" degraded=true degraded_reason={task.get('degraded_reason', '')}"
        lines.append(
            (
                f"- id={task.get('task_id', '')} "
                f"type={task.get('task_type', '')} "
                f"status={task.get('status', '')} "
                f"question={task.get('question', '')} "
                f"result={_compact_task_result(task)}"
                f"{degraded_text}"
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
                "rewritten_query": prior.get("rewritten_query", ""),
                "sub_queries": prior.get("sub_queries", []),
                "rewrite_reason": prior.get("rewrite_reason", ""),
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
            f"{_compact_task_result(task)}"
        )
        for task in completed
    )


def _task_queries(task: SubTask) -> list[str]:
    queries: list[str] = []
    rewritten = task.get("rewritten_query", "").strip()
    if rewritten:
        queries.append(rewritten)

    for item in task.get("sub_queries", []):
        cleaned = item.strip()
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    if not queries and task.get("question", "").strip():
        queries.append(task["question"].strip())

    return queries[:3]


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip("?.。！？")


def _should_decompose_query(task_question: str) -> bool:
    lowered = task_question.lower()
    signals = ["以及", "并且", "和", "与", "及", "and", "or", ",", "，", "、", "vs"]
    has_signal = any(signal in lowered for signal in signals)
    return has_signal and len(_tokenize_for_match(task_question)) >= 12


def _truncate_text(text: str, limit: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _compact_task_result(task: SubTask) -> str:
    evidence = task.get("evidence", [])
    if evidence:
        snippets = []
        for item in evidence[:2]:
            title = item.get("title", "unknown")
            content = _truncate_text(item.get("content", ""), 140)
            snippets.append(f"- {title}: {content}")
        return "\n".join(snippets)

    return _truncate_text(task.get("result", ""), 220)


def _format_evidence_for_prompt(evidence_items: list[EvidenceItem], limit: int = 4) -> str:
    if not evidence_items:
        return "None"

    rendered = []
    seen: set[str] = set()
    for item in evidence_items:
        label = _citation_label(item)
        if label in seen:
            continue
        seen.add(label)
        rendered.append(
            f"- [{item.get('source_type', 'unknown')}] "
            f"{item.get('source_name', 'unknown')} / {label}: "
            f"{_truncate_text(item.get('content', ''), 160)}"
        )
        if len(rendered) >= limit:
            break

    return "\n".join(rendered) if rendered else "None"


def _domain_from_url(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1).lower() if match else ""


def _query_focus_terms(query: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query):
        lowered = token.lower()
        if lowered not in {"what", "current", "features", "capabilities", "limitations", "restrictions"}:
            terms.append(lowered)
    return terms[:3]


def _search_result_rank(item: dict, query: str) -> tuple[float, str]:
    url = str(item.get("url", "")).strip()
    title = str(item.get("title", "")).lower()
    content = str(item.get("content", "")).lower()
    domain = _domain_from_url(url)
    focus_terms = _query_focus_terms(query)

    score = 0.0
    if "docs." in domain:
        score += 3.0
    if any(term in domain for term in focus_terms):
        score += 2.5
    if any(term in title for term in focus_terms):
        score += 1.5
    if any(term in content for term in focus_terms):
        score += 0.5
    if domain.endswith(".com"):
        score += 0.2

    low_trust_domains = ["lobehub.com", "aiagentsdirectory.com", "docs.sim.ai"]
    if any(domain.endswith(item_domain) for item_domain in low_trust_domains):
        score -= 2.0

    return (score, url)


def _extract_quoted_text(text: str) -> str:
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)

    cn_match = re.search(r"[“‘]([^”’]+)[”’]", text)
    if cn_match:
        return cn_match.group(1)

    return ""


def _maybe_mock_action_result(action_input: str) -> str:
    quoted = _extract_quoted_text(action_input)

    if quoted and ("大写" in action_input or "uppercase" in action_input.lower()):
        transformed = quoted.upper()
        return (
            f"Mock action execution result:\n"
            f"- operation: uppercase\n"
            f"- input: {quoted}\n"
            f"- output: {transformed}\n"
            "当前结果由 mock action_agent 生成，用于验证 agent 编排与可信检查链路。"
        )

    return (
        f"Mock action executed successfully for task: {action_input}\n"
        "当前结果由 mock action_agent 生成，用于验证 agent 编排与可信检查链路。"
    )


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _tokenize_for_match(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _support_score(paragraph: str, evidence: EvidenceItem) -> float:
    paragraph_tokens = set(_tokenize_for_match(paragraph))
    evidence_tokens = set(_tokenize_for_match(evidence.get("content", "")))

    if not paragraph_tokens or not evidence_tokens:
        return 0.0

    overlap = paragraph_tokens & evidence_tokens
    base = len(overlap) / max(1, len(paragraph_tokens))
    semantic_hint = float(evidence.get("score", 0.0)) if evidence.get("score") is not None else 0.0
    return base + min(semantic_hint, 1.0) * 0.1


def _citation_label(evidence: EvidenceItem) -> str:
    source_id = evidence.get("source_id", "").strip()
    source_name = evidence.get("source_name", "").strip()

    if source_id:
        return source_id
    if source_name:
        return source_name
    return "unknown_source"


def _is_substantive_paragraph(paragraph: str, force_reason: str) -> bool:
    cleaned = paragraph.strip()
    if not cleaned:
        return False
    if force_reason and cleaned == force_reason.strip():
        return False
    return len(_tokenize_for_match(cleaned)) >= 6


def _mentions_limitations(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "mock",
        "模拟",
        "限制",
        "非实时",
        "最佳努力",
        "当前信息",
        "不完全",
        "可能",
    ]
    return any(marker in lowered for marker in markers)


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
    elapsed_seconds = max(0.0, time.time() - state["started_at_ts"])
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

    if elapsed_seconds >= state["max_duration_seconds"] and decision.decision != "finish":
        planner_control["decision"] = "answer"
        planner_control["planner_note"] = "达到最大思考时长，转入回答生成。"
        planner_control["force_answer_reason"] = (
            f"已达到最大思考时长限制（{state['max_duration_seconds']} 秒），"
            "系统需要基于当前信息给出最佳努力回答。"
        )
    elif next_iteration >= state["max_iterations"] and decision.decision != "finish":
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


def query_refiner(state: AgentState) -> AgentState:
    """为 rag/search 子任务重写并拆分查询。"""
    task = state["current_task"]
    task_type = task.get("task_type", "").strip() or "rag"
    task_question = task.get("question", "").strip()

    prompt_template = load_prompt("query_refiner.md")
    llm = get_chat_model().with_structured_output(QueryRewritePlan)
    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(
            content=(
                f"任务类型：{task_type}\n"
                f"任务问题：{task_question}\n\n"
                "请输出适合当前任务的主查询 rewritten_query，"
                "并在必要时给出 0-3 个 sub_queries。"
            )
        ),
    ]

    plan = llm.invoke(messages)
    task_is_cjk = _contains_cjk(task_question)

    rewritten_query = _normalize_query_text(plan.rewritten_query.strip() or task_question)
    if task_is_cjk and not _contains_cjk(rewritten_query):
        rewritten_query = _normalize_query_text(task_question)

    allow_decomposition = _should_decompose_query(task_question)
    sub_queries: list[str] = []
    if allow_decomposition:
        for item in plan.sub_queries:
            cleaned = _normalize_query_text(item)
            if not cleaned:
                continue
            if task_is_cjk and not _contains_cjk(cleaned):
                continue
            if cleaned == rewritten_query or cleaned in sub_queries:
                continue
            sub_queries.append(cleaned)

    updated_task: SubTask = {
        **task,
        "rewritten_query": rewritten_query,
        "sub_queries": sub_queries[:2],
        "rewrite_reason": plan.rewrite_reason.strip(),
    }
    subtasks = _replace_task(state, updated_task)

    observation = (
        f"已完成查询重写。主查询：{updated_task.get('rewritten_query', '')}；"
        f"子查询数：{len(updated_task.get('sub_queries', []))}。"
    )

    return {
        **state,
        "subtasks": subtasks,
        "current_task": updated_task,
        "observation": observation,
        "intermediate_steps": _append_step(
            state,
            "query_refiner",
            task_question,
            observation,
        ),
    }


def rag_agent(state: AgentState) -> AgentState:
    """本地知识库检索 agent。"""
    task = state["current_task"]
    queries = _task_queries(task)
    try:
        merged_docs: list[str] = []
        merged_sources: list[str] = []
        merged_evidence: list[EvidenceItem] = []
        seen_sources: set[str] = set()

        for query in queries:
            retrieval_context = retrieve_as_context(query)
            for doc in retrieval_context["retrieved_docs"]:
                if doc not in merged_docs:
                    merged_docs.append(doc)
            for source in retrieval_context["retrieved_sources"]:
                if source not in seen_sources:
                    seen_sources.add(source)
                    merged_sources.append(source)
            for item in retrieval_context["evidence"]:
                source_id = item.get("source_id", "")
                if source_id and source_id in seen_sources:
                    pass
                if not any(existing.get("source_id", "") == source_id for existing in merged_evidence):
                    merged_evidence.append(item)

        retrieved_docs = merged_docs[:4]
        retrieved_sources = merged_sources[:4]
        evidence = merged_evidence[:4]
        result = "\n".join(retrieved_docs) if retrieved_docs else "未命中相关知识库内容。"
        observation = f"RAG 子任务完成，执行 {len(queries)} 个查询，命中 {len(retrieved_docs)} 条片段。"
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
        "intermediate_steps": _append_step(state, "rag_agent", " | ".join(queries), observation),
    }


def search_agent(state: AgentState) -> AgentState:
    """信息获取类 agent，优先走 Tavily，缺失配置时回退到 mock。"""
    task = state["current_task"]
    queries = _task_queries(task)
    settings = get_settings()

    tool_name = "tavily_search" if settings.tavily_api_key.strip() else "mock_search"
    degraded = not settings.tavily_api_key.strip()
    evidence_items: list[EvidenceItem] = []

    if degraded:
        result = (
            f"Search findings for query: {queries[0]}\n"
            "1. OpenAI 近期主要产品线集中在通用对话模型、推理模型和轻量化模型。\n"
            "2. 代表性方向包括：高质量通用模型、强调多模态能力的模型，以及强调推理效率的小型模型。\n"
            "3. 对外描述通常会围绕响应质量、工具使用能力和成本效率展开。\n"
            "4. 当前结果为搜索代理的模拟回传，用于验证多 agent 编排与总结链路。"
        )
        observation = "搜索子任务已通过 mock 搜索执行完成，并返回结构化摘要。"
        evidence_items.append(
            {
                "source_type": "tool",
                "source_name": tool_name,
                "source_id": f"{task.get('task_id', 'search')}_result",
                "title": "Mock Search Result",
                "content": result,
                "score": 1.0,
                "metadata": {"query": queries[0], "degraded": True},
            }
        )
    else:
        rendered_parts: list[str] = []
        seen_urls: OrderedDict[str, None] = OrderedDict()
        result_limit = min(max(settings.tavily_max_results, 1), 3)
        counter = 1
        for query in queries:
            payload = tavily_search(
                query,
                api_key=settings.tavily_api_key,
                search_depth=settings.tavily_search_depth,
                max_results=settings.tavily_max_results,
            )
            results = sorted(
                payload.get("results", []),
                key=lambda item: _search_result_rank(item, query),
                reverse=True,
            )
            rendered_parts.append(f"Search results for query: {query}")
            for item in results:
                url = str(item.get("url", "")).strip()
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls[url] = None

                title = str(item.get("title", "")).strip() or f"Result {counter}"
                content = _truncate_text(str(item.get("content", "")).strip(), 220)
                rendered_parts.append(f"{counter}. {title} - {content}")
                evidence_items.append(
                    {
                        "source_type": "tool",
                        "source_name": tool_name,
                        "source_id": f"{task.get('task_id', 'search')}_result_{counter}",
                        "title": title,
                        "content": content,
                        "score": 1.0,
                        "metadata": {
                            "query": query,
                            "url": url,
                            "degraded": False,
                        },
                    }
                )
                counter += 1
                if counter > result_limit:
                    break
            if counter > result_limit:
                break

        result = "\n".join(rendered_parts) if rendered_parts else f"No Tavily results for query: {queries[0]}"
        observation = f"搜索子任务已通过 Tavily 执行完成，查询数 {len(queries)}，结果数 {len(evidence_items)}。"

    updated_task: SubTask = {
        **task,
        "status": "done",
        "result": result,
        "evidence": evidence_items,
        "sources": [tool_name],
        "error": "",
        "degraded": degraded,
        "degraded_reason": (
            "search_agent 当前为 mock，实现用于验证编排链路而非真实联网搜索。"
            if degraded
            else ""
        ),
    }
    subtasks = _replace_task(state, updated_task)

    return {
        **state,
        "subtasks": subtasks,
        "current_task": updated_task,
        "used_tools": [*state["used_tools"], tool_name],
        "evidence": [*state["evidence"], *evidence_items],
        "aggregated_context": _build_aggregated_context(subtasks),
        "observation": observation,
        "intermediate_steps": _append_step(state, "search_agent", " | ".join(queries), observation),
    }


def action_agent(state: AgentState) -> AgentState:
    """执行类 agent，当前为 mock action。"""
    task = state["current_task"]
    action_input = task.get("question", "").strip()

    tool_name = "mock_action"
    result = _maybe_mock_action_result(action_input)
    observation = "执行子任务已通过 mock action 完成。"

    evidence: EvidenceItem = {
        "source_type": "tool",
        "source_name": tool_name,
        "source_id": f"{task.get('task_id', 'action')}_result",
        "title": "Mock Action Result",
        "content": result,
        "score": 1.0,
        "metadata": {"action_input": action_input, "degraded": True},
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
    evidence_text = _format_evidence_for_prompt(state["evidence"])

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
        f"elapsed={max(0.0, time.time() - state['started_at_ts']):.1f}s"
    )

    return {
        **state,
        "answer_draft": answer_text,
        "grounded_answer": "",
        "citations": [],
        "verification_result": {
            "needs_revision": False,
            "citation_coverage": 0.0,
            "confidence": 0.0,
            "supported_paragraphs": 0,
            "total_paragraphs": 0,
            "unsupported_claims": [],
            "degraded_citations": [],
            "summary": "",
        },
        "trace_summary": trace_summary,
        "observation": "已生成答案草稿，等待引用映射与校验。",
        "intermediate_steps": _append_step(state, "answer_generator", question, "draft generated"),
    }


def citation_mapper(state: AgentState) -> AgentState:
    """将答案草稿的段落映射到最相关的证据，并补充引用标记。"""
    paragraphs = _split_paragraphs(state["answer_draft"])
    evidence_items = state["evidence"]
    force_reason = state["planner_control"].get("force_answer_reason", "").strip()
    completed_subtasks = [task for task in state["subtasks"] if task.get("status") == "done"]

    grounded_parts: list[str] = []
    citations: list[CitationItem] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        if not _is_substantive_paragraph(paragraph, force_reason):
            grounded_parts.append(paragraph)
            continue

        scoped_evidence = []
        if paragraph_index < len(completed_subtasks):
            scoped_evidence = completed_subtasks[paragraph_index].get("evidence", [])

        candidate_evidence = scoped_evidence or evidence_items
        scored = sorted(
            (
                (
                    _support_score(paragraph, evidence),
                    evidence,
                )
                for evidence in candidate_evidence
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        matched = [
            (score, evidence)
            for score, evidence in scored[:2]
            if score > 0
        ]

        if not matched and candidate_evidence is not evidence_items:
            scored = sorted(
                (
                    (
                        _support_score(paragraph, evidence),
                        evidence,
                    )
                    for evidence in evidence_items
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            matched = [
                (score, evidence)
                for score, evidence in scored[:2]
                if score > 0
            ]

        # Single-evidence answers are often short paraphrases; keep one fallback citation.
        if not matched and len(candidate_evidence) == 1:
            matched = [(0.01, candidate_evidence[0])]
        elif not matched and len(evidence_items) == 1:
            matched = [(0.01, evidence_items[0])]

        if not matched:
            grounded_parts.append(paragraph)
            continue

        labels: list[str] = []
        seen_labels: set[str] = set()
        for score, evidence in matched:
            label = _citation_label(evidence)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            labels.append(label)
            citations.append(
                {
                    "paragraph_index": paragraph_index,
                    "label": label,
                    "source_type": evidence.get("source_type", "tool"),
                    "source_name": evidence.get("source_name", ""),
                    "source_id": evidence.get("source_id", ""),
                    "title": evidence.get("title", ""),
                    "snippet": evidence.get("content", "")[:180],
                    "score": round(score, 4),
                    "degraded": bool(evidence.get("metadata", {}).get("degraded", False)),
                }
            )

        citation_text = " ".join(f"[{label}]" for label in labels)
        grounded_parts.append(f"{paragraph} {citation_text}")

    grounded_answer = "\n\n".join(grounded_parts).strip()

    return {
        **state,
        "grounded_answer": grounded_answer,
        "citations": citations,
        "observation": "已完成答案段落与证据的引用映射。",
        "intermediate_steps": _append_step(
            state,
            "citation_mapper",
            state["question"],
            f"mapped {len(citations)} citations",
        ),
    }


def verifier(state: AgentState) -> AgentState:
    """对 grounded answer 做轻量引用覆盖与支持度校验。"""
    answer_text = state["grounded_answer"].strip() or state["answer_draft"].strip()
    paragraphs = _split_paragraphs(answer_text)
    force_reason = state["planner_control"].get("force_answer_reason", "").strip()

    citations_by_paragraph: dict[int, list[CitationItem]] = {}
    for citation in state["citations"]:
        paragraph_index = int(citation.get("paragraph_index", -1))
        citations_by_paragraph.setdefault(paragraph_index, []).append(citation)

    total_paragraphs = 0
    supported_paragraphs = 0
    unsupported_claims: list[str] = []
    degraded_citations: list[str] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        if not _is_substantive_paragraph(paragraph, force_reason):
            continue

        total_paragraphs += 1
        paragraph_citations = citations_by_paragraph.get(paragraph_index, [])

        if paragraph_citations:
            supported_paragraphs += 1
            for citation in paragraph_citations:
                if citation.get("degraded"):
                    degraded_citations.append(citation.get("label", "unknown_source"))
        else:
            unsupported_claims.append(paragraph)

    citation_coverage = (
        supported_paragraphs / total_paragraphs if total_paragraphs else 1.0
    )
    mentions_limits = _mentions_limitations(answer_text)
    degraded_ack_ok = not degraded_citations or mentions_limits
    needs_revision = bool(unsupported_claims) or citation_coverage < 0.8 or not degraded_ack_ok

    confidence = citation_coverage * 0.7
    if not unsupported_claims:
        confidence += 0.2
    if degraded_ack_ok:
        confidence += 0.1
    confidence = round(min(confidence, 1.0), 3)

    summary_parts = [
        f"citation_coverage={citation_coverage:.2f}",
        f"supported_paragraphs={supported_paragraphs}/{total_paragraphs}",
    ]
    if degraded_citations:
        summary_parts.append(f"degraded_sources={','.join(sorted(set(degraded_citations)))}")
    if unsupported_claims:
        summary_parts.append("存在未被引用支持的段落")
    if not degraded_ack_ok:
        summary_parts.append("答案使用了降级来源，但没有明确说明限制")

    verification_result = {
        "needs_revision": needs_revision,
        "citation_coverage": round(citation_coverage, 3),
        "confidence": confidence,
        "supported_paragraphs": supported_paragraphs,
        "total_paragraphs": total_paragraphs,
        "unsupported_claims": unsupported_claims,
        "degraded_citations": sorted(set(degraded_citations)),
        "summary": " | ".join(summary_parts),
    }

    return {
        **state,
        "verification_result": verification_result,
        "trace_summary": f"{state['trace_summary']} | coverage={citation_coverage:.2f} | confidence={confidence:.2f}",
        "observation": "已完成轻量引用覆盖与支持度校验。",
        "intermediate_steps": _append_step(
            state,
            "verifier",
            state["question"],
            verification_result["summary"],
        ),
    }


def checker(state: AgentState) -> AgentState:
    """检查答案草稿是否可以输出，否则返回 planner 补充信息。"""
    question = state["question"].strip()
    force_reason = state["planner_control"].get("force_answer_reason", "").strip()
    final_answer_candidate = state["grounded_answer"].strip() or state["answer_draft"].strip()

    if force_reason:
        return {
            **state,
            "checker_result": {
                "passed": True,
                "feedback": force_reason,
                "pass_reason": "forced_budget_pass",
            },
            "answer": final_answer_candidate,
            "status": "finished",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "observation": "因达到流程限制条件，checker 直接放行当前最佳努力答案。",
            "intermediate_steps": _append_step(
                state,
                "checker",
                question,
                f"forced pass: {force_reason}",
            ),
        }

    verification_result = state["verification_result"]
    if verification_result.get("needs_revision", False):
        feedback = (
            verification_result.get("summary", "引用覆盖不足或存在未支持结论。")
        )
        return {
            **state,
            "checker_result": {
                "passed": False,
                "feedback": feedback,
                "pass_reason": "quality_fail",
            },
            "answer": "",
            "status": "running",
            "observation": "verifier 发现引用或支持度不足，返回 planner 补充信息。",
            "intermediate_steps": _append_step(
                state,
                "checker",
                question,
                feedback,
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
                f"带引用答案：\n{state['grounded_answer'] or 'None'}\n\n"
                f"子任务执行情况：\n{_format_subtasks(state['subtasks'])}\n\n"
                f"汇总上下文：\n{state['aggregated_context'] or 'None'}\n\n"
                f"引用校验摘要：{verification_result.get('summary', 'None')}\n"
                f"引用覆盖率：{verification_result.get('citation_coverage', 0.0)}\n"
                f"置信度：{verification_result.get('confidence', 0.0)}\n"
            )
        ),
    ]

    decision = llm.invoke(messages)
    status = "finished" if decision.passed else "running"
    answer = final_answer_candidate if decision.passed else ""

    return {
        **state,
        "checker_result": {
            "passed": decision.passed,
            "feedback": decision.feedback,
            "pass_reason": "quality_pass" if decision.passed else "quality_fail",
        },
        "answer": answer,
        "status": status,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds") if decision.passed else "",
        "observation": "checker 通过，准备输出最终答案。" if decision.passed else "checker 未通过，返回 planner。",
        "intermediate_steps": _append_step(
            state,
            "checker",
            question,
            decision.feedback,
        ),
    }
