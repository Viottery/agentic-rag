from __future__ import annotations

from datetime import datetime

from langgraph.graph import END, StateGraph

from app.agent import nodes
from app.agent.skill_runtime import task_type_to_executor
from app.agent.state import ExecutionResult, SubtaskState


def _executor(state: SubtaskState) -> str:
    task = state.get("current_task", {})
    executor = task.get("executor", "").strip()
    if executor:
        return executor
    return task_type_to_executor(task.get("task_type", "rag"))


def bootstrap_subtask(state: SubtaskState) -> SubtaskState:
    """标准化子任务入口，作为未来异步执行的统一切点。"""
    task = {
        **state["current_task"],
        "executor": _executor(state),
        "status": "running",
    }
    return {
        **state,
        "current_task": task,
        "subtasks": [task],
        "status": "running",
    }


def route_after_bootstrap(state: SubtaskState) -> str:
    executor = _executor(state)
    if executor == "local_kb_retrieve":
        return "local_kb_retrieve_service"
    if executor == "tool_execute":
        return "action_agent"
    return "query_refiner"


def route_after_query_refiner(state: SubtaskState) -> str:
    executor = _executor(state)
    if executor == "web_search_retrieve":
        return "search_agent"
    return "action_agent"


def finalize_subtask(state: SubtaskState) -> SubtaskState:
    """收口单任务执行结果，方便主图同步或异步消费。"""
    task = state.get("current_task", {})
    result: ExecutionResult = {
        "task_id": task.get("task_id", ""),
        "executor": task.get("executor", ""),
        "status": task.get("status", "failed"),
        "summary": nodes._compact_task_result(task),  # noqa: SLF001
        "evidence_count": len(task.get("evidence", [])),
        "source_count": len(task.get("sources", [])),
        "error": task.get("error", ""),
        "evidence": task.get("evidence", []),
        "sources": task.get("sources", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "retrieved_sources": state.get("retrieved_sources", []),
        "used_tools": state.get("used_tools", []),
        "degraded": task.get("degraded", False),
        "degraded_reason": task.get("degraded_reason", ""),
        "trace": state.get("intermediate_steps", []),
    }
    return {
        **state,
        "execution_result": result,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "finished",
    }


def build_subtask_graph(*, async_mode: bool = False):
    """
    单原子任务执行图。

    当前结构：
        bootstrap
          ├─ tool_execute -> action_agent -> finalize
          └─ query_refiner
               ├─ local_kb_retrieve -> rag_router -> rag_agent -> finalize
               └─ web_search_retrieve -> search_agent -> finalize

    后续如果要做异步分发或多任务并行，主图只需要调度该图的运行实例。
    """
    graph = StateGraph(SubtaskState)
    local_kb_retrieve_node = (
        nodes.local_kb_retrieve_service_async
        if async_mode
        else nodes.local_kb_retrieve_service
    )
    query_refiner_node = nodes.query_refiner_async if async_mode else nodes.query_refiner
    rag_router_node = nodes.rag_router_async if async_mode else nodes.rag_router
    rag_agent_node = nodes.rag_agent_async if async_mode else nodes.rag_agent
    search_agent_node = nodes.search_agent_async if async_mode else nodes.search_agent
    action_agent_node = nodes.action_agent_async if async_mode else nodes.action_agent

    graph.add_node("bootstrap", bootstrap_subtask)
    graph.add_node("local_kb_retrieve_service", local_kb_retrieve_node)
    graph.add_node("query_refiner", query_refiner_node)
    graph.add_node("rag_router", rag_router_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("action_agent", action_agent_node)
    graph.add_node("finalize", finalize_subtask)

    graph.set_entry_point("bootstrap")

    graph.add_conditional_edges(
        "bootstrap",
        route_after_bootstrap,
        {
            "local_kb_retrieve_service": "local_kb_retrieve_service",
            "query_refiner": "query_refiner",
            "action_agent": "action_agent",
        },
    )

    graph.add_conditional_edges(
        "query_refiner",
        route_after_query_refiner,
        {
            "rag_router": "rag_router",
            "search_agent": "search_agent",
            "action_agent": "action_agent",
        },
    )

    graph.add_edge("local_kb_retrieve_service", "finalize")
    graph.add_edge("rag_router", "rag_agent")
    graph.add_edge("rag_agent", "finalize")
    graph.add_edge("search_agent", "finalize")
    graph.add_edge("action_agent", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


subtask_agent_graph = build_subtask_graph()
async_subtask_agent_graph = build_subtask_graph(async_mode=True)


def invoke_subtask_graph(state: SubtaskState) -> SubtaskState:
    """同步执行子任务图。"""
    return subtask_agent_graph.invoke(state)


async def ainvoke_subtask_graph(state: SubtaskState) -> SubtaskState:
    """异步执行子任务图，供未来并发调度使用。"""
    return await async_subtask_agent_graph.ainvoke(state)
