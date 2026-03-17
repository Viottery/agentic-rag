from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    action_agent,
    answer_generator,
    checker,
    citation_mapper,
    dispatcher,
    planner,
    query_refiner,
    rag_agent,
    rag_router,
    search_agent,
    verifier,
)
from app.agent.state import AgentState


def route_after_planner(state: AgentState) -> str:
    decision = state["planner_control"].get("decision", "answer")

    if decision == "dispatch":
        return "dispatcher"
    if decision == "finish":
        return "end"
    return "answer_generator"


def route_after_dispatcher(state: AgentState) -> str:
    task_type = state["current_task"].get("task_type", "")

    if task_type == "rag":
        return "query_refiner"
    if task_type == "search":
        return "query_refiner"
    if task_type == "action":
        return "action_agent"
    return "planner"


def route_after_query_refiner(state: AgentState) -> str:
    task_type = state["current_task"].get("task_type", "")

    if task_type == "rag":
        return "rag_router"
    if task_type == "search":
        return "search_agent"
    return "planner"


def route_after_checker(state: AgentState) -> str:
    if state["checker_result"].get("passed", False):
        return "end"
    return "planner"


def build_graph():
    """
    构建 planner 驱动的 supervisor 风格 agent 图。

    当前链路：
        planner
          ├─ dispatcher -> query_refiner -> rag_router -> rag_agent    -> planner
          ├─ dispatcher -> query_refiner -> search_agent -> planner
          ├─ dispatcher -> action_agent -> planner
          ├─ answer_generator -> citation_mapper -> verifier -> checker -> END
          └─ answer_generator -> citation_mapper -> verifier -> checker -> planner
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("dispatcher", dispatcher)
    graph.add_node("query_refiner", query_refiner)
    graph.add_node("rag_router", rag_router)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("search_agent", search_agent)
    graph.add_node("action_agent", action_agent)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("citation_mapper", citation_mapper)
    graph.add_node("verifier", verifier)
    graph.add_node("checker", checker)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "dispatcher": "dispatcher",
            "answer_generator": "answer_generator",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "dispatcher",
        route_after_dispatcher,
        {
            "query_refiner": "query_refiner",
            "action_agent": "action_agent",
            "planner": "planner",
        },
    )

    graph.add_conditional_edges(
        "query_refiner",
        route_after_query_refiner,
        {
            "rag_router": "rag_router",
            "search_agent": "search_agent",
            "planner": "planner",
        },
    )

    graph.add_edge("rag_router", "rag_agent")
    graph.add_edge("rag_agent", "planner")
    graph.add_edge("search_agent", "planner")
    graph.add_edge("action_agent", "planner")
    graph.add_edge("answer_generator", "citation_mapper")
    graph.add_edge("citation_mapper", "verifier")
    graph.add_edge("verifier", "checker")

    graph.add_conditional_edges(
        "checker",
        route_after_checker,
        {
            "planner": "planner",
            "end": END,
        },
    )

    return graph.compile()


agent_graph = build_graph()
