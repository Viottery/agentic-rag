from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    action_agent,
    answer_generator,
    checker,
    dispatcher,
    planner,
    rag_agent,
    search_agent,
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
        return "rag_agent"
    if task_type == "search":
        return "search_agent"
    if task_type == "action":
        return "action_agent"
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
          ├─ dispatcher -> rag_agent    -> planner
          ├─ dispatcher -> search_agent -> planner
          ├─ dispatcher -> action_agent -> planner
          ├─ answer_generator -> checker -> END
          └─ answer_generator -> checker -> planner
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("dispatcher", dispatcher)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("search_agent", search_agent)
    graph.add_node("action_agent", action_agent)
    graph.add_node("answer_generator", answer_generator)
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
            "rag_agent": "rag_agent",
            "search_agent": "search_agent",
            "action_agent": "action_agent",
            "planner": "planner",
        },
    )

    graph.add_edge("rag_agent", "planner")
    graph.add_edge("search_agent", "planner")
    graph.add_edge("action_agent", "planner")
    graph.add_edge("answer_generator", "checker")

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
