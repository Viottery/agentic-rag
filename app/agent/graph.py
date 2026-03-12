from langgraph.graph import END, StateGraph

from app.agent.nodes import planner, responder, retrieve, tool_executor
from app.agent.state import AgentState


def route_next_action(state: AgentState) -> str:
    """
    根据 planner 写入的 next_action 决定后续节点。
    """
    action = state["next_action"]

    if action == "retrieve":
        return "retrieve"
    if action == "tool":
        return "tool_executor"
    return "responder"


def build_graph():
    """
    构建当前阶段的 Agent 图。

    当前链路：
        planner
          ├─ retrieve -> responder -> END
          ├─ tool_executor -> responder -> END
          └─ responder -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("retrieve", retrieve)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("responder", responder)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_next_action,
        {
            "retrieve": "retrieve",
            "tool_executor": "tool_executor",
            "responder": "responder",
        },
    )

    graph.add_edge("retrieve", "responder")
    graph.add_edge("tool_executor", "responder")
    graph.add_edge("responder", END)

    return graph.compile()


agent_graph = build_graph()