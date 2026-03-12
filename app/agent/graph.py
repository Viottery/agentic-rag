from langgraph.graph import END, StateGraph

from app.agent.nodes import planner, responder
from app.agent.state import AgentState


def build_graph():
    """
    构建当前阶段的最小 Agent 图。

    当前链路：
        planner -> responder -> END

    虽然 state 已按 ReAct 方式设计，但此阶段先验证：
    - state 能否在节点间传递
    - graph 能否被 API 正常调用
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("responder", responder)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "responder")
    graph.add_edge("responder", END)

    return graph.compile()


agent_graph = build_graph()