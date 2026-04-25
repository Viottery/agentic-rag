from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    answer_synthesizer,
    answer_synthesizer_async,
    execution_agent,
    execution_agent_async,
    fast_answer,
    fast_answer_async,
    fast_gate,
    planner,
    planner_async,
    task_dispatcher,
    validator,
    validator_async,
)
from app.agent.state import AgentState


def route_after_fast_gate(state: AgentState) -> str:
    mode = state["fast_path_decision"].get("mode", "planner_loop")

    if mode == "direct_answer":
        return "fast_answer"
    if mode == "single_skill":
        return "task_dispatcher"
    return "planner"


def route_after_planner(state: AgentState) -> str:
    decision = state["planner_control"].get("decision", "answer")

    if decision == "dispatch":
        return "task_dispatcher"
    return "answer_synthesizer"


def route_after_task_dispatcher(state: AgentState) -> str:
    current_task = state.get("current_task", {})
    if current_task.get("task_id", "").strip():
        return "execution_agent"
    return "answer_synthesizer"


def route_after_execution_agent(state: AgentState) -> str:
    mode = state["fast_path_decision"].get("mode", "planner_loop")
    if mode == "single_skill":
        return "answer_synthesizer"
    return "planner"


def route_after_validator(state: AgentState) -> str:
    if state["checker_result"].get("passed", False):
        return "end"
    return "planner"


# Backward-compatible aliases for older tests or imports.
route_after_skill_executor = route_after_execution_agent


def build_main_graph(*, async_mode: bool = False):
    """
    构建主编排图。

    当前主链路：
        fast_gate
          ├─ fast_answer -> END
          ├─ task_dispatcher -> execution_agent -> answer_synthesizer -> validator -> END / planner
          └─ planner -> task_dispatcher -> execution_agent -> planner
                    -> answer_synthesizer -> validator -> END / planner

    这里的 execution_agent 不再直接承载所有执行逻辑，
    而是负责启动单子任务执行图。这样后续要做多任务并发或异步调用时，
    只需要调整 execution_agent 的调度方式，而不必重写主图。
    """
    graph = StateGraph(AgentState)
    fast_answer_node = fast_answer_async if async_mode else fast_answer
    planner_node = planner_async if async_mode else planner
    execution_agent_node = execution_agent_async if async_mode else execution_agent
    answer_synthesizer_node = (
        answer_synthesizer_async if async_mode else answer_synthesizer
    )
    validator_node = validator_async if async_mode else validator

    graph.add_node("fast_gate", fast_gate)
    graph.add_node("fast_answer", fast_answer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("task_dispatcher", task_dispatcher)
    graph.add_node("execution_agent", execution_agent_node)
    graph.add_node("answer_synthesizer", answer_synthesizer_node)
    graph.add_node("validator", validator_node)

    graph.set_entry_point("fast_gate")

    graph.add_conditional_edges(
        "fast_gate",
        route_after_fast_gate,
        {
            "fast_answer": "fast_answer",
            "task_dispatcher": "task_dispatcher",
            "planner": "planner",
        },
    )

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "task_dispatcher": "task_dispatcher",
            "answer_synthesizer": "answer_synthesizer",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "task_dispatcher",
        route_after_task_dispatcher,
        {
            "execution_agent": "execution_agent",
            "answer_synthesizer": "answer_synthesizer",
        },
    )

    graph.add_conditional_edges(
        "execution_agent",
        route_after_execution_agent,
        {
            "answer_synthesizer": "answer_synthesizer",
            "planner": "planner",
        },
    )

    graph.add_edge("answer_synthesizer", "validator")
    graph.add_edge("fast_answer", END)

    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "planner": "planner",
            "end": END,
        },
    )

    return graph.compile()


build_graph = build_main_graph
main_agent_graph = build_main_graph()
async_main_agent_graph = build_main_graph(async_mode=True)
agent_graph = main_agent_graph
