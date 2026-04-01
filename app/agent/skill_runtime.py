from __future__ import annotations

from typing import Callable

from app.agent.state import AgentState


SkillHandler = Callable[[AgentState], AgentState]


def task_type_to_executor(task_type: str) -> str:
    mapping = {
        "rag": "local_kb_retrieve",
        "search": "web_search_retrieve",
        "action": "tool_execute",
    }
    return mapping.get(task_type.strip(), "local_kb_retrieve")


def execute_skill(state: AgentState, executor: str) -> AgentState:
    handlers: dict[str, SkillHandler] = {
        "local_kb_retrieve": _execute_local_kb_retrieve,
        "web_search_retrieve": _execute_web_search_retrieve,
        "tool_execute": _execute_tool_execute,
    }
    handler = handlers.get(executor)
    if handler is None:
        raise ValueError(f"unknown skill executor: {executor}")
    return handler(state)


def _execute_local_kb_retrieve(state: AgentState) -> AgentState:
    from app.agent import nodes

    return nodes.local_kb_retrieve_service(state)


def _execute_web_search_retrieve(state: AgentState) -> AgentState:
    from app.agent import nodes

    updated = nodes.query_refiner(state)
    updated = nodes.search_agent(updated)
    return updated


def _execute_tool_execute(state: AgentState) -> AgentState:
    from app.agent import nodes

    return nodes.action_agent(state)
