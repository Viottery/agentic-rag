from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "pydantic_settings",
    types.SimpleNamespace(BaseSettings=object, SettingsConfigDict=lambda **kwargs: kwargs),
)
sys.modules.setdefault(
    "langchain_core.messages",
    types.SimpleNamespace(
        HumanMessage=type("HumanMessage", (), {"__init__": lambda self, content=None: setattr(self, "content", content)}),
        SystemMessage=type("SystemMessage", (), {"__init__": lambda self, content=None: setattr(self, "content", content)}),
    ),
)
sys.modules.setdefault(
    "langchain_openai",
    types.SimpleNamespace(ChatOpenAI=type("ChatOpenAI", (), {"__init__": lambda self, *args, **kwargs: None})),
)


class _FakeStateGraph:
    def __init__(self, _state_type) -> None:  # noqa: ANN001
        self.nodes: dict[str, object] = {}

    def add_node(self, name: str, node) -> None:  # noqa: ANN001
        self.nodes[name] = node

    def set_entry_point(self, _name: str) -> None:
        return None

    def add_conditional_edges(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def add_edge(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def compile(self):  # noqa: ANN201
        return self

    def invoke(self, state):  # noqa: ANN001, ANN201
        return state


sys.modules.setdefault(
    "langgraph.graph",
    types.SimpleNamespace(END="END", StateGraph=_FakeStateGraph),
)

import app.agent.graph as graph_module
import app.agent.nodes as nodes_module
import app.agent.subtask_graph as subtask_graph_module
from app.agent.state_factory import build_subtask_initial_state


def _base_state(question: str) -> dict:
    return {
        "question": question,
        "messages": [],
        "fast_path_decision": {
            "mode": "planner_loop",
            "reason": "",
            "executor": "",
            "question": "",
            "success_criteria": "",
        },
        "conversation_summary": "",
        "recent_turn_summaries": [],
        "memory_notes": [],
        "thought": "",
        "subtasks": [],
        "planner_control": {
            "decision": "dispatch",
            "selected_task_id": "",
            "planner_note": "",
            "checker_feedback": "",
            "force_answer_reason": "",
        },
        "current_task": {},
        "intermediate_steps": [],
        "execution_results": [],
        "skill_results": [],
        "kb_structure_summary": "",
        "aggregated_context": "",
        "evidence": [],
        "retrieved_docs": [],
        "retrieved_sources": [],
        "used_tools": [],
        "observation": "",
        "answer_draft": "",
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
        "checker_result": {
            "passed": False,
            "feedback": "",
            "pass_reason": "quality_fail",
        },
        "trace_summary": "",
        "started_at": "",
        "started_at_ts": 0.0,
        "finished_at": "",
        "iteration_count": 0,
        "max_iterations": 6,
        "max_duration_seconds": 90,
        "error": "",
        "answer": "",
        "status": "running",
    }


def test_fast_gate_routes_local_project_question_to_single_skill() -> None:
    state = _base_state("请根据知识库介绍一下这个项目的整体架构。")

    updated = nodes_module.fast_gate(state)

    assert updated["fast_path_decision"]["mode"] == "single_skill"
    assert updated["subtasks"][0]["executor"] == "local_kb_retrieve"
    assert updated["current_task"]["task_id"] == "fast-1"


def test_fast_gate_routes_simple_question_to_direct_answer() -> None:
    state = _base_state("什么是 HTTP？")

    updated = nodes_module.fast_gate(state)

    assert updated["fast_path_decision"]["mode"] == "direct_answer"
    assert updated["subtasks"] == []


def test_fast_gate_routes_explicit_command_to_single_tool() -> None:
    state = _base_state("请执行命令: printf ok，并返回命令输出。")

    updated = nodes_module.fast_gate(state)

    assert updated["fast_path_decision"]["mode"] == "single_skill"
    assert updated["fast_path_decision"]["executor"] == "tool_execute"
    assert updated["subtasks"][0]["executor"] == "tool_execute"


def test_build_subtask_initial_state_creates_isolated_single_task_runtime() -> None:
    parent_state = _base_state("请根据知识库介绍一下这个项目的整体架构。")
    parent_state["evidence"] = [
        {
            "source_type": "local_kb",
            "source_name": "kb",
            "source_id": "existing",
            "title": "existing",
            "content": "existing evidence",
            "score": 0.8,
            "metadata": {},
        }
    ]
    task = {
        "task_id": "t1",
        "task_type": "rag",
        "executor": "",
        "question": "项目架构",
        "success_criteria": "返回证据",
        "status": "pending",
    }

    subtask_state = build_subtask_initial_state(parent_state, task)

    assert subtask_state["current_task"]["task_id"] == "t1"
    assert subtask_state["current_task"]["executor"] == "local_kb_retrieve"
    assert subtask_state["subtasks"] == [subtask_state["current_task"]]
    assert subtask_state["evidence"] == []


def test_subtask_bootstrap_routes_local_kb_executor_to_program_service() -> None:
    parent_state = _base_state("请从知识库介绍一下真理。")
    task = {
        "task_id": "t-rag",
        "task_type": "rag",
        "executor": "local_kb_retrieve",
        "question": "介绍真理",
        "success_criteria": "返回证据",
        "status": "pending",
    }

    subtask_state = build_subtask_initial_state(parent_state, task)

    assert subtask_graph_module.route_after_bootstrap(subtask_state) == "local_kb_retrieve_service"


def test_execution_agent_records_structured_execution_result(monkeypatch) -> None:
    state = _base_state("请根据知识库介绍一下这个项目的整体架构。")
    state["fast_path_decision"]["mode"] = "single_skill"
    state["current_task"] = {
        "task_id": "t1",
        "task_type": "rag",
        "executor": "local_kb_retrieve",
        "question": "项目架构",
        "success_criteria": "返回证据",
        "status": "pending",
        "result": "",
        "evidence": [],
        "sources": [],
        "error": "",
    }
    state["subtasks"] = [state["current_task"]]

    def _fake_invoke_subtask_graph(subtask_state):  # noqa: ANN001, ANN201
        task = {
            **subtask_state["current_task"],
            "status": "done",
            "result": "项目采用 fast gate + planner loop。",
            "evidence": [
                {
                    "source_type": "local_kb",
                    "source_name": "kb",
                    "source_id": "chunk-1",
                    "title": "architecture",
                    "content": "项目采用 fast gate + planner loop。",
                    "score": 0.9,
                    "metadata": {},
                }
            ],
            "sources": ["chunk-1"],
            "error": "",
        }
        return {
            **subtask_state,
            "current_task": task,
            "subtasks": [task],
            "retrieved_docs": ["项目采用 fast gate + planner loop。"],
            "retrieved_sources": ["chunk-1"],
            "used_tools": [],
            "observation": "execution agent executed",
            "execution_result": {
                "task_id": "t1",
                "executor": "local_kb_retrieve",
                "status": "done",
                "summary": "项目采用 fast gate + planner loop。",
                "evidence_count": 1,
                "source_count": 1,
                "error": "",
                "evidence": task["evidence"],
                "sources": task["sources"],
                "retrieved_docs": ["项目采用 fast gate + planner loop。"],
                "retrieved_sources": ["chunk-1"],
                "used_tools": [],
                "degraded": False,
                "degraded_reason": "",
                "trace": [],
            },
        }

    monkeypatch.setattr(nodes_module, "_invoke_subtask_graph", _fake_invoke_subtask_graph)

    updated = nodes_module.execution_agent(state)

    assert updated["current_task"]["status"] == "done"
    assert updated["execution_results"][0]["executor"] == "local_kb_retrieve"
    assert updated["skill_results"][0]["executor"] == "local_kb_retrieve"
    assert updated["skill_results"][0]["evidence_count"] == 1


def test_task_dispatcher_accepts_fast_single_skill_task_without_planner_selection() -> None:
    state = _base_state("请根据知识库介绍一下这个项目的整体架构。")
    state["fast_path_decision"]["mode"] = "single_skill"
    state["current_task"] = {
        "task_id": "fast-1",
        "task_type": "rag",
        "executor": "local_kb_retrieve",
        "question": "项目架构",
        "success_criteria": "返回证据",
        "status": "pending",
    }
    state["subtasks"] = [state["current_task"]]

    updated = nodes_module.task_dispatcher(state)

    assert updated["current_task"]["task_id"] == "fast-1"
    assert updated["current_task"]["status"] == "running"


def test_subtask_graph_routes_are_executor_specific() -> None:
    assert subtask_graph_module.route_after_bootstrap(
        {"current_task": {"task_type": "rag", "executor": "local_kb_retrieve"}}
    ) == "local_kb_retrieve_service"
    assert subtask_graph_module.route_after_bootstrap(
        {"current_task": {"task_type": "action", "executor": "tool_execute"}}
    ) == "action_agent"
    assert subtask_graph_module.route_after_query_refiner(
        {"current_task": {"task_type": "search", "executor": "web_search_retrieve"}}
    ) == "search_agent"


def test_action_agent_executes_shell_runtime_when_plan_requests_shell(monkeypatch) -> None:
    state = _base_state("请执行一个简单操作：把字符串 'agentic rag' 转成大写并返回结果。")
    state["current_task"] = {
        "task_id": "a1",
        "task_type": "action",
        "executor": "tool_execute",
        "question": "把字符串 'agentic rag' 转成大写并返回结果。",
        "success_criteria": "返回转换结果",
        "status": "running",
        "result": "",
        "evidence": [],
        "sources": [],
        "error": "",
    }
    state["subtasks"] = [state["current_task"]]

    monkeypatch.setattr(
        nodes_module,
        "_plan_tool_execution",
        lambda _state: types.SimpleNamespace(
            mode="shell",
            command="python - <<'PY'\nprint('AGENTIC RAG')\nPY",
            response_text="",
            rationale="deterministic shell transform",
        ),
    )
    monkeypatch.setattr(
        nodes_module,
        "run_shell_command",
        lambda _command: types.SimpleNamespace(
            allowed=True,
            command="python - <<'PY' print('AGENTIC RAG') PY",
            cwd="/tmp",
            exit_code=0,
            stdout="AGENTIC RAG\n",
            stderr="",
            duration_seconds=0.01,
            risk_level="low",
            policy_reason="allowed",
            truncated=False,
        ),
    )

    updated = nodes_module.action_agent(state)

    assert updated["current_task"]["status"] == "done"
    assert updated["current_task"]["degraded"] is False
    assert updated["used_tools"][-1] == "shell_runtime"
    assert "AGENTIC RAG" in updated["current_task"]["result"]


def test_explicit_command_uses_deterministic_shell_plan() -> None:
    state = _base_state("请执行命令: printf ok")
    state["current_task"] = {
        "task_id": "a1",
        "task_type": "action",
        "executor": "tool_execute",
        "question": "执行命令: printf ok",
        "success_criteria": "返回命令输出",
        "status": "running",
        "result": "",
        "evidence": [],
        "sources": [],
        "error": "",
    }

    plan = nodes_module._plan_tool_execution(state)

    assert plan.mode == "shell"
    assert plan.command == "printf ok"


def test_explicit_command_plan_strips_return_output_suffix() -> None:
    state = _base_state("请执行命令: printf ok，并返回命令输出。")
    state["current_task"] = {
        "task_id": "a1",
        "task_type": "action",
        "executor": "tool_execute",
        "question": "执行命令: printf ok，并返回命令输出。",
        "success_criteria": "返回命令输出",
        "status": "running",
        "result": "",
        "evidence": [],
        "sources": [],
        "error": "",
    }

    plan = nodes_module._plan_tool_execution(state)

    assert plan.mode == "shell"
    assert plan.command == "printf ok"


def test_terminal_tool_result_is_rendered_and_checker_finishes() -> None:
    state = _base_state("执行命令: printf ok")
    state["subtasks"] = [
        {
            "task_id": "execute_shell_command",
            "task_type": "action",
            "executor": "tool_execute",
            "question": "执行命令: printf ok",
            "success_criteria": "返回命令输出",
            "status": "done",
            "result": "shell command: printf ok\nexit_code: 0\nstdout:\nok\n\nstderr:\n(empty)",
            "evidence": [],
            "sources": ["shell_runtime"],
            "error": "",
            "degraded": False,
            "degraded_reason": "",
        }
    ]
    state["used_tools"] = ["shell_runtime"]
    state["evidence"] = [
        {
            "source_type": "tool",
            "source_name": "shell_runtime",
            "source_id": "execute_shell_command_shell",
            "title": "Shell Runtime Result",
            "content": "shell command: printf ok\nexit_code: 0\nstdout:\nok\n\nstderr:\n(empty)",
            "score": 1.0,
            "metadata": {
                "command": "printf ok",
                "exit_code": 0,
                "policy_reason": "allowed",
                "degraded": False,
            },
        }
    ]

    answered = nodes_module.answer_generator(state)
    checked = nodes_module.checker(answered)

    assert "printf ok" in answered["answer_draft"]
    assert checked["status"] == "finished"
    assert checked["checker_result"]["pass_reason"] == "terminal_tool_pass"
    assert "stdout" in checked["answer"]


def test_graph_routes_use_main_and_subtask_boundaries() -> None:
    assert graph_module.route_after_fast_gate({"fast_path_decision": {"mode": "direct_answer"}}) == "fast_answer"
    assert graph_module.route_after_fast_gate({"fast_path_decision": {"mode": "single_skill"}}) == "task_dispatcher"
    assert graph_module.route_after_planner({"planner_control": {"decision": "dispatch"}}) == "task_dispatcher"
    assert graph_module.route_after_planner({"planner_control": {"decision": "finish"}}) == "answer_synthesizer"
    assert graph_module.route_after_task_dispatcher({"current_task": {"task_id": "t1"}}) == "execution_agent"
    assert graph_module.route_after_execution_agent({"fast_path_decision": {"mode": "single_skill"}}) == "answer_synthesizer"
    assert graph_module.route_after_validator({"checker_result": {"passed": False}}) == "planner"
