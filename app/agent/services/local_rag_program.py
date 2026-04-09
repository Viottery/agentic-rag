from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.agent import nodes
from app.agent.state_factory import build_initial_agent_state, build_subtask_initial_state
from app.runtime.conversation_store import get_conversation_store


class LocalRAGProgramRequest(BaseModel):
    question: str = Field(..., description="需要检索的用户问题")
    conversation_id: str = Field(default="", description="所属会话 ID，可选")
    max_duration_seconds: int = Field(default=90, description="本次检索的时间预算")


class LocalRAGProgramResponse(BaseModel):
    status: str = "failed"
    question: str = ""
    executor: str = "local_kb_retrieve"
    result: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieved_docs: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    rewritten_query: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    rewrite_reason: str = ""
    routed_source_name: str = ""
    routed_top_level_group: str = ""
    routed_hierarchy_scope: str = ""
    route_reason: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""
    degraded: bool = False
    degraded_reason: str = ""


def _model_validate(model_cls, payload: dict[str, Any]):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _build_service_subtask(question: str) -> dict[str, Any]:
    return {
        "task_id": "service-rag",
        "task_type": "rag",
        "executor": "local_kb_retrieve",
        "question": question,
        "success_criteria": "返回可直接支撑回答的本地证据。",
        "status": "pending",
        "result": "",
        "evidence": [],
        "sources": [],
        "error": "",
        "degraded": False,
        "degraded_reason": "",
        "rewritten_query": "",
        "sub_queries": [],
        "rewrite_reason": "",
        "routed_source_name": "",
        "routed_top_level_group": "",
        "routed_hierarchy_scope": "",
        "route_reason": "",
    }


def _to_response(state: dict[str, Any]) -> LocalRAGProgramResponse:
    task = state.get("current_task", {})
    return LocalRAGProgramResponse(
        status=task.get("status", state.get("status", "failed")),
        question=task.get("question", state.get("question", "")),
        result=task.get("result", ""),
        evidence=task.get("evidence", []),
        sources=task.get("sources", []),
        retrieved_docs=state.get("retrieved_docs", []),
        retrieved_sources=state.get("retrieved_sources", []),
        rewritten_query=task.get("rewritten_query", ""),
        sub_queries=task.get("sub_queries", []),
        rewrite_reason=task.get("rewrite_reason", ""),
        routed_source_name=task.get("routed_source_name", ""),
        routed_top_level_group=task.get("routed_top_level_group", ""),
        routed_hierarchy_scope=task.get("routed_hierarchy_scope", ""),
        route_reason=task.get("route_reason", ""),
        trace=state.get("intermediate_steps", []),
        error=task.get("error", "") or state.get("error", ""),
        degraded=bool(task.get("degraded", False)),
        degraded_reason=task.get("degraded_reason", ""),
    )


def run_local_rag_program(request: LocalRAGProgramRequest) -> LocalRAGProgramResponse:
    context_bundle = get_conversation_store().load_context_bundle(
        request.conversation_id,
        request.question,
    )
    parent_state = build_initial_agent_state(
        request.question,
        max_iterations=1,
        max_duration_seconds=request.max_duration_seconds,
        conversation_id=request.conversation_id,
        messages=context_bundle.messages,
        conversation_summary=context_bundle.conversation_summary,
        recent_turn_summaries=context_bundle.recent_turn_summaries,
        memory_notes=context_bundle.memory_notes,
    )
    subtask_state = build_subtask_initial_state(
        parent_state,
        _build_service_subtask(request.question),
    )
    updated = nodes.query_refiner(subtask_state)
    updated = nodes.rag_router(updated)
    updated = nodes.rag_agent(updated)
    return _to_response(updated)


async def run_local_rag_program_async(request: LocalRAGProgramRequest) -> LocalRAGProgramResponse:
    context_bundle = await asyncio.to_thread(
        get_conversation_store().load_context_bundle,
        request.conversation_id,
        request.question,
    )
    parent_state = build_initial_agent_state(
        request.question,
        max_iterations=1,
        max_duration_seconds=request.max_duration_seconds,
        conversation_id=request.conversation_id,
        messages=context_bundle.messages,
        conversation_summary=context_bundle.conversation_summary,
        recent_turn_summaries=context_bundle.recent_turn_summaries,
        memory_notes=context_bundle.memory_notes,
    )
    subtask_state = build_subtask_initial_state(
        parent_state,
        _build_service_subtask(request.question),
    )
    updated = await nodes.query_refiner_async(subtask_state)
    updated = await nodes.rag_router_async(updated)
    updated = await nodes.rag_agent_async(updated)
    return _to_response(updated)


def _load_request(path: str) -> LocalRAGProgramRequest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _model_validate(LocalRAGProgramRequest, payload)


def _write_response(response: LocalRAGProgramResponse, output_path: str | None) -> None:
    body = json.dumps(_model_dump(response), ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(body, encoding="utf-8")
        return
    print(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local KB retrieve program entry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    invoke_parser = subparsers.add_parser("invoke", help="Invoke local KB retrieve program")
    invoke_parser.add_argument("--request-file", required=True, help="Path to request JSON")
    invoke_parser.add_argument("--output-file", default="", help="Path to response JSON")
    invoke_parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Run the internal local rag pipeline through async node implementations.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    request = _load_request(args.request_file)
    if args.async_mode:
        response = asyncio.run(run_local_rag_program_async(request))
    else:
        response = run_local_rag_program(request)
    _write_response(response, args.output_file or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
