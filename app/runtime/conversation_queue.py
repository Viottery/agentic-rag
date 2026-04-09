from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal, Protocol
from uuid import uuid4

from app.core.config import get_settings
from app.runtime.conversation_store import ConversationContextBundle, get_conversation_store


JobStatus = Literal["queued", "running", "finished", "failed"]
TurnRunner = Callable[[str, str, str, str], Awaitable[dict[str, Any]]]


class ConversationStoreLike(Protocol):
    def load_context_bundle(
        self,
        conversation_id: str,
        query: str = "",
    ) -> ConversationContextBundle: ...

    def save_turn_result(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        job_id: str,
        question: str,
        result: dict[str, Any],
    ) -> None: ...


class _NullConversationStore:
    def load_context_bundle(
        self,
        conversation_id: str,
        query: str = "",
    ) -> ConversationContextBundle:
        return ConversationContextBundle(
            conversation_id=conversation_id.strip(),
            messages=[],
            conversation_summary="",
            recent_turn_summaries=[],
            memory_notes=[],
        )

    def save_turn_result(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        job_id: str,
        question: str,
        result: dict[str, Any],
    ) -> None:
        return None


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


async def _default_turn_runner(
    question: str,
    conversation_id: str,
    turn_id: str,
    job_id: str,
) -> dict[str, Any]:
    from app.agent.graph import async_main_agent_graph, main_agent_graph
    from app.agent.state_factory import build_initial_agent_state

    settings = get_settings()
    store = get_conversation_store()
    context_bundle = await asyncio.to_thread(
        store.load_context_bundle,
        conversation_id,
        question,
    )
    initial_state = build_initial_agent_state(
        question,
        max_iterations=settings.agent_max_iterations,
        max_duration_seconds=settings.agent_max_duration_seconds,
        conversation_id=conversation_id,
        turn_id=turn_id,
        job_id=job_id,
        messages=context_bundle.messages,
        conversation_summary=context_bundle.conversation_summary,
        recent_turn_summaries=context_bundle.recent_turn_summaries,
        memory_notes=context_bundle.memory_notes,
    )

    if hasattr(async_main_agent_graph, "ainvoke"):
        return await async_main_agent_graph.ainvoke(initial_state)
    if hasattr(main_agent_graph, "ainvoke"):
        return await main_agent_graph.ainvoke(initial_state)
    return await asyncio.to_thread(main_agent_graph.invoke, initial_state)


@dataclass
class ConversationJob:
    job_id: str
    turn_id: str
    conversation_id: str
    question: str
    mode: Literal["wait", "background"]
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_iso_now)
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    result: dict[str, Any] | None = None
    _task: asyncio.Task[None] | None = field(default=None, repr=False, compare=False)

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }
        if self.result is not None:
            payload["result"] = self.result
        return payload


class ConversationQueueManager:
    """
    为 conversation 维持串行 turn 执行队列，同时允许多会话并发。

    现阶段先使用进程内内存队列：
    - 同一 conversation 通过 asyncio.Lock 串行
    - 不同 conversation 通过全局 semaphore 控制并发度
    - 后续可平滑替换为持久化 job store / external queue
    """

    def __init__(
        self,
        *,
        runner: TurnRunner | None = None,
        max_concurrent_conversations: int = 4,
        store: ConversationStoreLike | None = None,
    ) -> None:
        self._runner = runner or _default_turn_runner
        self._store = store or _NullConversationStore()
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._jobs: dict[str, ConversationJob] = {}
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_conversations))

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        lock = self._conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_id] = lock
        return lock

    def get_job(self, job_id: str) -> ConversationJob | None:
        return self._jobs.get(job_id)

    async def submit(
        self,
        *,
        question: str,
        conversation_id: str | None = None,
        mode: Literal["wait", "background"] = "wait",
    ) -> ConversationJob:
        resolved_conversation_id = (conversation_id or "").strip() or f"conv_{uuid4().hex}"
        job = ConversationJob(
            job_id=f"job_{uuid4().hex}",
            turn_id=f"turn_{uuid4().hex}",
            conversation_id=resolved_conversation_id,
            question=question,
            mode=mode,
        )
        self._jobs[job.job_id] = job

        if mode == "background":
            job._task = asyncio.create_task(self._run_job(job))
            return job

        await self._run_job(job)
        return job

    async def _run_job(self, job: ConversationJob) -> None:
        conversation_lock = self._lock_for(job.conversation_id)

        async with conversation_lock:
            async with self._semaphore:
                job.status = "running"
                job.started_at = _iso_now()

                try:
                    result = await self._runner(
                        job.question,
                        job.conversation_id,
                        job.turn_id,
                        job.job_id,
                    )
                    await asyncio.to_thread(
                        self._store.save_turn_result,
                        conversation_id=job.conversation_id,
                        turn_id=job.turn_id,
                        job_id=job.job_id,
                        question=job.question,
                        result=result,
                    )
                    job.result = {
                        **result,
                        "conversation_id": job.conversation_id,
                        "turn_id": job.turn_id,
                        "job_id": job.job_id,
                    }
                    job.status = "finished"
                except Exception as exc:  # noqa: BLE001
                    job.error = str(exc)
                    job.status = "failed"
                    await asyncio.to_thread(
                        self._store.save_turn_result,
                        conversation_id=job.conversation_id,
                        turn_id=job.turn_id,
                        job_id=job.job_id,
                        question=job.question,
                        result={
                            "status": "failed",
                            "error": job.error,
                            "started_at": job.started_at,
                            "finished_at": _iso_now(),
                            "intermediate_steps": [],
                            "trace_summary": "",
                        },
                    )
                finally:
                    job.finished_at = _iso_now()


_MANAGER: ConversationQueueManager | None = None


def get_conversation_queue_manager() -> ConversationQueueManager:
    global _MANAGER
    if _MANAGER is None:
        settings = get_settings()
        _MANAGER = ConversationQueueManager(
            max_concurrent_conversations=settings.agent_max_concurrent_conversations,
            store=get_conversation_store(),
        )
    return _MANAGER
